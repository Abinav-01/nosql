import hashlib
import json
import re
from .core import Hit
from .data import read_jsonl


def corpus_key(settings):
    path = settings.data_root / "chunks.jsonl"
    if not path.exists():
        raise RuntimeError("Prepare the corpus before initializing retrieval")
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class VectorStore:
    def __init__(self, settings):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(settings.embedding_model,
                                         cache_folder=str(settings.root / "data/models"))
        self.client = chromadb.PersistentClient(path=str(settings.root / "data/chroma"),
                                                settings=ChromaSettings(anonymized_telemetry=False))
        model_key = hashlib.sha256(settings.embedding_model.encode()).hexdigest()[:10]
        self.collection = self.client.get_or_create_collection("evidence_" + model_key + "_" + corpus_key(settings))
        # Read index size during initialization, outside per-question call accounting.
        self.count = self.collection.count()

    def index(self, chunks):
        for offset in range(0, len(chunks), 64):
            batch = chunks[offset:offset + 64]
            embeddings = self.model.encode([c["text"] for c in batch], normalize_embeddings=True).tolist()
            self.collection.upsert(ids=[c["id"] for c in batch], embeddings=embeddings,
                                   documents=[c["text"] for c in batch],
                                   metadatas=[{"title": c["title"], "dataset": c["dataset"]} for c in batch])
        self.count = self.collection.count()

    def search(self, query, limit):
        count = self.count
        if not count:
            raise RuntimeError("Vector index is empty. Run index first.")
        result = self.collection.query(query_embeddings=self.model.encode([query], normalize_embeddings=True).tolist(),
                                       n_results=min(limit, count))
        return [Hit(cid, text, meta["title"], -distance, ["vector"], [])
                for cid, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                                     result["metadatas"][0], result["distances"][0])]


class GraphStore:
    def __init__(self, settings):
        from neo4j import GraphDatabase
        if not settings.neo4j_password:
            raise RuntimeError("Set NEO4J_PASSWORD in .env before connecting to Neo4j")
        self.driver = GraphDatabase.driver(settings.neo4j_uri,
                                           auth=(settings.neo4j_user, settings.neo4j_password),
                                           connection_timeout=10)
        self.driver.verify_connectivity()
        self.hops = settings.graph_hops
        self.corpus = corpus_key(settings)

    def close(self):
        self.driver.close()

    def index(self, chunks, extractions):
        for label in ("Evidence", "Entity", "Fact"):
            self.driver.execute_query(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                                      f"FOR (n:{label}) REQUIRE n.id IS UNIQUE")
        for chunk in chunks:
            self.driver.execute_query(
                "MERGE (c:Evidence {id:$id}) SET c.text=$text, c.title=$title "
                "SET c.corpora = CASE WHEN $corpus IN coalesce(c.corpora, []) "
                "THEN c.corpora ELSE coalesce(c.corpora, []) + $corpus END",
                id=chunk["id"], text=chunk["text"], title=chunk["title"], corpus=self.corpus)
            for triple in extractions.get(chunk["id"], []):
                subject = " ".join(triple["subject"].casefold().split())
                object_ = " ".join(triple["object"].casefold().split())
                fact = hashlib.sha256(json.dumps([subject, triple["relation"].casefold(), object_]).encode()).hexdigest()
                # Fact nodes avoid executing LLM-produced relationship types as Cypher.
                self.driver.execute_query(
                    "MATCH (c:Evidence {id:$chunk}) "
                    "MERGE (s:Entity {id:$subject}) SET s.name=$subject_name "
                    "MERGE (o:Entity {id:$object}) SET o.name=$object_name "
                    "MERGE (f:Fact {id:$fact}) SET f.relation=$relation "
                    "MERGE (s)-[:SUBJECT_OF]->(f) MERGE (f)-[:OBJECT]->(o) "
                    "MERGE (f)-[:SUPPORTED_BY]->(c)",
                    chunk=chunk["id"], subject=subject, object=object_,
                    subject_name=triple["subject"], object_name=triple["object"],
                    fact=fact, relation=triple["relation"])

    def search(self, query, limit):
        # Local phrase extraction, followed by one parameterized DB operation.
        words = re.findall(r"[\w'-]+", query.casefold())
        phrases = sorted({" ".join(words[i:i+n]) for n in range(1, 7)
                          for i in range(len(words) - n + 1)})
        # Two graph edges represent one semantic hop through a Fact node.
        rows, _, _ = self.driver.execute_query(
            "MATCH (seed:Entity) WHERE seed.id IN $phrases "
            f"MATCH p=(seed)-[:SUBJECT_OF|OBJECT*1..{2 * self.hops}]-(end) "
            "WITH p, [n IN nodes(p) WHERE n:Fact] AS facts "
            "WHERE size(facts)>0 "
            "UNWIND facts AS f MATCH (s:Entity)-[:SUBJECT_OF]->(f)-[:OBJECT]->(o:Entity) "
            "MATCH (f)-[:SUPPORTED_BY]->(c:Evidence) "
            "WHERE $corpus IN coalesce(c.corpora, []) "
            "WITH c, collect(DISTINCT s.name + ' — ' + f.relation + ' → ' + o.name) AS paths, "
            "min(length(p)) AS distance "
            "RETURN c.id AS id, c.text AS text, c.title AS title, paths, distance "
            "ORDER BY distance, id LIMIT $limit",
            phrases=phrases, limit=limit, corpus=self.corpus)
        return [Hit(r["id"], r["text"], r["title"], -r["distance"], ["graph"], r["paths"]) for r in rows]


def extract_corpus(settings, llm, limit=None):
    chunks = read_jsonl(settings.data_root / "chunks.jsonl")
    if limit:
        chunks = chunks[:limit]
    path = settings.data_root / "extractions.jsonl"
    existing = {r["chunk_id"]: r for r in read_jsonl(path)} if path.exists() else {}
    # Append after each successful extraction: resumable after service interruption.
    target_count = min(limit, len(chunks)) if limit else len(chunks)
    completed = len(existing)
    with path.open("a") as stream:
        for index, chunk in enumerate(chunks, 1):
            if chunk["id"] in existing:
                continue
            try:
                triples, extraction_error = llm.extract(chunk)
            except Exception as exc:
                # Keep the partial file valid and let a later invocation resume.
                print(f"Extraction stopped at {index}/{target_count}: {exc}", flush=True)
                raise
            record = {"chunk_id": chunk["id"], "triples": triples,
                      "model": settings.ollama_model}
            if extraction_error:
                record["extraction_error"] = extraction_error
                print(f"Warning {completed + 1}/{target_count}: {extraction_error}", flush=True)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            existing[chunk["id"]] = True
            completed += 1
            print(f"Extracted {completed}/{target_count}: {len(triples)} supported triples", flush=True)
