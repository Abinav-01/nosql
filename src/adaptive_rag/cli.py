import argparse
import json
import sys
from .config import Settings
from .core import RuleRouter
from .data import download, prepare, read_jsonl, make_pilot


def make_pipeline(settings, mode="adaptive", router_name="rules"):
    from .stores import VectorStore, GraphStore
    from .llm import Ollama
    from .pipeline import Pipeline
    vector = VectorStore(settings) if mode != "graph" else None
    graph = None
    if mode != "vector":
        from .stores import corpus_key
        manifest_path = settings.data_root / "graph_index.json"
        if not manifest_path.exists():
            raise RuntimeError("Graph index is not complete. Finish extraction and run index --store graph.")
        manifest = json.loads(manifest_path.read_text())
        if manifest["corpus"] != corpus_key(settings) or not manifest["complete"]:
            raise RuntimeError("Graph index does not cover the current corpus. Rebuild it before comparisons.")
        graph = GraphStore(settings)
    router = RuleRouter()
    if router_name == "learned":
        from .router import NaiveBayesRouter
        training_path = settings.data_root / "development.jsonl"
        if not training_path.exists():
            raise RuntimeError("Learned router requires development.jsonl with expected_route labels")
        router = NaiveBayesRouter().fit(read_jsonl(training_path))
    return Pipeline(vector, graph, Ollama(settings), settings.top_k, router=router)


def main():
    parser = argparse.ArgumentParser(description="Adaptive vector–graph RAG research prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    route = sub.add_parser("route", help="Inspect a rule decision without loading any model")
    route.add_argument("question")
    explain_router = sub.add_parser("explain-router", help="Show named features and both router decisions")
    explain_router.add_argument("question")
    train_router = sub.add_parser("train-router", help="Compare a lightweight learned router with rules")
    train_router.add_argument("--source", default="development.jsonl")
    data = sub.add_parser("prepare", help="Download and prepare public pilot datasets")
    data.add_argument("--per-dataset", type=int, default=150)
    data.add_argument("--seed", type=int, default=42)
    pilot = sub.add_parser("pilot", help="Create a reproducible small pilot corpus")
    pilot.add_argument("--chunks", type=int, default=200)
    pilot.add_argument("--seed", type=int, default=42)
    synthetic = sub.add_parser("synthetic", help="Create a deterministic demo corpus")
    extract = sub.add_parser("extract", help="Extract source-backed graph facts using local Ollama")
    extract.add_argument("--limit", type=int)
    index = sub.add_parser("index")
    index.add_argument("--store", choices=["vector", "graph", "both"], default="both")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=["adaptive", "vector", "graph", "static_hybrid"], default="adaptive")
    ask.add_argument("--retrieval-only", action="store_true")
    ask.add_argument("--fallback", action="store_true")
    ask.add_argument("--router", choices=["rules", "learned"], default="rules")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--repeats", type=int, default=3)
    evaluation.add_argument("--generate", action="store_true")
    evaluation.add_argument("--name", default="pilot")
    evaluation.add_argument("--router", choices=["rules", "learned"], default="rules")
    sub.add_parser("doctor", help="Check local service availability and prepared data")
    args = parser.parse_args()
    if args.command == "route":
        from dataclasses import asdict
        print(json.dumps(asdict(RuleRouter().route(args.question)), indent=2))
        return
    if args.command == "explain-router":
        from dataclasses import asdict
        from .router import extract_features
        result = {"question": args.question, "features": extract_features(args.question),
                  "rule_decision": asdict(RuleRouter().route(args.question))}
        print(json.dumps(result, indent=2))
        return
    settings = Settings.load()
    processed = settings.data_root
    if args.command == "prepare":
        inputs = []
        for dataset in ("squad", "hotpot"):
            for source_split, split in (("train", "development"), ("dev", "evaluation")):
                name = dataset + "_" + source_split
                print(f"Preparing {name}", flush=True)
                inputs.append((dataset, split, download(name, settings.root / "data/raw")))
        print(json.dumps(prepare(inputs, processed, args.per_dataset, args.seed), indent=2))
    elif args.command == "train-router":
        from .router import compare
        rows = read_jsonl(processed / args.source)
        result = compare(rows)
        (processed / "router_comparison.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    elif args.command == "pilot":
        target = settings.root / "data/pilot_200"
        if target.exists():
            raise ValueError("data/pilot_200 already exists; use the existing pilot or remove it before recreating")
        print(json.dumps(make_pilot(processed, target, args.chunks, args.seed), indent=2))
    elif args.command == "synthetic":
        from .synthetic import build
        target = settings.root / "data/synthetic_100_v2"
        if target.exists():
            raise ValueError("data/synthetic_100_v2 already exists; use the existing corpus or remove it before recreating")
        print(json.dumps(build(target), indent=2))
    elif args.command == "extract":
        from .stores import extract_corpus
        from .llm import Ollama
        extract_corpus(settings, Ollama(settings), args.limit)
    elif args.command == "index":
        from .stores import VectorStore, GraphStore
        chunks = read_jsonl(processed / "chunks.jsonl")
        if args.store in ("vector", "both"):
            VectorStore(settings).index(chunks)
            print(f"Vector index: {len(chunks)} chunks")
        if args.store in ("graph", "both"):
            extracted = read_jsonl(processed / "extractions.jsonl")
            facts = {r["chunk_id"]: r["triples"] for r in extracted}
            missing = set(c["id"] for c in chunks) - facts.keys()
            if missing:
                raise ValueError(f"Extraction incomplete: {len(missing)} chunks missing; finish extract first")
            graph = GraphStore(settings)
            try:
                graph.index(chunks, facts)
                from .stores import corpus_key
                (processed / "graph_index.json").write_text(json.dumps({
                    "corpus": corpus_key(settings), "complete": True,
                    "chunks": len(chunks), "triples": sum(len(facts[c["id"]]) for c in chunks)
                }, indent=2))
            finally:
                graph.close()
            print(f"Graph index: {sum(len(v) for v in facts.values())} source-backed triples")
    elif args.command in ("ask", "evaluate"):
        pipeline = make_pipeline(settings, getattr(args, "mode", "adaptive"), args.router)
        try:
            if args.command == "ask":
                result = pipeline.run(args.question, args.mode, not args.retrieval_only, args.fallback)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                if result["status"] == "error":
                    sys.exit(1)
            else:
                from .evaluate import benchmark
                if not args.name.replace("-", "").replace("_", "").isalnum():
                    raise ValueError("Run name must contain letters, digits, hyphens or underscores")
                output = settings.root / "runs" / args.name
                if output.exists():
                    raise ValueError("Run already exists; choose a new --name to preserve previous results")
                print(json.dumps(benchmark(pipeline, processed / "evaluation.jsonl", output,
                                           args.generate, args.repeats, args.limit), indent=2))
        finally:
            if pipeline.graph:
                pipeline.graph.close()
    elif args.command == "doctor":
        import requests
        checks = {"prepared_corpus": (processed / "chunks.jsonl").exists(),
                  "graph_extractions": (processed / "extractions.jsonl").exists()}
        try:
            response = requests.get(settings.ollama_url + "/api/tags", timeout=5)
            response.raise_for_status()
            checks["ollama"] = [m["name"] for m in response.json().get("models", [])]
        except Exception as exc:
            checks["ollama"] = str(exc)
        try:
            from .stores import GraphStore
            graph = GraphStore(settings)
            graph.close()
            checks["neo4j"] = True
        except Exception as exc:
            checks["neo4j"] = str(exc)
        print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
