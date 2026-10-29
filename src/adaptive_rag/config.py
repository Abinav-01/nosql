from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    top_k: int = 5
    graph_hops: int = 2
    data_root_name: str = "data/processed"

    @classmethod
    def load(cls):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        obj = cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            neo4j_uri=os.getenv("NEO4J_URI", cls.neo4j_uri),
            neo4j_user=os.getenv("NEO4J_USER", cls.neo4j_user),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            top_k=int(os.getenv("TOP_K", "5")),
            graph_hops=int(os.getenv("GRAPH_HOPS", "2")),
            data_root_name=os.getenv("ADAPTIVE_RAG_DATA_ROOT", "data/processed"),
        )
        if not 1 <= obj.graph_hops <= 3 or obj.top_k < 1:
            raise ValueError("GRAPH_HOPS must be 1–3 and TOP_K must be positive")
        return obj

    @property
    def data_root(self):
        return self.root / self.data_root_name
