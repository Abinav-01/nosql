from dataclasses import dataclass, asdict
import re


@dataclass(frozen=True)
class Decision:
    route: str
    reason: str
    rule: str
    # Rules do not produce calibrated probabilities.
    confidence: float | None = None


class RuleRouter:
    version = "rules-v1"
    relational = re.compile(
        r"\b(spouse|married|parent|father|mother|sibling|child|children|"
        r"born|birthplace|founded|founder|directed|director|author|wrote|"
        r"member|belongs|located|capital|connected|relationship|"
        r"technology|uses|use|company|requires|"
        r"older|younger|earlier|later|same|both)\b", re.I)
    explanatory = re.compile(r"\b(why|how|explain|describe|impact|influence|significance)\b", re.I)

    def route(self, query: str) -> Decision:
        if not query.strip():
            raise ValueError("Enter a non-empty question")
        relation = bool(self.relational.search(query))
        explanation = bool(self.explanatory.search(query))
        if relation and explanation:
            return Decision("hybrid", "Relationship and explanatory language detected", "relation+explanation")
        if relation:
            return Decision("graph", "Explicit relationship or comparison language detected", "relation")
        if explanation:
            return Decision("vector", "Explanatory text requested without a relationship cue", "explanation")
        return Decision("vector", "No explicit relationship cue; start with text retrieval", "default")


@dataclass
class Hit:
    chunk_id: str
    text: str
    title: str
    score: float
    sources: list[str]
    paths: list[str]

    def to_dict(self):
        return asdict(self)


def rrf(rankings: list[list[Hit]], limit: int = 5, constant: int = 60) -> list[Hit]:
    """Fuse evidence IDs, not incomparable vector distances and graph scores."""
    if constant < 1 or limit < 1:
        raise ValueError("RRF constant and limit must be positive")
    merged: dict[str, Hit] = {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            if hit.chunk_id not in merged:
                merged[hit.chunk_id] = Hit(hit.chunk_id, hit.text, hit.title, 0.0, [], [])
            target = merged[hit.chunk_id]
            target.score += 1 / (constant + rank)
            target.sources = sorted(set(target.sources + hit.sources))
            target.paths = sorted(set(target.paths + hit.paths))
    return sorted(merged.values(), key=lambda h: (-h.score, h.chunk_id))[:limit]


def retrieval_metrics(retrieved: list[str], gold: list[str]) -> dict:
    found, relevant = set(retrieved), set(gold)
    if not relevant:
        return {"precision": None, "recall": None}
    overlap = len(found & relevant)
    return {"precision": overlap / len(found) if found else 0.0,
            "recall": overlap / len(relevant)}
