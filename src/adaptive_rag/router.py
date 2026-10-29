"""Small, dependency-free learned router used for comparison experiments."""
import math
import re
from collections import Counter, defaultdict
from .core import RuleRouter, Decision

TOKEN = re.compile(r"[a-z0-9]+")
RELATION_CUES = {"use", "uses", "technology", "supports", "supervises", "belongs", "connected", "founded", "directed", "relationship"}
EXPLANATION_CUES = {"why", "how", "explain", "describe", "impact", "influence", "significance"}
COMPARISON_CUES = {"older", "younger", "earlier", "later", "same", "both", "difference", "compare"}

def extract_features(query):
    """Named, inspectable intent features; does not change the production router."""
    tokens = TOKEN.findall(query.casefold())
    token_set = set(tokens)
    relation = sorted(token_set & RELATION_CUES)
    explanation = sorted(token_set & EXPLANATION_CUES)
    comparison = sorted(token_set & COMPARISON_CUES)
    question_word = next((t for t in tokens if t in {"who", "what", "which", "where", "when", "why", "how"}), None)
    capitalized_entities = re.findall(r"\b[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+)*", query)
    return {"question_word": question_word, "relation_cues": relation,
            "explanation_cues": explanation, "comparison_cues": comparison,
            "entity_count": len(capitalized_entities), "token_count": len(tokens),
            "has_relation": bool(relation or comparison), "has_explanation": bool(explanation)}

class NaiveBayesRouter:
    version = "nb-v1"
    def __init__(self):
        self.classes = []
        self.priors = Counter()
        self.words = defaultdict(Counter)
        self.total = Counter()
        self.vocabulary = set()

    def fit(self, rows):
        for row in rows:
            label = row.get("expected_route") or row.get("category")
            if label not in {"vector", "graph", "hybrid"}:
                continue
            tokens = TOKEN.findall(row["question"].casefold())
            self.priors[label] += 1
            self.words[label].update(tokens)
            self.total[label] += len(tokens)
            self.vocabulary.update(tokens)
        self.classes = sorted(self.priors)
        if not self.classes:
            raise ValueError("Training data has no route labels")
        return self

    def route(self, query):
        if not self.classes:
            raise ValueError("Router must be trained before use")
        tokens = TOKEN.findall(query.casefold())
        scores = {}
        vocab = max(1, len(self.vocabulary))
        total_rows = sum(self.priors.values())
        for label in self.classes:
            score = math.log(self.priors[label] / total_rows)
            denominator = self.total[label] + vocab
            for token in tokens:
                score += math.log((self.words[label][token] + 1) / denominator)
            scores[label] = score
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        route, best = ordered[0]
        confidence = 1 / (1 + math.exp(-(best - ordered[1][1]))) if len(ordered) > 1 else 1.0
        features = extract_features(query)
        reason = (f"learned score; relation cues={features['relation_cues']}, "
                  f"explanation cues={features['explanation_cues']}, "
                  f"question word={features['question_word']}")
        return Decision(route, reason, "learned", confidence)

def compare(rows):
    model = NaiveBayesRouter().fit(rows)
    rule = RuleRouter()
    learned_correct = rule_correct = total = 0
    for row in rows:
        label = row.get("expected_route") or row.get("category")
        if label not in {"vector", "graph", "hybrid"}: continue
        total += 1
        learned_correct += model.route(row["question"]).route == label
        rule_correct += rule.route(row["question"]).route == label
    return {"samples": total, "learned_accuracy": learned_correct / total,
            "rule_accuracy": rule_correct / total, "learned_router": model.version,
            "rule_router": rule.version}
