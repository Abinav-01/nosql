from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import random
import re
import string
from statistics import mean, median
from .core import retrieval_metrics
from .data import read_jsonl, write_jsonl
from .pipeline import MODES


def normalize(text):
    text = text.casefold().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_metrics(answer, references):
    if answer is None:
        return {"exact_match": None, "answer_token_f1": None,
                "faithfulness": None, "answer_relevance": None}
    # Citations are formatting, not answer content; remaining prose still counts.
    answer = normalize(re.sub(r"\[\d+\]", "", answer))
    f1, exact = 0.0, 0.0
    for reference in references:
        reference = normalize(reference)
        exact = max(exact, float(answer == reference))
        a, b = Counter(answer.split()), Counter(reference.split())
        overlap = sum((a & b).values())
        if not a and not b:
            f1 = 1.0
        elif overlap:
            precision, recall = overlap / sum(a.values()), overlap / sum(b.values())
            f1 = max(f1, 2 * precision * recall / (precision + recall))
    return {"exact_match": exact, "answer_token_f1": f1,
            "faithfulness": None, "answer_relevance": None}


def ragas_style_metrics(question, answer, hits, references):
    """Deterministic local proxy for RAGAS-style checks.

    It is deliberately labeled as a proxy: faithfulness is answer-token coverage
    by retrieved context; relevance is question-token overlap with the answer/context.
    Install/use a full RAGAS judge separately when an LLM evaluation budget is available.
    """
    if not answer or not hits:
        return {"faithfulness": None, "answer_relevance": None}
    answer_tokens = set(normalize(answer).split())
    context_tokens = set(normalize(" ".join(h["text"] for h in hits)).split())
    question_tokens = set(normalize(question).split()) - {"what", "which", "how", "does", "the", "is"}
    faith = len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0
    relevance = len(question_tokens & (context_tokens | answer_tokens)) / len(question_tokens) if question_tokens else 1.0
    return {"faithfulness": faith, "answer_relevance": relevance}


def confidence_interval(values, z=1.96):
    values = [float(v) for v in values if v is not None]
    if not values: return {"mean": None, "low": None, "high": None, "n": 0}
    avg = mean(values)
    if len(values) < 2: return {"mean": avg, "low": avg, "high": avg, "n": len(values)}
    variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
    margin = z * (variance ** 0.5) / (len(values) ** 0.5)
    return {"mean": avg, "low": avg - margin, "high": avg + margin, "n": len(values)}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for group in ("overall", "dataset:" + row["dataset"], "category:" + row["category"]):
            groups[(group, row["mode"])].append(row)
    summaries = []
    for (group, mode), records in sorted(groups.items()):
        # Failed runs remain in failure rates and latency/call statistics, not successful quality averages.
        good = [r for r in records if r["status"] != "error"]
        summary = {"group": group, "mode": mode, "runs": len(records),
                   "error_rate": mean(r["status"] == "error" for r in records),
                   "no_evidence_rate": mean(not r["hits"] for r in records),
                   "fallback_rate": mean(r["fallback_used"] for r in records),
                   "median_total_ms": median(r["total_ms"] for r in records),
                   "mean_database_calls": mean(sum(r["database_calls"].values()) for r in records)}
        for key in ("precision", "recall", "exact_match", "answer_token_f1"):
            values = [r[key] for r in good if r.get(key) is not None]
            summary[key] = mean(values) if values else None
        for key in ("faithfulness", "answer_relevance"):
            values = [r[key] for r in good if r.get(key) is not None]
            summary[key] = mean(values) if values else None
        summary["confidence_intervals"] = {key: confidence_interval([r.get(key) for r in good])
                                            for key in ("precision", "recall", "median_total_ms")}
        summaries.append(summary)
    return summaries


def benchmark(pipeline, questions_path, output, generate=False, repeats=1, limit=None, seed=42):
    questions = read_jsonl(questions_path)
    if limit:
        questions = questions[:limit]
    if not questions:
        raise ValueError("Evaluation set is empty")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rng, rows = random.Random(seed), []
    output.mkdir(parents=True, exist_ok=True)
    # Every mode on every question, randomized order to reduce timing-order effects.
    for question in questions:
        for repeat in range(repeats):
            modes = list(MODES)
            rng.shuffle(modes)
            for mode in modes:
                result = pipeline.run(question["question"], mode, generate=generate, fallback=False)
                result.update({"question_id": question["id"], "dataset": question["dataset"],
                               "category": question["category"], "repeat": repeat})
                result.update(retrieval_metrics([h["chunk_id"] for h in result["hits"]], question["gold_chunk_ids"]))
                result.update(answer_metrics(result["answer"], question["answers"]))
                result.update(ragas_style_metrics(question["question"], result["answer"], result["hits"], question["answers"]))
                rows.append(result)
                print(f"{len(rows)}: {question['id']} / {mode} / {result['status']}", flush=True)
                write_jsonl(output / "results.jsonl", rows)
    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "run.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "questions": len(questions), "repeats": repeats, "generation": generate,
        "router_version": pipeline.router.version, "fallback": False,
        "quality_note": "faithfulness and answer_relevance are deterministic RAGAS-style proxies unless an external LLM judge is added; EM/F1 penalizes explanatory prose.",
        "timing_note": "Model/database initialization excluded. Initial query may include cold-cache overhead."
    }, indent=2))
    return summary
