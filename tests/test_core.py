import json
import pytest
from adaptive_rag.core import Hit, RuleRouter, rrf, retrieval_metrics
from adaptive_rag.data import prepare, read_jsonl, stable_id
from adaptive_rag.llm import validate_triples, parse_json_object
from adaptive_rag.pipeline import Pipeline, MODES
from adaptive_rag.evaluate import summarize, answer_metrics


def hit(identifier, source="vector"):
    return Hit(identifier, "Evidence " + identifier, "Document", 0.5, [source], [])


def test_fusion_rewards_shared_evidence_without_double_counting():
    result = rrf([[hit("a"), hit("a"), hit("b")], [hit("b", "graph")]])
    assert [h.chunk_id for h in result] == ["b", "a"]
    assert result[0].sources == ["graph", "vector"]
    assert result[1].score == pytest.approx(1 / 61)


def test_empty_query_and_uncalibrated_confidence():
    with pytest.raises(ValueError):
        RuleRouter().route("   ")
    assert RuleRouter().route("What is photosynthesis?").confidence is None


class Store:
    def __init__(self, results=None, error=False):
        self.results = results or []
        self.calls = 0
        self.error = error

    def search(self, query, limit):
        self.calls += 1
        if self.error:
            raise ConnectionError("unavailable")
        return self.results[:limit]


@pytest.mark.parametrize("question,route", [
    ("What is photosynthesis?", "vector"),
    ("Who directed Inception?", "graph"),
    ("Explain the influence of the director of Inception.", "hybrid"),
])
def test_execution_matches_selected_route(question, route):
    v, g = Store([hit("v")]), Store([hit("g", "graph")])
    result = Pipeline(v, g, None).run(question, generate=False)
    assert result["selected_route"] == route
    assert v.calls == (route in ("vector", "hybrid"))
    assert g.calls == (route in ("graph", "hybrid"))


def test_fallback_is_opt_in_and_counted():
    pipeline = Pipeline(Store(), Store([hit("g", "graph")]), None)
    without = pipeline.run("What is photosynthesis?", generate=False)
    with_fallback = pipeline.run("What is photosynthesis?", generate=False, fallback=True)
    assert without["database_calls"] == {"vector": 1, "graph": 0}
    assert with_fallback["database_calls"] == {"vector": 1, "graph": 1}
    assert with_fallback["fallback_used"] is True


def test_hybrid_failure_is_visible_not_claimed_as_success():
    result = Pipeline(Store([hit("v")]), Store(error=True), None).run(
        "Question", "static_hybrid", generate=False)
    assert result["status"] == "error"
    assert result["database_calls"] == {"vector": 1, "graph": 1}
    assert result["hits"]
    summary = summarize([dict(result, dataset="test", category="test", precision=1, recall=1)])
    assert all(r["error_rate"] == 1 and r["precision"] is None for r in summary)


def test_metrics_do_not_count_duplicates_or_missing_labels_as_perfect():
    assert retrieval_metrics(["a", "a", "b"], ["a", "c"]) == {"precision": .5, "recall": .5}
    assert retrieval_metrics([], []) == {"precision": None, "recall": None}
    assert answer_metrics("Paris [1]", ["Paris"])["exact_match"] == 1


def test_extraction_rejects_unbacked_fact_and_deduplicates():
    triple = {"subject": "Ada", "relation": "works at", "object": "Lab", "evidence": "Ada works at Lab."}
    result = validate_triples({"triples": [triple, triple, dict(triple, evidence="Invented")]},
                              "Ada works at Lab.")
    assert result == [triple]


def test_extraction_parser_accepts_markdown_fence_and_preamble():
    assert parse_json_object('Here is the result:\n```json\n{"triples": []}\n```') == {"triples": []}


def test_dataset_adapter_preserves_gold_and_keeps_answers_out_of_chunks(tmp_path):
    squad = {"data": [{"title": "Ada", "paragraphs": [{"context": "Ada works at Lab. She studies plants.",
              "qas": [{"id": "s1", "question": "Where does Ada work?",
                       "answers": [{"text": "Lab", "answer_start": 13}]}]}]}]}
    hotpot = [{"_id": "h1", "question": "What does Ada study?", "answer": "plants", "type": "bridge",
               "context": [["Ada", ["Ada works at Lab.", "She studies plants."]]],
               "supporting_facts": [["Ada", 1]]}]
    s, h = tmp_path / "s.json", tmp_path / "h.json"
    s.write_text(json.dumps(squad)); h.write_text(json.dumps(hotpot))
    out = tmp_path / "prepared"
    prepare([("squad", "development", s), ("hotpot", "evaluation", h)], out, 5)
    chunks = read_jsonl(out / "chunks.jsonl")
    evaluation = read_jsonl(out / "evaluation.jsonl")
    assert len(chunks) == 2
    assert evaluation[0]["gold_chunk_ids"] == [stable_id("Ada", "She studies plants.")]
    assert all("answers" not in c and "question" not in c for c in chunks)
    assert evaluation[0]["expected_route"] is None


def test_all_modes_share_same_final_context_limit():
    pipeline = Pipeline(Store([hit("a"), hit("b")]), Store([hit("c", "graph")]), None, top_k=1)
    for mode in MODES:
        assert len(pipeline.run("Who directed this film?", mode, generate=False)["hits"]) <= 1
