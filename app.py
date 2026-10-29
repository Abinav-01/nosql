import json
import streamlit as st
from adaptive_rag.config import Settings
from adaptive_rag.core import RuleRouter
from adaptive_rag.cli import make_pipeline
from adaptive_rag.pipeline import MODES
from adaptive_rag.data import read_jsonl

st.set_page_config(page_title="Adaptive Vector–Graph RAG", layout="wide")
st.title("Adaptive Vector–Graph RAG")
st.caption("Research prototype · rule-based routing · measured retrieval comparisons")
settings = Settings.load()
processed = settings.data_root
manifest_path = processed / "manifest.json"
graph_ready = (processed / "graph_index.json").exists()
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    stats = st.columns(3)
    stats[0].metric("Evidence chunks", manifest["chunks"])
    stats[1].metric("Development questions", manifest["questions"]["development"])
    stats[2].metric("Evaluation questions", manifest["questions"]["evaluation"])
if not graph_ready:
    extracted = len(read_jsonl(processed / "extractions.jsonl")) if (processed / "extractions.jsonl").exists() else 0
    st.warning(f"Graph preparation is incomplete ({extracted} chunks extracted). "
               "Vector retrieval can be explored now. Four-mode comparisons require the complete graph.")
st.sidebar.header("Experiment controls")
labels = {"vector": "Vector only", "graph": "Graph only", "static_hybrid": "Static hybrid", "adaptive": "Adaptive routing"}
mode = st.sidebar.selectbox("Retrieval mode", MODES, index=3 if graph_ready else 0,
                            format_func=labels.get, key="retrieval_mode")
generate = st.sidebar.checkbox("Generate an answer with the local LLM", value=False)
fallback = st.sidebar.checkbox("Allow fallback for empty adaptive retrieval", value=False)
st.sidebar.caption("Fallback is disabled in benchmark runs. Rule decisions have no calibrated confidence score.")
examples_path = processed / "development.jsonl"
examples = [q["question"] for q in read_jsonl(examples_path)[:8]] if examples_path.exists() else []
example = st.selectbox("Try a development question", [""] + examples, format_func=lambda q: q or "Choose an example…")
question = st.text_area("Ask a question about the indexed corpus", value=example, placeholder="Enter a question…")
if question.strip():
    decision = RuleRouter().route(question)
    st.info(f"Rule preview: {decision.route.upper()} — {decision.reason}")


@st.cache_resource
def pipeline_for(mode):
    return make_pipeline(Settings.load(), mode)


if st.button("Run retrieval", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving evidence…"):
            result = pipeline_for(mode).run(question, mode, generate, fallback)
        st.session_state["result"] = result
    except Exception as exc:
        st.error(f"Cannot start retrieval: {exc}")
        st.caption("Follow README setup steps and run adaptive-rag doctor. No simulated results are shown.")

result = st.session_state.get("result")
if result:
    st.subheader(result["question"])
    columns = st.columns(4)
    columns[0].metric("Selected route", result["selected_route"].upper())
    columns[1].metric("Database calls", sum(result["database_calls"].values()))
    columns[2].metric("Retrieval", f"{result['retrieval_ms']:.1f} ms")
    columns[3].metric("Total", f"{result['total_ms']:.1f} ms")
    if result["retrieval_errors"] or result["generation_error"]:
        st.error({"retrieval": result["retrieval_errors"], "generation": result["generation_error"]})
    if result["fallback_used"]:
        st.warning("Fallback used; the call count includes the second retrieval attempt.")
    if result["answer"]:
        st.subheader("Answer")
        st.write(result["answer"])
    if result["usage"].get("citation_warning"):
        st.warning(result["usage"]["citation_warning"])
    st.subheader("Retrieved evidence")
    if not result["hits"]:
        st.warning("No evidence retrieved. This does not prove the question is globally unanswerable.")
    for index, hit in enumerate(result["hits"], 1):
        with st.expander(f"[{index}] {hit['title']} · {', '.join(hit['sources'])}", expanded=True):
            st.write(hit["text"])
            for path in hit["paths"]:
                st.code(path, language=None)
            st.caption(f"Evidence ID: {hit['chunk_id']}")
    st.download_button("Download this run", json.dumps(result, indent=2), "retrieval-run.json", "application/json")

st.subheader("Routing efficiency dashboard")
st.caption("Novelty: the efficiency score measures avoided database calls against always-on Static Hybrid retrieval.")
summary_files = sorted((settings.root / "runs").glob("*/summary.json"))
if summary_files:
    selected_summary = st.selectbox("Benchmark summary", summary_files, format_func=lambda p: p.parent.name, key="summary_run")
    summary_rows = json.loads(selected_summary.read_text())
    overall = [r for r in summary_rows if r["group"] == "overall"]
    if overall:
        import pandas as pd
        frame = pd.DataFrame(overall)
        static_calls = float(frame.loc[frame["mode"] == "static_hybrid", "mean_database_calls"].iloc[0])
        adaptive_calls = float(frame.loc[frame["mode"] == "adaptive", "mean_database_calls"].iloc[0])
        avoided = max(0.0, 1 - adaptive_calls / static_calls) if static_calls else 0.0
        a, b, c = st.columns(3)
        a.metric("Adaptive calls/query", f"{adaptive_calls:.2f}")
        b.metric("Static hybrid calls/query", f"{static_calls:.2f}")
        c.metric("Calls avoided", f"{avoided:.0%}")
        st.bar_chart(frame.set_index("mode")[["mean_database_calls", "median_total_ms"]])
        st.dataframe(frame[["mode", "mean_database_calls", "median_total_ms", "precision", "recall", "error_rate"]], use_container_width=True)
        rows_with_ci = [row for row in overall if row.get("confidence_intervals")]
        if rows_with_ci:
            ci = [
                {"mode": row["mode"], "precision 95% low": row["confidence_intervals"].get("precision", {}).get("low"),
                 "precision 95% high": row["confidence_intervals"].get("precision", {}).get("high"),
                 "recall 95% low": row["confidence_intervals"].get("recall", {}).get("low"),
                 "recall 95% high": row["confidence_intervals"].get("recall", {}).get("high")}
                for row in rows_with_ci
            ]
            st.caption("Approximate 95% confidence intervals across repeated query runs")
            st.dataframe(ci, use_container_width=True)
        else:
            st.info("This saved benchmark predates confidence intervals. Run a new evaluation to add them; existing results remain usable.")
else:
    st.info("Run the benchmark to populate the efficiency dashboard.")
router_comparison = processed / "router_comparison.json"
if router_comparison.exists():
    st.subheader("Rule router versus lightweight learned router")
    st.json(json.loads(router_comparison.read_text()))

