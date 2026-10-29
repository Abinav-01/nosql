# Current project status

Verified on 10 October 2026.

## Working and checked

- Public pilot corpus downloaded and normalized: 4,552 chunks across 1,060 source titles.
- 100 development questions and 100 evaluation questions; each split has 50 SQuAD and 50 HotpotQA questions.
- All gold evidence IDs resolve to indexed corpus chunks.
- Full ChromaDB index built with all-MiniLM-L6-v2.
- Real vector retrieval checked in the command line and the browser interface.
- Five chunks extracted using the existing local llama3.1 model: 19 supported-quote triples.
- Neo4j indexing and graph retrieval checked on those three chunks.
- Local answer generation returned a cited answer from graph evidence.
- 11 automated tests pass; interface startup and rule-preview checks pass.

## Still required

- Extract and inspect the remaining corpus, then build the complete graph index.
- Improve graph predicate handling and entity aliases.
- Add verified explanatory questions: the initial sample triggers no Hybrid routes under rules-v1.
- Freeze a representative final evaluation set. The current set is a pilot.
- Run four-mode benchmarks; no efficiency improvement is claimed yet.
- Add the lightweight model router, RAGAS metrics, and final review materials.

The local interface defaults to Vector-only until the complete graph index exists. Incomplete graph data cannot be used through the normal four-mode benchmark entry point. The three-chunk graph check is integration evidence only.

Running local services: project Neo4j container, Ollama application, and Streamlit at http://127.0.0.1:8501. To stop the project interface, stop its process; to stop Neo4j without deleting data, use `docker compose stop` from the project directory. Ollama was started through its installed application.

Full-corpus LLM extraction has not been started in the background. It is resumable using `adaptive-rag extract`; runtime depends on the selected local model and computer. Start with a representative batch, inspect quality, then continue to avoid spending hours building a noisy graph.

The current resumable extraction contains five chunks after the latest run was interrupted during a slow local-model request. The interruption did not corrupt the JSONL file. The command now stops on a failed request and preserves completed work for the next run.

## Fast synthetic demo path

The project also includes a deterministic 100-chunk synthetic corpus under `data/synthetic_100/`. It has pre-validated graph relationships, so it does not require LLM extraction. Use `ADAPTIVE_RAG_DATA_ROOT=data/synthetic_100` for a fast controlled demo and benchmark. This path is intended for the final review when reproducible execution time matters more than open-domain corpus scale.
