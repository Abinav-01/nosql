# Adaptive Vector Graph Retrieval Augmented Generation

This project investigates whether selecting vector, graph, or hybrid retrieval for each question can reduce retrieval work while preserving answer quality. The first router uses explicit rules. A lightweight model router and RAGAS evaluation are later milestones, not completed features.

## Current implementation

- Public SQuAD 1.1 and HotpotQA dataset adapters with deterministic sampling, stable evidence IDs, reference evidence, and separate development/evaluation questions.
- Persistent ChromaDB with all-MiniLM-L6-v2 embeddings.
- Local Ollama-assisted extraction of source-backed triples and answer generation.
- Neo4j entity/fact/evidence representation and bounded neighborhood retrieval.
- Rule-based Vector/Graph/Hybrid routing; rule reasons exposed, no invented confidence percentages.
- Reciprocal Rank Fusion over shared passage IDs, retaining relationship evidence.
- Four-mode evaluation with randomized execution order, real call counts, latency, failure reporting, retrieval precision/recall, and optional answer EM/token F1.
- Streamlit evidence inspector and saved benchmark viewer.

This is an initial research prototype. The final project requires measured experiments and the improvements tracked in PROJECT_PLAN.md.

## Setup

Use Python 3.11–3.13. A local `.venv` is included in this workspace but should be recreated on another computer.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
cp .env.example .env
```

Set `NEO4J_PASSWORD` in `.env` to a local password of at least eight characters. Start Docker Desktop, then:

```sh
docker compose up -d
```

Start the installed Ollama application (or `ollama serve`), then download the configured generation/extraction model:

```sh
ollama pull llama3.2:3b
```

Ollama is the initial local provider; the model is configurable through `.env`. Its extraction and answer quality must be evaluated. It is not the future lightweight routing model. Dataset/model downloads require internet; query execution can run locally once downloads and indexes are complete.

On this computer, the existing `llama3.1:latest` model is selected in the private `.env` for initial verification, avoiding an additional model download. `.env.example` is a portable example, not the current machine's exact configuration.

## Prepare the pilot corpus

```sh
adaptive-rag prepare --per-dataset 50
```

This selects up to 50 questions from each source in each split: 100 development and 100 evaluation questions. Source train splits supply development; source validation/dev splits supply local evaluation. HotpotQA's initial training download is the first of two published training shards; this is a pilot, not a full benchmark reproduction.

Passages from all selected questions are pooled into a shared corpus, including Hotpot distractor passages. Only text and source metadata enter the indexes. Questions, answers, and gold supporting evidence stay in separate evaluation files. Corpus inclusion of evaluation documents is intentional; evaluation answers and questions do not enter the retrieval index.

## Run the synthetic and real demonstrations separately

The final review uses two independent data roots. The synthetic corpus is deterministic and is already indexed as `data/synthetic_100_v2`. The real demonstration uses public SQuAD 1.1 and HotpotQA passages under `data/real_100`; its extraction and indexes never overwrite the synthetic experiment.

Synthetic demonstration:

```sh
export ADAPTIVE_RAG_DATA_ROOT=data/synthetic_100_v2
adaptive-rag ask 'Which technology does the Data Science department use?' --mode adaptive --retrieval-only
adaptive-rag evaluate --repeats 3 --name synthetic-demo-v1
streamlit run app.py
```

Real demonstration (download and prepare a small corpus first):

```sh
export ADAPTIVE_RAG_DATA_ROOT=data/real_100
adaptive-rag prepare --per-dataset 3
adaptive-rag extract
adaptive-rag index --store both
adaptive-rag evaluate --repeats 3 --name real-demo-v1
streamlit run app.py
```

The `--per-dataset 3` setting gives a small, reviewable real corpus from SQuAD train/dev and HotpotQA train/dev. Increase it only after the first extraction succeeds. Extraction uses the configured local Ollama model and may take several minutes; it resumes from the saved `extractions.jsonl` file. Use questions from `data/real_100/development.jsonl` in the dashboard. To switch back to the synthetic dashboard, change `ADAPTIVE_RAG_DATA_ROOT` and restart Streamlit.

Keep the benchmark names unique. The program refuses to overwrite an existing run so that results remain reproducible.

Dataset names do not determine route labels. `expected_route` starts as null. Original question categories (single passage, bridge, comparison) are not equivalent to retrieval routes.

Sources and attribution:
- SQuAD: https://rajpurkar.github.io/SQuAD-explorer/ (CC BY-SA 4.0).
- HotpotQA: https://hotpotqa.github.io/ (CC BY-SA 4.0); distribution: https://huggingface.co/datasets/hotpotqa/hotpot_qa.
- Preserve source attribution and the applicable share-alike terms if redistributing derived data. Raw dataset downloads are excluded from version control.

## Inspect routing, extract and index

```sh
adaptive-rag route 'Who directed Inception?'
adaptive-rag doctor
adaptive-rag extract
adaptive-rag index --store both
```

Extraction is sequential, writes after each completed chunk, and resumes on restart. Start with `extract --limit 10` to inspect extraction quality; complete extraction before graph indexing. Exact supporting quotes are required, but a matching quote alone does not prove a triple is correct. Manually inspect extracted facts.

The graph uses Entity → Fact → Entity paths and Fact → Evidence links. Two graph edges represent one semantic relationship hop. Entity normalization initially handles case and whitespace; alias resolution, predicate-aware traversal, and structured intent parsing are unfinished. The current graph baseline retrieves nearby facts rather than compiling arbitrary questions into exact logical queries.

Index writes are idempotent for the same corpus. Vector collections include model/corpus fingerprints. Graph results filter supporting evidence to the current corpus. Changing models or extraction policies requires a deliberate new extraction/index version; cached extraction is not automatically replaced.

## Run the interface

```sh
streamlit run app.py --server.address 127.0.0.1
```

The interface previews routing even before services are ready. Retrieval requires indexed databases; generation additionally requires Ollama. Errors appear explicitly. The application does not substitute fake answers or synthetic benchmark numbers when services fail.

Command-line retrieval:

```sh
adaptive-rag ask 'Who directed Inception?' --mode adaptive --retrieval-only
```

Questions must be supported by the selected corpus; the example illustrates syntax, not guaranteed coverage. Use questions from `data/processed/development.jsonl` when exploring the pilot.

## Measure all four modes

```sh
adaptive-rag evaluate --limit 10 --repeats 3 --name smoke
adaptive-rag evaluate --repeats 3 --generate --name pilot-with-generation
```

Each run saves per-query results, summaries, and protocol notes in `runs/<name>/`. Existing runs cannot be overwritten. Fallback is disabled for baseline comparisons. Optional fallback in interactive adaptive mode is recorded and counted.

Initialization/model-loading time is excluded from query timing; first queries can still include cache overhead. Hybrid branches run in parallel in both static and adaptive modes. Actual retrieval attempts are counted, including failures. Database health checks and initialization calls are outside these counts. Graph traversal complexity is not represented by call count, so inspect latency and later graph profiling too.

Retrieval metrics use annotated sentence evidence mapped to chunk IDs; unannotated but valid evidence may be scored as irrelevant. Answer EM/F1 favors short reference answers and penalizes extra explanation. Neither metric is a substitute for faithfulness evaluation. RAGAS metrics, calibrated route labels, and statistical confidence intervals remain to be added.

## Verification

```sh
python -m pytest
```

Tests cover evidence fusion, isolation of evaluation answers, gold evidence mapping, actual execution paths, fallback accounting, and failure-aware metrics. Fake stores in tests verify orchestration only; their outputs are not experimental results. Service-backed checks must be run with ChromaDB, Neo4j, the embedding model, and Ollama available.

`requirements.lock.txt` records the installed environment for this macOS/Python 3.12 pilot. Recreate with `pip install -r requirements.lock.txt` from this directory, then `pip install -e .`. Cross-platform dependency resolution may differ.
