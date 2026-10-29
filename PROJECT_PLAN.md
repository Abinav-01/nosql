# Implementation and final review checklist

The primary outcome is a reproducible comparison of retrieval quality and work across Vector-only, Graph-only, Static Hybrid and Adaptive Hybrid. Fewer calls alone are not proof of faster answers or lower financial cost.

## Phase 1 Rule based prototype

- [x] Establish package, configuration, and local service setup.
- [x] Implement public dataset adapters and deterministic pilot sampling.
- [x] Implement transparent routing rules and evidence-ID fusion.
- [x] Implement ChromaDB and Neo4j connectors.
- [x] Implement local graph extraction and answer generation adapters.
- [x] Implement comparison runner and evidence inspection interface.
- [ ] Verify both databases and model end to end on the real pilot corpus.
- [ ] Audit corpus overlap, source coverage, chunk sizes, and question composition.
- [ ] Inspect extracted triples for correctness, alias problems and missing relationships.

## Phase 2 Retrieval quality

- [ ] Add predicate-aware graph query planning with bounded, parameterized templates.
- [ ] Add alias handling and explicit entity disambiguation.
- [ ] Check full graph paths, evidence coverage and difficult comparison questions.
- [ ] Add manually reviewed explanatory and relationship questions over the same corpus.
- [ ] Audit near-duplicate questions across development and evaluation; group paraphrases.
- [ ] Freeze a final evaluation set before tuning rules.

## Phase 3 Efficiency experiments

- [ ] Establish measured results for all four modes and expose route errors.
- [ ] Choose the cheapest route meeting a declared quality threshold on development questions; allow multiple acceptable routes.
- [ ] Evaluate a lightweight classifier or small LLM router against the rule baseline.
- [ ] Measure router overhead, retrieval latency, end-to-end latency, tokens, call count and fallback rates.
- [ ] Add RAGAS faithfulness/relevance evaluation and human spot checks.
- [ ] Report category/dataset breakdowns, confidence intervals, and changes under different query mixtures.
- [ ] Add 2WikiMultiHopQA only if needed for an independent robustness check.

## Phase 4 Final review

- [ ] Lock exact dependencies, model revisions, corpus hashes, extraction version and prompts.
- [ ] Verify startup on the review computer and rehearse without network.
- [ ] Demo vector, graph, hybrid, unsupported-question and failure behavior.
- [ ] Compare identical questions across modes with evidence and measured results visible.
- [ ] Prepare report, slides, setup instructions and a demo fallback recording.
- [ ] Replace the earlier assignment's claimed completion/results with verified implementation status and actual measurements.

## Scope decisions

The initial corpus is general Wikipedia knowledge from SQuAD 1.1 and HotpotQA. No new application domain is required. We begin with rules; the later router model choice remains open. No paid API or external publishing is required for the local prototype.
