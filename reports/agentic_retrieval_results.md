# Agentic Retrieval Results

Final release-readiness report for the Tool-Memory-Evidence Retrieval Benchmark. All four implemented strategies were rerun in Qdrant `local-memory` mode during the release-readiness check. Metrics below are from those successful reruns.

Latency can vary by machine, cache state, model load behavior, and run order. Quality metrics stayed stable compared with the Phase 4/5 results.

## Dataset Summary

- tools: 10
- memories: 10
- evidence docs: 10
- tasks: 20
- dataset type: small synthetic JSONL benchmark
- runtime mode: Qdrant `local-memory`
- external LLM APIs: none

## Strategy Comparison

| Strategy | Tool@1 | Tool@3 | Memory@1 | Evidence@3 | Mean latency ms | P95 latency ms | Tool@1 gain vs dense | Memory@1 gain vs dense | Evidence@3 gain vs dense | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense | 0.700 | 0.900 | 0.650 | 0.550 | 61.46 | 83.96 | n/a | n/a | n/a | runtime-validated |
| dense-rerank | 0.750 | 0.850 | 0.700 | 0.750 | 1123.66 | 1235.73 | +0.050 | +0.050 | +0.200 | runtime-validated |
| sparse | 0.550 | 0.800 | 0.600 | 0.750 | 1.52 | 2.05 | -0.150 | -0.050 | +0.200 | runtime-validated |
| hybrid-rerank | 0.750 | 0.850 | 0.700 | 0.750 | 1099.49 | 1322.92 | +0.050 | +0.050 | +0.200 | runtime-validated |

## Best Strategy By Target

- Tool@1: `dense-rerank` and `hybrid-rerank` tied at 0.750.
- Tool@3: `dense` was best at 0.900.
- Memory@1: `dense-rerank` and `hybrid-rerank` tied at 0.700.
- Evidence@3: `dense-rerank`, `sparse`, and `hybrid-rerank` tied at 0.750.

## Quality-Latency Trade-Off Analysis

Dense retrieval is the baseline and remains strong for broad semantic matching. It produced the best Tool@3, which suggests good recall for tool retrieval on this dataset.

Sparse retrieval was the fastest strategy and improved Evidence@3 from 0.550 to 0.750. It did not improve Tool@1 or Memory@1, which suggests exact-term matching helped evidence retrieval more than top-ranked tool or memory selection in this dataset.

Dense-rerank improved Tool@1, Memory@1, and Evidence@3, but it added substantial latency. The reranker improves top-rank quality by scoring query-candidate pairs, which is more expensive than vector search.

Hybrid-rerank matched dense-rerank quality while showing slightly lower mean latency and higher P95 latency than dense-rerank in this rerun. It remains much slower than dense or sparse retrieval because it also uses cross-encoder reranking.

## Model Download Notes

FastEmbed may download model files on first use. During Phase 4 validation, the dense-rerank run visibly fetched 5 cross-encoder files for `Xenova/ms-marco-MiniLM-L-6-v2`. During this final release-readiness rerun, no explicit model download progress appeared in the observed command output.

Default models:

- dense: `BAAI/bge-small-en-v1.5`
- sparse: `Qdrant/bm25`
- reranker: `Xenova/ms-marco-MiniLM-L-6-v2`

## Local-Memory Qdrant Notes

The default runtime path uses Qdrant `local-memory` mode through `qdrant-client`. This avoids requiring Docker or a running Qdrant server. Docker/server mode remains optional for users who prefer it.

## Failure-Mode Interpretation

Dense Evidence@3 was low at 0.550. Sparse and rerank strategies improved Evidence@3 to 0.750, but that is still below the original demonstration target of 0.800.

This should be treated as a useful benchmark signal. The dataset labels should not be tuned or rewritten just to improve metrics. Future work can review evidence phrasing, add per-task failure reporting, and expand the dataset, but those changes should be documented as benchmark-design changes.

## Synthetic Dataset Limitation

These results come from a small synthetic dataset. They are useful for methodology checks and failure-mode inspection, not leaderboard or scientific claims. Results should not be presented as broad evidence that one retrieval strategy is generally superior.

## Recommended Next Experimental Directions

- Add per-task failure tables so retrieval errors are easier to inspect.
- Review evidence retrieval examples without changing labels solely to increase scores.
- Test larger synthetic slices with more distractors.
- Measure cold-start versus warm-cache model behavior separately.
- Let any upstream Qdrant/FastEmbed feedback guide packaging before expanding scope.
