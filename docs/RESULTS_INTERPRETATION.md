# Results Interpretation

## Summary

The final release-readiness rerun shows strategy trade-offs rather than a single winner. Dense retrieval remains strong for broad semantic tool matching. Sparse retrieval is very fast and improved Evidence@3 in this dataset. Reranking improved top-rank quality but added substantial latency.

These numbers come from a small synthetic dataset and should be read as methodology/demo results.

## Verified Results

| Strategy | Tool@1 | Tool@3 | Memory@1 | Evidence@3 | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.700 | 0.900 | 0.650 | 0.550 | 61.46 | 83.96 |
| dense-rerank | 0.750 | 0.850 | 0.700 | 0.750 | 1123.66 | 1235.73 |
| sparse | 0.550 | 0.800 | 0.600 | 0.750 | 1.52 | 2.05 |
| hybrid-rerank | 0.750 | 0.850 | 0.700 | 0.750 | 1099.49 | 1322.92 |

## Strategy-By-Strategy Notes

Dense retrieval:

- Best Tool@3 in this run.
- Good baseline for broad semantic matching.
- Weak Evidence@3 suggests some evidence labels or evidence phrasing may be harder than the tool/memory surfaces.

Sparse retrieval:

- Fastest strategy in this run.
- Improved Evidence@3 to 0.750.
- Lower Tool@1 and Memory@1 than dense, likely because some task-to-target matches are paraphrased rather than exact-term matches.

Dense-rerank:

- Improved Tool@1, Memory@1, and Evidence@3 versus dense.
- Reduced Tool@3 from 0.900 to 0.850.
- Added high mean and P95 latency because every task reranks multiple candidate lists with a cross-encoder.

Hybrid-rerank:

- Matched dense-rerank quality in this run.
- Slightly lower mean latency and higher P95 latency than dense-rerank in this run, while still much slower than dense or sparse retrieval.
- Useful as a simple production-like pattern, but it adds more moving parts.

## Why Sparse Is Fast

The sparse strategy uses BM25-style sparse vectors and Qdrant sparse search. On this small dataset, sparse vector scoring is cheap and does not invoke cross-encoder reranking at query time.

The result is very low observed latency, but lower latency does not imply better overall quality.

## Why Reranking Is Expensive

Reranking uses a cross-encoder to score query-candidate pairs. Unlike vector search, the reranker evaluates each candidate text against the query. This can improve top-rank relevance but scales with candidate count and number of target surfaces.

In this benchmark each task retrieves tools, memories, and evidence, so reranking is applied across three candidate lists.

## Why Evidence@3 Matters

Agentic systems often need evidence before an action can be justified or an answer can be grounded. If evidence retrieval is weak, a system may choose a reasonable tool and policy while still grounding the final output poorly.

Dense Evidence@3 was 0.550. Sparse and rerank strategies improved it to 0.750, but that is still below the original demonstration target of 0.800.

## Honest Failure Reporting

Low metrics should be reported, not hidden. The dataset should not be tuned or relabeled just to improve results. Future work can review evidence phrasing and ground-truth design, but any dataset change should be treated as a benchmark-design change and documented separately.
