# Future PR Plan

## Why Start With Discussion

The benchmark crosses documentation, examples, FastEmbed models, Qdrant local mode, and evaluation methodology. A discussion gives maintainers a low-friction way to decide whether the idea fits the project and where it belongs before any pull request adds files.

The first upstream step should ask for direction, not submit a large unsolicited repo dump.

The intended upstream contribution should likely be a trimmed docs/example or experiments artifact, not this full repository.

## Smallest Acceptable Contribution

The smallest useful contribution is likely a concise docs page that explains the Tool-Memory-Evidence retrieval pattern and links to a minimal runnable example.

The example should preserve the core idea:

- one small synthetic dataset
- three retrieval targets
- dense baseline
- optional sparse or rerank comparison if maintainers want it
- no external LLM API dependency

## PR Scope Options

Docs-only example:

- A markdown page explaining the benchmark idea, metrics, and local run shape.
- Best if maintainers want concept documentation without extra example code.

Runnable script example:

- A single Python script plus short docs.
- Best if maintainers want a compact FastEmbed + Qdrant example users can run locally.

Experiments benchmark folder:

- A small folder with README, dataset, and benchmark script.
- Best if maintainers want to keep benchmark-style material separate from docs examples.

## Likely Upstream Files

Possible docs placement:

```text
docs/examples/tool_memory_evidence_benchmark.md
docs/examples/tool_memory_evidence_benchmark.py
```

Possible experiments placement:

```text
experiments/agentic_retrieval_benchmark/
  README.md
  benchmark.py
  data/
```

## What Not To Include Initially

Do not include:

- the full repository
- this repository governance system
- unrelated planning docs
- job-seeking language
- cloud deployment material
- a UI
- FastAPI
- external LLM clients
- broad benchmark claims from the synthetic dataset

## Upstream-Ready Acceptance Criteria

An upstream PR should be:

- small enough to review quickly
- consistent with the maintainers' preferred placement
- runnable without paid APIs
- clear about first-run model downloads
- explicit that the dataset is synthetic
- honest about quality-latency trade-offs
- focused on Qdrant + FastEmbed usage
- free of unrelated repository governance artifacts

## Risks Maintainers May Raise

Synthetic dataset:

The dataset is useful for methodology but not broad enough for scientific claims.

Scope too broad:

Tool, memory, evidence, dense, sparse, rerank, and hybrid may be more than maintainers want in one first example.

Dependency/model download:

FastEmbed model downloads can make first-run behavior slower or impossible in restricted networks.

Benchmark not general enough:

The task set is compact and agentic, but may not cover enough domains for general benchmark status.

## Mitigations

- Present it as an example or experiment, not an official benchmark.
- Offer a docs-only version if code scope is too large.
- Keep the runnable version local-first and minimal.
- Document model downloads clearly.
- Keep metrics and labels unchanged unless maintainers explicitly request dataset redesign.
- Include failure-mode notes instead of tuning results upward.
