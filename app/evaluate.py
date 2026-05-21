"""CLI entrypoint for retrieval strategy evaluation.

Phase 4 keeps the benchmark small while adding sparse retrieval and reranking
strategy comparison. Each CLI run evaluates the requested strategy, computes a
dense baseline for gain comparison when needed, and writes a reproducible
markdown report when requested.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from app.data import BenchmarkDataset, RetrievableItem, load_benchmark_dataset
from app.embed import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_RERANK_MODEL,
    DEFAULT_SPARSE_MODEL,
    CrossEncoderReranker,
    DenseEmbedder,
    SparseEmbedder,
)
from app.metrics import (
    compute_evidence_at_3,
    compute_memory_at_1,
    compute_rerank_gain,
    compute_tool_at_1,
    compute_tool_at_3,
    mean,
    percentile,
)
from app.qdrant_store import (
    COLLECTIONS,
    SPARSE_COLLECTIONS,
    create_qdrant_client,
    ensure_collection,
    ensure_sparse_collection,
    search_ids,
    search_sparse_ids,
    upsert_items,
    upsert_sparse_items,
)


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_MODE = "local-memory"
DEFAULT_QDRANT_PATH = ".qdrant_local"
DEFAULT_TOP_K = 3
DEFAULT_CANDIDATE_K = 10
SUPPORTED_STRATEGIES = ("dense", "dense-rerank", "sparse", "hybrid-rerank")
RERANK_STRATEGIES = ("dense-rerank", "hybrid-rerank")


@dataclass(frozen=True)
class StrategyBenchmarkResult:
    """Computed strategy metrics, latency summary, and gain comparison."""

    strategy: str
    dense_model: str
    sparse_model: str | None
    reranker_model: str | None
    qdrant_mode: str
    qdrant_url: str
    qdrant_path: str
    tools_count: int
    memories_count: int
    evidence_count: int
    tasks_count: int
    top_k: int
    candidate_k: int
    tool_at_1: float
    tool_at_3: float
    memory_at_1: float
    evidence_at_3: float
    mean_latency_ms: float
    p95_latency_ms: float
    tool_at_1_gain_vs_dense: float | None = None
    memory_at_1_gain_vs_dense: float | None = None
    evidence_at_3_gain_vs_dense: float | None = None
    status: str = "runtime-validated"


DenseBenchmarkResult = StrategyBenchmarkResult


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run the Tool-Memory-Evidence retrieval benchmark."
    )
    parser.add_argument("--strategy", default="dense")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--qdrant-mode", default=DEFAULT_QDRANT_MODE)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--qdrant-path", default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--recreate-collections", action="store_true")
    parser.add_argument("--write-report")
    return parser


def run_dense_benchmark(
    data_dir: Path,
    qdrant_mode: str,
    qdrant_url: str,
    qdrant_path: str,
    model_name: str,
    top_k: int,
    recreate_collections: bool,
) -> StrategyBenchmarkResult:
    """Run the dense baseline end to end against local Qdrant."""

    return run_strategy_benchmark(
        strategy="dense",
        data_dir=data_dir,
        qdrant_mode=qdrant_mode,
        qdrant_url=qdrant_url,
        qdrant_path=qdrant_path,
        dense_model_name=model_name,
        sparse_model_name=DEFAULT_SPARSE_MODEL,
        reranker_model_name=DEFAULT_RERANK_MODEL,
        top_k=top_k,
        candidate_k=max(DEFAULT_CANDIDATE_K, top_k),
        recreate_collections=recreate_collections,
    ).requested_result


@dataclass(frozen=True)
class StrategyRun:
    """Results produced by one CLI run."""

    requested_result: StrategyBenchmarkResult
    dense_baseline: StrategyBenchmarkResult
    comparison_results: tuple[StrategyBenchmarkResult, ...]


def run_strategy_benchmark(
    strategy: str,
    data_dir: Path,
    qdrant_mode: str,
    qdrant_url: str,
    qdrant_path: str,
    dense_model_name: str,
    sparse_model_name: str,
    reranker_model_name: str,
    top_k: int,
    candidate_k: int,
    recreate_collections: bool,
) -> StrategyRun:
    """Run one supported retrieval strategy and return comparison results."""

    validate_strategy(strategy)
    validate_top_k(top_k)
    validate_candidate_k(strategy, top_k, candidate_k)

    dataset = load_benchmark_dataset(data_dir)
    client = create_qdrant_client(qdrant_mode, qdrant_url, qdrant_path)
    dense_embedder = DenseEmbedder(model_name=dense_model_name)

    dense_context = _build_dense_context(
        dataset=dataset,
        client=client,
        embedder=dense_embedder,
        qdrant_mode=qdrant_mode,
        qdrant_url=qdrant_url,
        qdrant_path=qdrant_path,
        dense_model_name=dense_model_name,
        top_k=top_k,
        recreate_collections=recreate_collections,
    )

    if strategy == "dense":
        return StrategyRun(
            requested_result=dense_context.result,
            dense_baseline=dense_context.result,
            comparison_results=(dense_context.result,),
        )

    if strategy == "dense-rerank":
        result = _run_dense_rerank(
            dataset=dataset,
            client=client,
            embedder=dense_embedder,
            dense_context=dense_context,
            reranker_model_name=reranker_model_name,
            qdrant_mode=qdrant_mode,
            qdrant_url=qdrant_url,
            qdrant_path=qdrant_path,
            top_k=top_k,
            candidate_k=candidate_k,
        )
    elif strategy == "sparse":
        result = _run_sparse(
            dataset=dataset,
            client=client,
            dense_context=dense_context,
            sparse_model_name=sparse_model_name,
            qdrant_mode=qdrant_mode,
            qdrant_url=qdrant_url,
            qdrant_path=qdrant_path,
            dense_model_name=dense_model_name,
            top_k=top_k,
            candidate_k=candidate_k,
            recreate_collections=recreate_collections,
        )
    else:
        result = _run_hybrid_rerank(
            dataset=dataset,
            client=client,
            embedder=dense_embedder,
            dense_context=dense_context,
            sparse_model_name=sparse_model_name,
            reranker_model_name=reranker_model_name,
            qdrant_mode=qdrant_mode,
            qdrant_url=qdrant_url,
            qdrant_path=qdrant_path,
            dense_model_name=dense_model_name,
            top_k=top_k,
            candidate_k=candidate_k,
            recreate_collections=recreate_collections,
        )

    return StrategyRun(
        requested_result=result,
        dense_baseline=dense_context.result,
        comparison_results=(dense_context.result, result),
    )


@dataclass(frozen=True)
class DenseContext:
    """Dense indexes, rankings, and item lookup shared by rerank strategies."""

    result: StrategyBenchmarkResult
    tool_rankings: dict[str, list[str]]
    memory_rankings: dict[str, list[str]]
    evidence_rankings: dict[str, list[str]]
    tool_lookup: dict[str, RetrievableItem]
    memory_lookup: dict[str, RetrievableItem]
    evidence_lookup: dict[str, RetrievableItem]


def _build_dense_context(
    dataset: BenchmarkDataset,
    client: object,
    embedder: DenseEmbedder,
    qdrant_mode: str,
    qdrant_url: str,
    qdrant_path: str,
    dense_model_name: str,
    top_k: int,
    recreate_collections: bool,
) -> DenseContext:
    tool_vectors = embedder.embed_documents([item.text for item in dataset.tools])
    memory_vectors = embedder.embed_documents([item.text for item in dataset.memories])
    evidence_vectors = embedder.embed_documents([item.text for item in dataset.evidence])
    vector_size = len(tool_vectors[0])

    ensure_collection(client, COLLECTIONS["tool"], vector_size, recreate_collections)
    ensure_collection(client, COLLECTIONS["memory"], vector_size, recreate_collections)
    ensure_collection(client, COLLECTIONS["evidence"], vector_size, recreate_collections)

    upsert_items(client, COLLECTIONS["tool"], dataset.tools, tool_vectors)
    upsert_items(client, COLLECTIONS["memory"], dataset.memories, memory_vectors)
    upsert_items(client, COLLECTIONS["evidence"], dataset.evidence, evidence_vectors)

    tool_rankings: dict[str, list[str]] = {}
    memory_rankings: dict[str, list[str]] = {}
    evidence_rankings: dict[str, list[str]] = {}
    latencies_ms: list[float] = []

    for task in dataset.tasks:
        start = time.perf_counter()
        query_vector = embedder.embed_query(task.query)
        tool_rankings[task.id] = search_ids(client, COLLECTIONS["tool"], query_vector, top_k)
        memory_rankings[task.id] = search_ids(
            client,
            COLLECTIONS["memory"],
            query_vector,
            top_k,
        )
        evidence_rankings[task.id] = search_ids(
            client,
            COLLECTIONS["evidence"],
            query_vector,
            top_k,
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    result = _compute_result(
        dataset=dataset,
        strategy="dense",
        dense_model_name=dense_model_name,
        sparse_model_name=None,
        reranker_model_name=None,
        qdrant_mode=qdrant_mode,
        qdrant_url=qdrant_url,
        qdrant_path=qdrant_path,
        top_k=top_k,
        candidate_k=top_k,
        tool_rankings=tool_rankings,
        memory_rankings=memory_rankings,
        evidence_rankings=evidence_rankings,
        latencies_ms=latencies_ms,
        dense_baseline=None,
    )

    return DenseContext(
        result=result,
        tool_rankings=tool_rankings,
        memory_rankings=memory_rankings,
        evidence_rankings=evidence_rankings,
        tool_lookup=_item_lookup(dataset.tools),
        memory_lookup=_item_lookup(dataset.memories),
        evidence_lookup=_item_lookup(dataset.evidence),
    )


def _run_dense_rerank(
    dataset: BenchmarkDataset,
    client: object,
    embedder: DenseEmbedder,
    dense_context: DenseContext,
    reranker_model_name: str,
    qdrant_mode: str,
    qdrant_url: str,
    qdrant_path: str,
    top_k: int,
    candidate_k: int,
) -> StrategyBenchmarkResult:
    reranker = CrossEncoderReranker(model_name=reranker_model_name)
    rankings, latencies_ms = _rerank_dense_candidates(
        dataset=dataset,
        client=client,
        embedder=embedder,
        reranker=reranker,
        dense_context=dense_context,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    return _compute_result(
        dataset=dataset,
        strategy="dense-rerank",
        dense_model_name=dense_context.result.dense_model,
        sparse_model_name=None,
        reranker_model_name=reranker_model_name,
        qdrant_mode=qdrant_mode,
        qdrant_url=qdrant_url,
        qdrant_path=qdrant_path,
        top_k=top_k,
        candidate_k=candidate_k,
        tool_rankings=rankings["tool"],
        memory_rankings=rankings["memory"],
        evidence_rankings=rankings["evidence"],
        latencies_ms=latencies_ms,
        dense_baseline=dense_context.result,
    )


def _run_sparse(
    dataset: BenchmarkDataset,
    client: object,
    dense_context: DenseContext,
    sparse_model_name: str,
    qdrant_mode: str,
    qdrant_url: str,
    qdrant_path: str,
    dense_model_name: str,
    top_k: int,
    candidate_k: int,
    recreate_collections: bool,
) -> StrategyBenchmarkResult:
    sparse_embedder = _build_sparse_indexes(
        dataset=dataset,
        client=client,
        sparse_model_name=sparse_model_name,
        recreate_collections=recreate_collections,
    )
    tool_rankings: dict[str, list[str]] = {}
    memory_rankings: dict[str, list[str]] = {}
    evidence_rankings: dict[str, list[str]] = {}
    latencies_ms: list[float] = []

    for task in dataset.tasks:
        start = time.perf_counter()
        query_vector = sparse_embedder.embed_query(task.query)
        tool_rankings[task.id] = search_sparse_ids(
            client,
            SPARSE_COLLECTIONS["tool"],
            query_vector,
            top_k,
        )
        memory_rankings[task.id] = search_sparse_ids(
            client,
            SPARSE_COLLECTIONS["memory"],
            query_vector,
            top_k,
        )
        evidence_rankings[task.id] = search_sparse_ids(
            client,
            SPARSE_COLLECTIONS["evidence"],
            query_vector,
            top_k,
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return _compute_result(
        dataset=dataset,
        strategy="sparse",
        dense_model_name=dense_model_name,
        sparse_model_name=sparse_model_name,
        reranker_model_name=None,
        qdrant_mode=qdrant_mode,
        qdrant_url=qdrant_url,
        qdrant_path=qdrant_path,
        top_k=top_k,
        candidate_k=candidate_k,
        tool_rankings=tool_rankings,
        memory_rankings=memory_rankings,
        evidence_rankings=evidence_rankings,
        latencies_ms=latencies_ms,
        dense_baseline=dense_context.result,
    )


def _run_hybrid_rerank(
    dataset: BenchmarkDataset,
    client: object,
    embedder: DenseEmbedder,
    dense_context: DenseContext,
    sparse_model_name: str,
    reranker_model_name: str,
    qdrant_mode: str,
    qdrant_url: str,
    qdrant_path: str,
    dense_model_name: str,
    top_k: int,
    candidate_k: int,
    recreate_collections: bool,
) -> StrategyBenchmarkResult:
    sparse_embedder = _build_sparse_indexes(
        dataset=dataset,
        client=client,
        sparse_model_name=sparse_model_name,
        recreate_collections=recreate_collections,
    )
    reranker = CrossEncoderReranker(model_name=reranker_model_name)

    tool_rankings: dict[str, list[str]] = {}
    memory_rankings: dict[str, list[str]] = {}
    evidence_rankings: dict[str, list[str]] = {}
    latencies_ms: list[float] = []

    for task in dataset.tasks:
        start = time.perf_counter()
        dense_query = embedder.embed_query(task.query)
        sparse_query = sparse_embedder.embed_query(task.query)

        tool_ids = merge_candidate_ids(
            search_ids(client, COLLECTIONS["tool"], dense_query, candidate_k),
            search_sparse_ids(client, SPARSE_COLLECTIONS["tool"], sparse_query, candidate_k),
            candidate_k,
        )
        memory_ids = merge_candidate_ids(
            search_ids(client, COLLECTIONS["memory"], dense_query, candidate_k),
            search_sparse_ids(client, SPARSE_COLLECTIONS["memory"], sparse_query, candidate_k),
            candidate_k,
        )
        evidence_ids = merge_candidate_ids(
            search_ids(client, COLLECTIONS["evidence"], dense_query, candidate_k),
            search_sparse_ids(client, SPARSE_COLLECTIONS["evidence"], sparse_query, candidate_k),
            candidate_k,
        )

        tool_rankings[task.id] = _rerank_ids(
            reranker,
            task.query,
            tool_ids,
            dense_context.tool_lookup,
            top_k,
        )
        memory_rankings[task.id] = _rerank_ids(
            reranker,
            task.query,
            memory_ids,
            dense_context.memory_lookup,
            top_k,
        )
        evidence_rankings[task.id] = _rerank_ids(
            reranker,
            task.query,
            evidence_ids,
            dense_context.evidence_lookup,
            top_k,
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return _compute_result(
        dataset=dataset,
        strategy="hybrid-rerank",
        dense_model_name=dense_model_name,
        sparse_model_name=sparse_model_name,
        reranker_model_name=reranker_model_name,
        qdrant_mode=qdrant_mode,
        qdrant_url=qdrant_url,
        qdrant_path=qdrant_path,
        top_k=top_k,
        candidate_k=candidate_k,
        tool_rankings=tool_rankings,
        memory_rankings=memory_rankings,
        evidence_rankings=evidence_rankings,
        latencies_ms=latencies_ms,
        dense_baseline=dense_context.result,
    )


def _build_sparse_indexes(
    dataset: BenchmarkDataset,
    client: object,
    sparse_model_name: str,
    recreate_collections: bool,
) -> SparseEmbedder:
    sparse_embedder = SparseEmbedder(model_name=sparse_model_name)
    tool_vectors = sparse_embedder.embed_documents([item.text for item in dataset.tools])
    memory_vectors = sparse_embedder.embed_documents([item.text for item in dataset.memories])
    evidence_vectors = sparse_embedder.embed_documents([item.text for item in dataset.evidence])

    ensure_sparse_collection(client, SPARSE_COLLECTIONS["tool"], recreate_collections)
    ensure_sparse_collection(client, SPARSE_COLLECTIONS["memory"], recreate_collections)
    ensure_sparse_collection(client, SPARSE_COLLECTIONS["evidence"], recreate_collections)

    upsert_sparse_items(client, SPARSE_COLLECTIONS["tool"], dataset.tools, tool_vectors)
    upsert_sparse_items(client, SPARSE_COLLECTIONS["memory"], dataset.memories, memory_vectors)
    upsert_sparse_items(client, SPARSE_COLLECTIONS["evidence"], dataset.evidence, evidence_vectors)
    return sparse_embedder


def _rerank_dense_candidates(
    dataset: BenchmarkDataset,
    client: object,
    embedder: DenseEmbedder,
    reranker: CrossEncoderReranker,
    dense_context: DenseContext,
    top_k: int,
    candidate_k: int,
) -> tuple[dict[str, dict[str, list[str]]], list[float]]:
    rankings = {"tool": {}, "memory": {}, "evidence": {}}
    latencies_ms: list[float] = []

    for task in dataset.tasks:
        start = time.perf_counter()
        query_vector = embedder.embed_query(task.query)
        tool_ids = search_ids(client, COLLECTIONS["tool"], query_vector, candidate_k)
        memory_ids = search_ids(client, COLLECTIONS["memory"], query_vector, candidate_k)
        evidence_ids = search_ids(client, COLLECTIONS["evidence"], query_vector, candidate_k)
        rankings["tool"][task.id] = _rerank_ids(
            reranker,
            task.query,
            tool_ids,
            dense_context.tool_lookup,
            top_k,
        )
        rankings["memory"][task.id] = _rerank_ids(
            reranker,
            task.query,
            memory_ids,
            dense_context.memory_lookup,
            top_k,
        )
        rankings["evidence"][task.id] = _rerank_ids(
            reranker,
            task.query,
            evidence_ids,
            dense_context.evidence_lookup,
            top_k,
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return rankings, latencies_ms


def _rerank_ids(
    reranker: CrossEncoderReranker,
    query: str,
    candidate_ids: Sequence[str],
    lookup: Mapping[str, RetrievableItem],
    top_k: int,
) -> list[str]:
    candidates = [lookup[item_id] for item_id in candidate_ids if item_id in lookup]
    if not candidates:
        return []
    order = reranker.rerank(query, [candidate.text for candidate in candidates])
    return [candidates[index].id for index in order[:top_k]]


def merge_candidate_ids(
    primary_ids: Sequence[str],
    secondary_ids: Sequence[str],
    limit: int,
) -> list[str]:
    """Merge candidate ids without duplicates, preserving source priority."""

    if limit < 1:
        raise ValueError("limit must be >= 1")
    merged: list[str] = []
    seen: set[str] = set()
    for item_id in list(primary_ids) + list(secondary_ids):
        if item_id in seen:
            continue
        merged.append(item_id)
        seen.add(item_id)
        if len(merged) == limit:
            break
    return merged


def validate_strategy(strategy: str) -> None:
    """Validate a requested strategy name."""

    if strategy not in SUPPORTED_STRATEGIES:
        supported = ", ".join(SUPPORTED_STRATEGIES)
        raise ValueError(f"unsupported strategy {strategy!r}; expected one of: {supported}")


def validate_top_k(top_k: int) -> None:
    """Validate the final metric cutoff."""

    if top_k < 1:
        raise ValueError("top_k must be >= 1")


def validate_candidate_k(strategy: str, top_k: int, candidate_k: int) -> None:
    """Validate pre-rerank candidate count."""

    if candidate_k < 1:
        raise ValueError("candidate_k must be >= 1")
    if strategy in RERANK_STRATEGIES and candidate_k < top_k:
        raise ValueError("candidate_k must be >= top_k for rerank strategies")


def _compute_result(
    dataset: BenchmarkDataset,
    strategy: str,
    dense_model_name: str,
    sparse_model_name: str | None,
    reranker_model_name: str | None,
    qdrant_mode: str,
    qdrant_url: str,
    qdrant_path: str,
    top_k: int,
    candidate_k: int,
    tool_rankings: dict[str, list[str]],
    memory_rankings: dict[str, list[str]],
    evidence_rankings: dict[str, list[str]],
    latencies_ms: list[float],
    dense_baseline: StrategyBenchmarkResult | None,
) -> StrategyBenchmarkResult:
    tool_at_1 = compute_tool_at_1(dataset.tasks, tool_rankings)
    memory_at_1 = compute_memory_at_1(dataset.tasks, memory_rankings)
    evidence_at_3 = compute_evidence_at_3(dataset.tasks, evidence_rankings)

    return StrategyBenchmarkResult(
        strategy=strategy,
        dense_model=dense_model_name,
        sparse_model=sparse_model_name,
        reranker_model=reranker_model_name,
        qdrant_mode=qdrant_mode,
        qdrant_url=qdrant_url,
        qdrant_path=qdrant_path,
        tools_count=len(dataset.tools),
        memories_count=len(dataset.memories),
        evidence_count=len(dataset.evidence),
        tasks_count=len(dataset.tasks),
        top_k=top_k,
        candidate_k=candidate_k,
        tool_at_1=tool_at_1,
        tool_at_3=compute_tool_at_3(dataset.tasks, tool_rankings),
        memory_at_1=memory_at_1,
        evidence_at_3=evidence_at_3,
        mean_latency_ms=mean(latencies_ms),
        p95_latency_ms=percentile(latencies_ms, 95.0),
        tool_at_1_gain_vs_dense=_gain(dense_baseline, "tool_at_1", tool_at_1),
        memory_at_1_gain_vs_dense=_gain(dense_baseline, "memory_at_1", memory_at_1),
        evidence_at_3_gain_vs_dense=_gain(dense_baseline, "evidence_at_3", evidence_at_3),
    )


def _gain(
    dense_baseline: StrategyBenchmarkResult | None,
    metric_name: str,
    value: float,
) -> float | None:
    if dense_baseline is None:
        return None
    return compute_rerank_gain(float(getattr(dense_baseline, metric_name)), value)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        strategy_run = run_strategy_benchmark(
            strategy=args.strategy,
            data_dir=Path(args.data_dir),
            qdrant_mode=args.qdrant_mode,
            qdrant_url=args.qdrant_url,
            qdrant_path=args.qdrant_path,
            dense_model_name=args.model,
            sparse_model_name=args.sparse_model,
            reranker_model_name=args.reranker_model,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            recreate_collections=args.recreate_collections,
        )
    except Exception as exc:
        message = f"Error: {exc}"
        print(message, file=sys.stderr)
        if args.write_report:
            _write_failure_report(Path(args.write_report), args.strategy, message)
        return 2

    print(_format_summary(strategy_run))
    if args.write_report:
        _write_success_report(Path(args.write_report), strategy_run)
        print(f"Report written: {args.write_report}")
    else:
        print("Report written: no")
    return 0


def _format_summary(strategy_run: StrategyRun) -> str:
    result = strategy_run.requested_result
    lines = [
        "Tool-Memory-Evidence Retrieval Benchmark",
        "=======================================",
        "",
        f"Strategy: {result.strategy}",
        f"Dense model: {result.dense_model}",
        f"Sparse model: {result.sparse_model or 'not used'}",
        f"Reranker model: {result.reranker_model or 'not used'}",
        f"Qdrant mode: {result.qdrant_mode}",
        f"Qdrant target: {_qdrant_target(result)}",
        f"Top-k: {result.top_k}",
        f"Candidate-k: {result.candidate_k}",
        "",
        "Dataset:",
        f"- tools: {result.tools_count}",
        f"- memories: {result.memories_count}",
        f"- evidence docs: {result.evidence_count}",
        f"- tasks: {result.tasks_count}",
        "",
        "Results:",
        f"Tool@1: {result.tool_at_1:.3f}",
        f"Tool@3: {result.tool_at_3:.3f}",
        f"Memory@1: {result.memory_at_1:.3f}",
        f"Evidence@3: {result.evidence_at_3:.3f}",
        f"Mean latency: {result.mean_latency_ms:.2f} ms",
        f"P95 latency: {result.p95_latency_ms:.2f} ms",
    ]
    if result.strategy != "dense":
        lines.extend(
            [
                "Dense baseline for gains: computed in this run",
                f"Tool@1 gain vs dense: {format_gain(result.tool_at_1_gain_vs_dense)}",
                f"Memory@1 gain vs dense: {format_gain(result.memory_at_1_gain_vs_dense)}",
                f"Evidence@3 gain vs dense: {format_gain(result.evidence_at_3_gain_vs_dense)}",
            ]
        )
    return "\n".join(lines)


def format_gain(value: float | None) -> str:
    """Format a gain value for terminal and report output."""

    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def _write_success_report(path: Path, strategy_run: StrategyRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = _merge_report_results(path, strategy_run.comparison_results)
    path.write_text(_format_report(merged), encoding="utf-8")


def _merge_report_results(
    path: Path,
    current_results: Iterable[StrategyBenchmarkResult],
) -> dict[str, StrategyBenchmarkResult]:
    results = _read_existing_table(path)
    for result in current_results:
        results[result.strategy] = result
    return results


def _read_existing_table(path: Path) -> dict[str, StrategyBenchmarkResult]:
    if not path.exists():
        return {}
    rows: dict[str, StrategyBenchmarkResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 11 or cells[0] in {"Strategy", "---"}:
            continue
        try:
            rows[cells[0]] = StrategyBenchmarkResult(
                strategy=cells[0],
                dense_model=DEFAULT_DENSE_MODEL,
                sparse_model=None,
                reranker_model=None,
                qdrant_mode=DEFAULT_QDRANT_MODE,
                qdrant_url=DEFAULT_QDRANT_URL,
                qdrant_path=DEFAULT_QDRANT_PATH,
                tools_count=10,
                memories_count=10,
                evidence_count=10,
                tasks_count=20,
                top_k=DEFAULT_TOP_K,
                candidate_k=DEFAULT_CANDIDATE_K,
                tool_at_1=float(cells[1]),
                tool_at_3=float(cells[2]),
                memory_at_1=float(cells[3]),
                evidence_at_3=float(cells[4]),
                mean_latency_ms=float(cells[5]),
                p95_latency_ms=float(cells[6]),
                tool_at_1_gain_vs_dense=_parse_gain(cells[7]),
                memory_at_1_gain_vs_dense=_parse_gain(cells[8]),
                evidence_at_3_gain_vs_dense=_parse_gain(cells[9]),
                status=cells[10],
            )
        except ValueError:
            continue
    return rows


def _parse_gain(value: str) -> float | None:
    if value == "n/a":
        return None
    return float(value)


def _format_report(results_by_strategy: Mapping[str, StrategyBenchmarkResult]) -> str:
    ordered = [results_by_strategy[name] for name in SUPPORTED_STRATEGIES if name in results_by_strategy]
    if not ordered:
        raise ValueError("cannot write report without strategy results")
    first = ordered[0]
    rows = [
        "| Strategy | Tool@1 | Tool@3 | Memory@1 | Evidence@3 | Mean latency ms | P95 latency ms | Tool@1 gain vs dense | Memory@1 gain vs dense | Evidence@3 gain vs dense | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in ordered:
        rows.append(
            "| "
            + " | ".join(
                [
                    result.strategy,
                    f"{result.tool_at_1:.3f}",
                    f"{result.tool_at_3:.3f}",
                    f"{result.memory_at_1:.3f}",
                    f"{result.evidence_at_3:.3f}",
                    f"{result.mean_latency_ms:.2f}",
                    f"{result.p95_latency_ms:.2f}",
                    format_gain(result.tool_at_1_gain_vs_dense),
                    format_gain(result.memory_at_1_gain_vs_dense),
                    format_gain(result.evidence_at_3_gain_vs_dense),
                    result.status,
                ]
            )
            + " |"
        )

    return "\n".join(
        [
            "# Agentic Retrieval Results",
            "",
            "Phase 4 strategy comparison report with real metrics from successful local runs.",
            "",
            "## Dataset Summary",
            "",
            f"- tools: {first.tools_count}",
            f"- memories: {first.memories_count}",
            f"- evidence docs: {first.evidence_count}",
            f"- tasks: {first.tasks_count}",
            "- dataset type: small synthetic JSONL benchmark",
            "",
            "## Strategy Comparison",
            "",
            *rows,
            "",
            "## Quality Metrics",
            "",
            "- Tool@1 and Tool@3 measure tool retrieval.",
            "- Memory@1 measures top-ranked memory or policy retrieval.",
            "- Evidence@3 measures whether the expected grounding evidence appears in the final top 3.",
            "- Gain columns compare each strategy with the dense baseline when both were available.",
            "",
            "## Latency Metrics",
            "",
            "- Mean and P95 latency are measured per task for the requested retrieval strategy.",
            "- Rerank strategies include candidate retrieval and cross-encoder reranking time.",
            "- Dense baseline rows measure dense retrieval only.",
            "",
            "## Per-Strategy Notes",
            "",
            *_strategy_notes(ordered),
            "",
            "## Failure Analysis",
            "",
            *_failure_analysis(ordered),
            "",
            "## Synthetic Dataset Limitation Note",
            "",
            "These results come from a small synthetic dataset. They are useful for methodology checks and failure-mode inspection, not leaderboard or scientific claims.",
        ]
    ) + "\n"


def _strategy_notes(results: Sequence[StrategyBenchmarkResult]) -> list[str]:
    notes: list[str] = []
    strategies = {result.strategy for result in results}
    if "dense" in strategies:
        notes.append("- dense: broad semantic matching baseline using FastEmbed dense vectors in Qdrant.")
    if "dense-rerank" in strategies:
        notes.append("- dense-rerank: dense candidates are rescored with FastEmbed TextCrossEncoder; latency includes reranking.")
    if "sparse" in strategies:
        notes.append("- sparse: FastEmbed sparse BM25-style vectors are searched with Qdrant sparse vectors and IDF modifier.")
    if "hybrid-rerank" in strategies:
        notes.append("- hybrid-rerank: dense and sparse candidate ids are merged, deduplicated, then reranked.")
    return notes


def _failure_analysis(results: Sequence[StrategyBenchmarkResult]) -> list[str]:
    dense = next((result for result in results if result.strategy == "dense"), None)
    lines: list[str] = []
    if dense is not None and dense.evidence_at_3 < 0.8:
        lines.append(
            "- Evidence@3 remains below the PRD demonstration target for the dense baseline, "
            "which suggests evidence phrasing and ground-truth design should be reviewed in a later phase."
        )
    best_evidence = max(results, key=lambda result: result.evidence_at_3)
    lines.append(
        f"- Best observed Evidence@3 in this report is {best_evidence.evidence_at_3:.3f} from `{best_evidence.strategy}`."
    )
    lines.append(
        "- Dataset labels were not changed for Phase 4; weak evidence scores are reported rather than tuned away."
    )
    return lines


def _write_failure_report(path: Path, strategy: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Agentic Retrieval Results",
                "",
                f"Strategy `{strategy}` did not complete successfully.",
                "",
                "## Execution Blocker",
                "",
                reason,
                "",
                "Run locally after dependencies, model availability, and Qdrant mode are ready:",
                "",
                "```powershell",
                f"python -m app.evaluate --strategy {strategy} --qdrant-mode local-memory --recreate-collections --write-report reports/agentic_retrieval_results.md",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _item_lookup(items: Sequence[RetrievableItem]) -> dict[str, RetrievableItem]:
    return {item.id: item for item in items}


def _qdrant_target(result: StrategyBenchmarkResult) -> str:
    if result.qdrant_mode == "server":
        return result.qdrant_url
    if result.qdrant_mode == "local-disk":
        return result.qdrant_path
    return ":memory:"


if __name__ == "__main__":
    raise SystemExit(main())
