"""Pure deterministic retrieval metrics for Phase 2.

These functions operate on already-ranked ids. Empty expected mappings return
0.0 for hit-rate metrics because there are no tasks to score. Empty numeric
sequences raise ValueError for aggregate statistics.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


Ranking = Mapping[str, Sequence[str]]


def hit_at_k(expected_id: str, ranked_ids: Sequence[str], k: int) -> bool:
    """Return whether the expected id appears in the first k ranked ids."""

    _validate_k(k)
    return expected_id in ranked_ids[:k]


def compute_hit_rate_at_k(
    expected_by_task: Mapping[str, str],
    rankings_by_task: Ranking,
    k: int,
) -> float:
    """Compute hit rate at k, treating missing rankings as misses."""

    _validate_k(k)
    if not expected_by_task:
        return 0.0

    hits = 0
    for task_id, expected_id in expected_by_task.items():
        if hit_at_k(expected_id, rankings_by_task.get(task_id, ()), k):
            hits += 1
    return hits / len(expected_by_task)


def compute_tool_at_1(tasks: Sequence[Any], tool_rankings: Ranking) -> float:
    """Compute Tool@1 for benchmark tasks."""

    return compute_hit_rate_at_k(_expected_by_task(tasks, "expected_tool"), tool_rankings, 1)


def compute_tool_at_3(tasks: Sequence[Any], tool_rankings: Ranking) -> float:
    """Compute Tool@3 for benchmark tasks."""

    return compute_hit_rate_at_k(_expected_by_task(tasks, "expected_tool"), tool_rankings, 3)


def compute_memory_at_1(tasks: Sequence[Any], memory_rankings: Ranking) -> float:
    """Compute Memory@1 for benchmark tasks."""

    return compute_hit_rate_at_k(_expected_by_task(tasks, "expected_memory"), memory_rankings, 1)


def compute_evidence_at_3(tasks: Sequence[Any], evidence_rankings: Ranking) -> float:
    """Compute Evidence@3 for benchmark tasks."""

    return compute_hit_rate_at_k(_expected_by_task(tasks, "expected_evidence"), evidence_rankings, 3)


def percentile(values: Sequence[float], q: float) -> float:
    """Compute percentile q using linear interpolation over sorted values."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if q < 0.0 or q > 100.0:
        raise ValueError("percentile q must be between 0 and 100")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * (q / 100.0)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return float(lower_value + (upper_value - lower_value) * fraction)


def mean(values: Sequence[float]) -> float:
    """Compute arithmetic mean for a non-empty numeric sequence."""

    if not values:
        raise ValueError("mean requires at least one value")
    return float(sum(values) / len(values))


def compute_rerank_gain(before: float, after: float) -> float:
    """Compute absolute metric improvement from reranking."""

    return after - before


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")


def _expected_by_task(tasks: Sequence[Any], attr_name: str) -> dict[str, str]:
    return {task.id: getattr(task, attr_name) for task in tasks}
