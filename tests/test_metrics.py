import pytest

from app.data import BenchmarkTask
from app.metrics import (
    compute_evidence_at_3,
    compute_memory_at_1,
    compute_rerank_gain,
    compute_tool_at_1,
    compute_tool_at_3,
    hit_at_k,
    mean,
    percentile,
)


TASKS = [
    BenchmarkTask(
        id="task_001",
        query="Validate invoice tax.",
        expected_tool="tax_validator",
        expected_memory="invoice_validation_policy",
        expected_evidence="invoice_tax_rules",
    ),
    BenchmarkTask(
        id="task_002",
        query="Write an email.",
        expected_tool="email_writer",
        expected_memory="human_approval_policy",
        expected_evidence="audit_logging_requirements",
    ),
]


def test_hit_at_k_true_and_false_cases() -> None:
    assert hit_at_k("tax_validator", ["invoice_parser", "tax_validator"], 2) is True
    assert hit_at_k("tax_validator", ["invoice_parser", "audit_logger"], 2) is False


def test_hit_at_k_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        hit_at_k("tax_validator", ["tax_validator"], 0)


def test_compute_tool_at_1() -> None:
    rankings = {
        "task_001": ["tax_validator", "invoice_parser"],
        "task_002": ["summarizer", "email_writer"],
    }

    assert compute_tool_at_1(TASKS, rankings) == 0.5


def test_compute_tool_at_3() -> None:
    rankings = {
        "task_001": ["invoice_parser", "audit_logger", "tax_validator"],
        "task_002": ["summarizer", "rag_answerer", "email_writer"],
    }

    assert compute_tool_at_3(TASKS, rankings) == 1.0


def test_compute_memory_at_1() -> None:
    rankings = {
        "task_001": ["invoice_validation_policy"],
        "task_002": ["privacy_first_policy", "human_approval_policy"],
    }

    assert compute_memory_at_1(TASKS, rankings) == 0.5


def test_compute_evidence_at_3() -> None:
    rankings = {
        "task_001": ["dense_retrieval_note", "rag_eval_note", "invoice_tax_rules"],
        "task_002": ["audit_logging_requirements"],
    }

    assert compute_evidence_at_3(TASKS, rankings) == 1.0


def test_missing_rankings_count_as_misses() -> None:
    rankings = {"task_001": ["tax_validator"]}

    assert compute_tool_at_1(TASKS, rankings) == 0.5


def test_empty_rankings_count_as_misses() -> None:
    rankings = {"task_001": [], "task_002": ["email_writer"]}

    assert compute_tool_at_1(TASKS, rankings) == 0.5


def test_mean_calculation() -> None:
    assert mean([10.0, 20.0, 30.0]) == 20.0


def test_percentile_calculation() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 50.0) == 25.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 95.0) == pytest.approx(38.5)


def test_empty_numeric_sequence_rejected() -> None:
    with pytest.raises(ValueError, match="mean requires at least one value"):
        mean([])
    with pytest.raises(ValueError, match="percentile requires at least one value"):
        percentile([], 95.0)


def test_rerank_gain_calculation() -> None:
    assert compute_rerank_gain(before=0.65, after=0.8) == pytest.approx(0.15)
