from pathlib import Path

import pytest

from app.data import (
    RetrievableItem,
    load_benchmark_dataset,
    load_items,
    load_jsonl,
    validate_task_references,
    validate_unique_ids,
)


ROOT = Path(__file__).resolve().parents[1]


def test_load_benchmark_dataset_succeeds_with_real_data() -> None:
    dataset = load_benchmark_dataset(ROOT / "data")

    assert len(dataset.tools) == 10
    assert len(dataset.memories) == 10
    assert len(dataset.evidence) == 10
    assert len(dataset.tasks) == 20


def test_duplicate_item_ids_are_rejected() -> None:
    items = [
        RetrievableItem(id="same_id", text="First item.", type="tool"),
        RetrievableItem(id="same_id", text="Second item.", type="tool"),
    ]

    with pytest.raises(ValueError, match="duplicate tool id: same_id"):
        validate_unique_ids(items, "tool")


def test_missing_required_item_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tools.jsonl"
    path.write_text('{"id": "missing_text", "type": "tool"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="field 'text' must be a non-empty string"):
        load_items(path, "tool")


def test_wrong_item_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tools.jsonl"
    path.write_text(
        '{"id": "wrong_type", "text": "Wrong type.", "type": "memory"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected type 'tool', got 'memory'"):
        load_items(path, "tool")


def test_missing_task_references_are_rejected() -> None:
    dataset = load_benchmark_dataset(ROOT / "data")
    bad_task = dataset.tasks[0].__class__(
        id="task_bad_reference",
        query="Reference a missing tool.",
        expected_tool="missing_tool",
        expected_memory=dataset.memories[0].id,
        expected_evidence=dataset.evidence[0].id,
    )

    with pytest.raises(ValueError, match="task 'task_bad_reference' references missing tool"):
        validate_task_references([bad_task], dataset.tools, dataset.memories, dataset.evidence)


def test_invalid_jsonl_includes_path_and_line_number(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"id": "ok"}\n{"id": \n', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_jsonl(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert ":2:" in message
    assert "invalid JSON" in message
