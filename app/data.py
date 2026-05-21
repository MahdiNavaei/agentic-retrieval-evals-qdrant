"""Dataset loading and validation for the benchmark JSONL files.

Phase 2 keeps this module intentionally independent of retrieval systems. It
validates local JSONL contracts and ground-truth references only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RetrievableItem:
    """A tool, memory, or evidence item that can be retrieved in later phases."""

    id: str
    text: str
    type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkTask:
    """A benchmark query with expected retrieval targets."""

    id: str
    query: str
    expected_tool: str
    expected_memory: str
    expected_evidence: str


@dataclass(frozen=True)
class BenchmarkDataset:
    """Validated benchmark dataset loaded from the data directory."""

    tools: list[RetrievableItem]
    memories: list[RetrievableItem]
    evidence: list[RetrievableItem]
    tasks: list[BenchmarkTask]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, skipping blank lines and reporting path/line errors."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def load_items(path: Path, expected_type: str) -> list[RetrievableItem]:
    """Load and validate retrievable items from a JSONL file."""

    items: list[RetrievableItem] = []
    for index, record in enumerate(load_jsonl(path), start=1):
        item_id = _required_string(record, "id", path, index)
        text = _required_string(record, "text", path, index)
        item_type = _required_string(record, "type", path, index)
        if item_type != expected_type:
            raise ValueError(
                f"{path}:{index}: expected type {expected_type!r}, got {item_type!r}"
            )

        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}:{index}: field 'metadata' must be an object when present")

        items.append(
            RetrievableItem(
                id=item_id,
                text=text,
                type=item_type,
                metadata=metadata,
            )
        )
    return items


def load_tasks(path: Path) -> list[BenchmarkTask]:
    """Load and validate benchmark tasks from a JSONL file."""

    tasks: list[BenchmarkTask] = []
    for index, record in enumerate(load_jsonl(path), start=1):
        tasks.append(
            BenchmarkTask(
                id=_required_string(record, "id", path, index),
                query=_required_string(record, "query", path, index),
                expected_tool=_required_string(record, "expected_tool", path, index),
                expected_memory=_required_string(record, "expected_memory", path, index),
                expected_evidence=_required_string(record, "expected_evidence", path, index),
            )
        )
    return tasks


def validate_unique_ids(items: Sequence[RetrievableItem], label: str) -> None:
    """Reject duplicate item ids with a clear dataset label."""

    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"duplicate {label} id: {item.id}")
        seen.add(item.id)


def validate_task_references(
    tasks: Sequence[BenchmarkTask],
    tools: Sequence[RetrievableItem],
    memories: Sequence[RetrievableItem],
    evidence: Sequence[RetrievableItem],
) -> None:
    """Validate that every task points to existing expected target ids."""

    tool_ids = {item.id for item in tools}
    memory_ids = {item.id for item in memories}
    evidence_ids = {item.id for item in evidence}

    for task in tasks:
        if task.expected_tool not in tool_ids:
            raise ValueError(
                f"task {task.id!r} references missing tool {task.expected_tool!r}"
            )
        if task.expected_memory not in memory_ids:
            raise ValueError(
                f"task {task.id!r} references missing memory {task.expected_memory!r}"
            )
        if task.expected_evidence not in evidence_ids:
            raise ValueError(
                f"task {task.id!r} references missing evidence {task.expected_evidence!r}"
            )


def load_benchmark_dataset(data_dir: Path) -> BenchmarkDataset:
    """Load all benchmark JSONL files and validate cross-file references."""

    tools = load_items(data_dir / "tools.jsonl", "tool")
    memories = load_items(data_dir / "memories.jsonl", "memory")
    evidence = load_items(data_dir / "evidence.jsonl", "evidence")
    tasks = load_tasks(data_dir / "tasks.jsonl")

    validate_unique_ids(tools, "tool")
    validate_unique_ids(memories, "memory")
    validate_unique_ids(evidence, "evidence")
    validate_task_references(tasks, tools, memories, evidence)

    return BenchmarkDataset(
        tools=tools,
        memories=memories,
        evidence=evidence,
        tasks=tasks,
    )


def _required_string(record: Mapping[str, Any], field_name: str, path: Path, index: int) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{index}: field {field_name!r} must be a non-empty string")
    return value
