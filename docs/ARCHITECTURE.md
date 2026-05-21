# Architecture

## System Overview

The Tool-Memory-Evidence Retrieval Benchmark is a local evaluation harness for comparing retrieval strategies across three agentic retrieval targets:

- tool retrieval
- memory retrieval
- evidence retrieval

It is intentionally small. The system loads a synthetic JSONL dataset, validates deterministic ground-truth references, indexes each target surface in Qdrant, retrieves candidates with FastEmbed-backed strategies, computes pure metrics, and optionally writes a markdown report.

The benchmark is not a service, UI, or full agent framework.

## Execution Flow

1. Load dataset files from `data/`.
2. Validate JSONL structure, item ids, item types, duplicate ids, and task references.
3. Initialize Qdrant using `local-memory`, `local-disk`, or `server` mode.
4. Embed tools, memories, and evidence with the selected strategy.
5. Create or recreate Qdrant collections for dense and/or sparse vectors.
6. Upsert retrievable items with stable UUIDv5 point ids and simple payloads.
7. Retrieve tool, memory, and evidence candidates for each task.
8. Optionally rerank candidates with a FastEmbed cross-encoder.
9. Compute Tool@1, Tool@3, Memory@1, Evidence@3, mean latency, P95 latency, and gains versus dense where applicable.
10. Print a terminal summary and optionally write `reports/agentic_retrieval_results.md`.

## Component Map

`app/data.py`

- Loads JSONL records.
- Defines `RetrievableItem`, `BenchmarkTask`, and `BenchmarkDataset`.
- Validates required fields, item types, duplicate ids, and expected target references.

`app/metrics.py`

- Computes deterministic metrics over already-ranked ids.
- Has no Qdrant or FastEmbed dependency.
- Provides hit-rate metrics, mean, percentile, and rerank gain helpers.

`app/embed.py`

- Wraps FastEmbed dense embeddings.
- Wraps FastEmbed sparse embeddings.
- Wraps FastEmbed cross-encoder reranking.
- Keeps FastEmbed imports inside runtime wrappers so unit tests do not download models.

`app/qdrant_store.py`

- Creates Qdrant clients for `local-memory`, `local-disk`, and `server`.
- Manages dense and sparse collection creation.
- Upserts dense and sparse points.
- Searches dense and sparse collections and returns ranked item ids.

`app/evaluate.py`

- Owns CLI parsing, strategy validation, benchmark orchestration, latency measurement, terminal summaries, gain calculation, and report writing.
- Keeps supported strategies explicit: `dense`, `dense-rerank`, `sparse`, and `hybrid-rerank`.

## Retrieval Target Model

Tool retrieval asks whether the benchmark selected the right capability for the user task.

Memory retrieval asks whether the benchmark selected the right durable instruction, policy, preference, or safety rule.

Evidence retrieval asks whether the benchmark selected the right grounding document or note for the action or answer.

Each task contains one expected id for each target:

- `expected_tool`
- `expected_memory`
- `expected_evidence`

## Strategy Descriptions

`dense`

Embeds each query and item with `BAAI/bge-small-en-v1.5`, stores dense vectors in Qdrant, and searches with cosine distance. This is the baseline.

`sparse`

Embeds text with `Qdrant/bm25`, stores sparse vectors in Qdrant using the `text-sparse` vector name, and searches sparse collections. This strategy favors exact and procedural term overlap.

`dense-rerank`

Retrieves dense candidates, then reranks candidate texts with `Xenova/ms-marco-MiniLM-L-6-v2`. This can improve top-rank quality but adds cross-encoder latency.

`hybrid-rerank`

Retrieves dense and sparse candidates, merges and deduplicates ids, then reranks the combined candidate list. This compares a simple hybrid recall pattern against dense-only reranking.

## Qdrant Modes

`local-memory`

Uses qdrant-client local in-memory mode. It is the default because it avoids Docker, server setup, and network-dependent image pulls.

`local-disk`

Uses qdrant-client local disk mode and persists local state under the configured path.

`server`

Connects to a running Qdrant server, typically started with Docker Compose.

## Why Local-Memory Is Default

The project is designed to be runnable in restricted development environments. Docker Desktop may be unavailable, and Docker image pulls can fail behind restricted networks. `local-memory` keeps the default path reproducible without requiring a Qdrant server.

FastEmbed model files may still need to be downloaded on first run unless already cached.

## Boundaries And Non-Goals

This repository does not implement:

- FastAPI or a service layer
- UI
- external LLM API clients
- paid APIs
- cloud dependencies
- a full agent framework
- new benchmark claims beyond the small synthetic dataset

## Determinism And Reproducibility

Dataset validation is deterministic. Point ids are generated with stable UUIDv5 values. Metrics operate on explicit ranked ids. Reported runtime latency can vary by machine, cache state, and whether FastEmbed has to load or download model files.

The benchmark report should only contain metrics from commands that actually ran.

## Phase 5 Boundary

Phase 5 is documentation, packaging, presentation, and readiness work. It does not add strategies, change labels, tune model choices, or alter metric definitions.

Future work starts after human review and may include upstream discussion, optional per-task failure reporting, larger synthetic coverage, or maintainer-directed packaging changes.
