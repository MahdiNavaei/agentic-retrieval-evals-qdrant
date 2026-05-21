# Offline And Restricted Network Notes

## Why Local-Memory Mode Exists

The benchmark defaults to Qdrant local-memory mode so it can run without Docker or a running Qdrant server. This is useful when Docker Desktop is unavailable, when image pulls fail, or when the environment has restricted network access.

Local-memory mode uses qdrant-client local execution:

```powershell
python -m app.evaluate --strategy dense --qdrant-mode local-memory --recreate-collections --write-report reports/agentic_retrieval_results.md
```

## Running Without Docker

Install Python dependencies, then use `--qdrant-mode local-memory`:

```powershell
pip install -r requirements.txt
python -m pytest
python -m app.evaluate --strategy dense --qdrant-mode local-memory --recreate-collections --write-report reports/agentic_retrieval_results.md
```

Use local-disk mode when you want local Qdrant state persisted:

```powershell
python -m app.evaluate --strategy dense --qdrant-mode local-disk --qdrant-path .qdrant_local --recreate-collections --write-report reports/agentic_retrieval_results.md
```

## Docker/Server Mode Is Optional

Docker/server mode is available for environments with Docker and Qdrant server access:

```powershell
docker compose up -d
python -m app.evaluate --strategy dense --qdrant-mode server --qdrant-url http://localhost:6333 --recreate-collections --write-report reports/agentic_retrieval_results.md
```

Restricted networks may block Docker Hub image pulls or related infrastructure. Treat server mode as optional, not the default path.

## FastEmbed Model Downloads

The benchmark is local-first, but first-run FastEmbed model downloads may require network access unless the required model files are already cached. The default models are:

- dense: `BAAI/bge-small-en-v1.5`
- sparse: `Qdrant/bm25`
- reranker: `Xenova/ms-marco-MiniLM-L-6-v2`

Do not claim fully offline execution unless dependencies and model files are already cached locally.

## Suggested Restricted-Network Workflow

1. Install dependencies in an environment where package access is available.
2. Pre-cache FastEmbed model files if the target environment cannot download them.
3. Use `--qdrant-mode local-memory` by default.
4. Run `python -m pytest` before benchmark commands.
5. Run one strategy first, usually `dense`, before running rerank strategies.
6. Record any model download or network failure honestly in the report.

## What This Project Does Not Require

The benchmark does not require:

- OpenAI API keys
- Anthropic API keys
- external LLM APIs
- paid APIs
- cloud services
- Docker for the default local-memory path
