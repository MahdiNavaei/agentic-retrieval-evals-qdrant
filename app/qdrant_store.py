"""Qdrant collection helpers for dense and sparse benchmark strategies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from app.data import RetrievableItem
from app.embed import SparseVectorData


COLLECTIONS = {
    "tool": "tme_tools",
    "memory": "tme_memories",
    "evidence": "tme_evidence",
}

SPARSE_COLLECTIONS = {
    "tool": "tme_tools_sparse",
    "memory": "tme_memories_sparse",
    "evidence": "tme_evidence_sparse",
}

SPARSE_VECTOR_NAME = "text-sparse"


def create_qdrant_client(mode: str, url: str, path: str) -> Any:
    """Create a Qdrant client for local or server-backed Phase 3 execution."""

    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qdrant-client is not installed. Install dependencies with "
            "`pip install -r requirements.txt` before running the dense benchmark."
        ) from exc

    if mode == "local-memory":
        return QdrantClient(":memory:")
    if mode == "local-disk":
        if not path.strip():
            raise ValueError("qdrant path must be a non-empty string for local-disk mode")
        return QdrantClient(path=path)
    if mode == "server":
        if not url.strip():
            raise ValueError("qdrant URL must be a non-empty string for server mode")
        return QdrantClient(url=url)
    raise ValueError(
        f"unsupported qdrant mode {mode!r}; expected local-memory, local-disk, or server"
    )


def deterministic_point_id(item: RetrievableItem) -> str:
    """Return a stable UUIDv5 point id for a retrievable item."""

    return str(uuid5(NAMESPACE_URL, f"tool-memory-evidence:{item.type}:{item.id}"))


def ensure_collection(
    client: Any,
    collection_name: str,
    vector_size: int,
    recreate: bool = False,
) -> None:
    """Create or optionally recreate a dense-vector Qdrant collection."""

    if vector_size < 1:
        raise ValueError("vector_size must be >= 1")

    models = _qdrant_models()
    vectors_config = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)

    if recreate:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )
        return

    if _collection_exists(client, collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
    )


def ensure_sparse_collection(
    client: Any,
    collection_name: str,
    recreate: bool = False,
) -> None:
    """Create or optionally recreate a sparse-vector Qdrant collection."""

    models = _qdrant_models()
    sparse_config = {
        SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF),
    }

    if recreate:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={},
            sparse_vectors_config=sparse_config,
        )
        return

    if _collection_exists(client, collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config=sparse_config,
    )


def upsert_items(
    client: Any,
    collection_name: str,
    items: Sequence[RetrievableItem],
    vectors: Sequence[Sequence[float]],
) -> None:
    """Upsert retrievable items with dense vectors and simple payloads."""

    if len(items) != len(vectors):
        raise ValueError("items and vectors must have the same length")

    models = _qdrant_models()
    points = [
        models.PointStruct(
            id=deterministic_point_id(item),
            vector=[float(value) for value in vector],
            payload={
                "id": item.id,
                "text": item.text,
                "type": item.type,
                "metadata": dict(item.metadata),
            },
        )
        for item, vector in zip(items, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)


def upsert_sparse_items(
    client: Any,
    collection_name: str,
    items: Sequence[RetrievableItem],
    vectors: Sequence[SparseVectorData],
) -> None:
    """Upsert retrievable items with sparse vectors and simple payloads."""

    if len(items) != len(vectors):
        raise ValueError("items and vectors must have the same length")

    models = _qdrant_models()
    points = [
        models.PointStruct(
            id=deterministic_point_id(item),
            vector={
                SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=list(vector.indices),
                    values=list(vector.values),
                )
            },
            payload={
                "id": item.id,
                "text": item.text,
                "type": item.type,
                "metadata": dict(item.metadata),
            },
        )
        for item, vector in zip(items, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)


def search_ids(
    client: Any,
    collection_name: str,
    query_vector: Sequence[float],
    top_k: int,
) -> list[str]:
    """Search a Qdrant collection and return ranked item ids from payloads."""

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    query = [float(value) for value in query_vector]

    if hasattr(client, "search"):
        points = client.search(
            collection_name=collection_name,
            query_vector=query,
            limit=top_k,
            with_payload=True,
        )
    else:
        response = client.query_points(
            collection_name=collection_name,
            query=query,
            limit=top_k,
            with_payload=True,
        )
        points = response.points

    ranked_ids: list[str] = []
    for point in points:
        payload = getattr(point, "payload", None) or {}
        item_id = payload.get("id")
        if isinstance(item_id, str):
            ranked_ids.append(item_id)
    return ranked_ids


def search_sparse_ids(
    client: Any,
    collection_name: str,
    query_vector: SparseVectorData,
    top_k: int,
) -> list[str]:
    """Search a sparse Qdrant collection and return ranked item ids."""

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    models = _qdrant_models()
    response = client.query_points(
        collection_name=collection_name,
        query=models.SparseVector(
            indices=list(query_vector.indices),
            values=list(query_vector.values),
        ),
        using=SPARSE_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
    )

    ranked_ids: list[str] = []
    for point in response.points:
        payload = getattr(point, "payload", None) or {}
        item_id = payload.get("id")
        if isinstance(item_id, str):
            ranked_ids.append(item_id)
    return ranked_ids


def _collection_exists(client: Any, collection_name: str) -> bool:
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection_name=collection_name))

    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        return False
    return True


def _qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qdrant-client is not installed. Install dependencies with "
            "`pip install -r requirements.txt` before running the dense benchmark."
        ) from exc
    return models
