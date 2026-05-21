"""FastEmbed wrappers for dense, sparse, and rerank strategies.

The wrappers keep FastEmbed imports local so unit tests can validate contracts
without downloading model files. Runtime model initialization may download
FastEmbed assets on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class SparseVectorData:
    """Plain Python sparse vector representation shared with Qdrant helpers."""

    indices: list[int]
    values: list[float]


class DenseEmbedder:
    """Small typed wrapper around FastEmbed dense text embeddings."""

    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        try:
            from fastembed import TextEmbedding
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "fastembed is not installed. Install dependencies with "
                "`pip install -r requirements.txt` before running the dense benchmark."
            ) from exc

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed non-empty document texts as plain Python float vectors."""

        _validate_texts(texts)
        return [_to_float_vector(vector) for vector in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        """Embed one non-empty query as a plain Python float vector."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("query text must be a non-empty string")

        if hasattr(self._model, "query_embed"):
            vectors = self._model.query_embed(text)
        else:
            vectors = self._model.embed([text])
        return _first_vector(vectors)


class SparseEmbedder:
    """Small typed wrapper around FastEmbed sparse text embeddings."""

    def __init__(self, model_name: str = DEFAULT_SPARSE_MODEL) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        try:
            from fastembed import SparseTextEmbedding
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "fastembed is not installed. Install dependencies with "
                "`pip install -r requirements.txt` before running sparse strategies."
            ) from exc

        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVectorData]:
        """Embed non-empty document texts as sparse vectors."""

        _validate_texts(texts)
        return [_to_sparse_vector(vector) for vector in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> SparseVectorData:
        """Embed one non-empty query as a sparse vector."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("query text must be a non-empty string")

        vectors = self._model.query_embed(text)
        for vector in vectors:
            return _to_sparse_vector(vector)
        raise ValueError("sparse embedding model returned no vectors")


class CrossEncoderReranker:
    """FastEmbed cross-encoder reranker for candidate text lists."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "FastEmbed TextCrossEncoder is unavailable in this environment. "
                "Install a FastEmbed version with rerank cross-encoder support."
            ) from exc

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, candidate_texts: Sequence[str]) -> list[int]:
        """Return candidate indexes sorted by descending cross-encoder score."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query text must be a non-empty string")
        _validate_texts(candidate_texts)

        scores = [float(score) for score in self._model.rerank(query, list(candidate_texts))]
        if len(scores) != len(candidate_texts):
            raise ValueError("reranker returned a score count that does not match candidates")
        return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)


def _validate_texts(texts: Sequence[str]) -> None:
    if not texts:
        raise ValueError("texts must contain at least one item")
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"texts[{index}] must be a non-empty string")


def _first_vector(vectors: Iterable[object]) -> list[float]:
    for vector in vectors:
        return _to_float_vector(vector)
    raise ValueError("embedding model returned no vectors")


def _to_float_vector(vector: object) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]  # type: ignore[union-attr]


def _to_sparse_vector(vector: object) -> SparseVectorData:
    indices = getattr(vector, "indices", None)
    values = getattr(vector, "values", None)
    if indices is None or values is None:
        raise ValueError("sparse embedding model returned an invalid vector")
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return SparseVectorData(
        indices=[int(value) for value in indices],
        values=[float(value) for value in values],
    )
