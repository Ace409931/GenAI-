"""Sentence-transformer embedding wrapper (runs locally)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton).

    Tries loading from local HuggingFace cache first (no network needed after
    first download). Falls back to downloading if not cached yet.
    """
    global _model
    if _model is None:
        try:
            # Use cached model without hitting the network
            _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:
            # First run: download from HuggingFace Hub
            _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into a 2-D float32 numpy array.

    Args:
        texts: Strings to embed.

    Returns:
        Array of shape ``(len(texts), embedding_dim)`` with float32 values.
    """
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string.

    Args:
        query: Question or search string.

    Returns:
        1-D float32 array of shape ``(embedding_dim,)``.
    """
    return embed_texts([query])[0]
