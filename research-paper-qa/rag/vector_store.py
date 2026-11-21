"""FAISS in-memory vector store: build index and search."""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from .chunker import Chunk
from .embedder import embed_texts, embed_query


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # L2 distance (lower = more similar)


class VectorStore:
    """Wraps a FAISS flat-L2 index over a list of text chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        """Build a FAISS index from chunks.

        Args:
            chunks: List of :class:`~rag.chunker.Chunk` dicts.
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the top-k most similar chunks for *query*.

        Args:
            query: Question or search text.
            k: Number of results to retrieve.

        Returns:
            List of :class:`SearchResult` sorted by ascending L2 distance.
        """
        k = min(k, len(self.chunks))
        q_emb = embed_query(query).reshape(1, -1)
        distances, indices = self.index.search(q_emb, k)
        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(dist)))
        return results


def build_index(chunks: list[Chunk]) -> VectorStore:
    """Convenience wrapper around :class:`VectorStore`.

    Args:
        chunks: Chunks to index.

    Returns:
        A ready :class:`VectorStore` instance.
    """
    return VectorStore(chunks)
