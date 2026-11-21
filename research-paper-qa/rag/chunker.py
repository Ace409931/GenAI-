"""Recursive character-based text chunker with page metadata."""

from __future__ import annotations

import re
from typing import TypedDict

from .pdf_loader import PageData

CHUNK_SIZE = 800   # characters
CHUNK_OVERLAP = 150  # characters


class Chunk(TypedDict):
    chunk_id: int
    page: int
    text: str


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split a single string into overlapping windows, trying to break at sentence
    or paragraph boundaries first."""
    # Sentence-aware split: prefer splitting at '. ', '\n\n', '\n', then ' '
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def _split(t: str, seps: list[str]) -> list[str]:
        if not seps or len(t) <= size:
            return [t] if t else []
        sep = seps[0]
        rest = seps[1:]
        parts = t.split(sep)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) > size and current:
                chunks.extend(_split(current, rest))
                current = part
            else:
                current = candidate
        if current:
            chunks.extend(_split(current, rest))
        return chunks

    raw_chunks = _split(text, separators)

    # Re-merge small pieces and apply overlap sliding window
    merged: list[str] = []
    buffer = ""
    for piece in raw_chunks:
        if len(buffer) + len(piece) + 1 <= size:
            buffer = (buffer + " " + piece).strip() if buffer else piece
        else:
            if buffer:
                merged.append(buffer)
            buffer = piece
    if buffer:
        merged.append(buffer)

    # Apply overlap
    if overlap == 0 or len(merged) <= 1:
        return merged

    result: list[str] = []
    for i, chunk in enumerate(merged):
        if i == 0:
            result.append(chunk)
        else:
            # Prepend tail of previous chunk
            prev_tail = merged[i - 1][-overlap:]
            combined = (prev_tail + " " + chunk).strip()
            result.append(combined[:size + overlap])  # cap to avoid runaway growth
    return result


def chunk_pages(pages: list[PageData], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Split page texts into overlapping chunks with metadata.

    Args:
        pages: Output from :func:`rag.pdf_loader.load_pdf`.
        chunk_size: Target character length per chunk.
        overlap: Overlap between consecutive chunks in characters.

    Returns:
        Ordered list of :class:`Chunk` dicts.
    """
    chunks: list[Chunk] = []
    chunk_id = 0

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"].strip()
        if not text:
            continue

        splits = _split_text(text, chunk_size, overlap)
        for split in splits:
            split = split.strip()
            if split:
                chunks.append({"chunk_id": chunk_id, "page": page_num, "text": split})
                chunk_id += 1

    return chunks
