"""PDF text extraction using PyMuPDF."""

from __future__ import annotations

import io
from typing import TypedDict

import pymupdf as fitz  # PyMuPDF (new canonical import)


class PageData(TypedDict):
    page: int
    text: str


def load_pdf(file_bytes: bytes) -> list[PageData]:
    """Extract text from each page of a PDF.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        List of dicts with ``page`` (1-based) and ``text`` keys.

    Raises:
        ValueError: If the PDF has no extractable text (e.g. scanned).
        RuntimeError: If the file is corrupted or cannot be opened.
    """
    try:
        doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF: {exc}") from exc

    pages: list[PageData] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i, "text": text})

    doc.close()

    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned or image-based."
        )

    return pages
