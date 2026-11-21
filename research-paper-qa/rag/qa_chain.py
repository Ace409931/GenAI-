"""RAG Q&A chain: retrieve -> build prompt -> call LLM."""

from __future__ import annotations

import os
import time
from typing import TypedDict

from dotenv import load_dotenv
from google import genai

from .prompts import SYSTEM_PROMPT
from .vector_store import VectorStore, SearchResult

load_dotenv()

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Please add it to your .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _llm_call(prompt: str, retries: int = 3) -> str:
    """Call the Gemini LLM with retry and model fallback.

    Swapping the LLM provider requires changing only this function.

    Args:
        prompt: Full prompt string (system + context + question).
        retries: Number of retry attempts on transient errors.

    Returns:
        The model's response text.
    """
    primary_model = os.getenv("LLM_MODEL", "gemini-3.5-flash")
    client = _get_client()

    # Fallback model candidates if primary model is unavailable or rate-limited
    candidates = [primary_model]
    for fb in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        if fb not in candidates:
            candidates.append(fb)

    last_exc = None
    for active_model in candidates:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=active_model,
                    contents=prompt,
                )
                return response.text or ""
            except Exception as exc:
                last_exc = exc
                err = str(exc).lower()
                is_transient = any(
                    k in err for k in ["quota", "rate", "429", "503", "unavailable", "high demand", "overloaded"]
                )
                if is_transient and attempt < retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                break  # Try next candidate

    raise RuntimeError(f"Gemini API error: {last_exc}") from last_exc


class QAResult(TypedDict):
    answer: str
    sources: list[SearchResult]


def answer(
    question: str,
    store: VectorStore,
    k: int = 5,
) -> QAResult:
    """Run the full RAG pipeline for a question.

    Args:
        question: User's question.
        store: :class:`~rag.vector_store.VectorStore` built from the uploaded PDF.
        k: Number of chunks to retrieve.

    Returns:
        Dict with ``answer`` (str) and ``sources`` (list of :class:`~rag.vector_store.SearchResult`).
    """
    sources = store.search(question, k=k)

    # Build context block with page labels
    context_lines: list[str] = []
    for i, result in enumerate(sources, start=1):
        chunk = result.chunk
        context_lines.append(
            f"[Excerpt {i} | Page {chunk['page']}]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_lines)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT ===\n{context_text}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== ANSWER ==="
    )

    answer_text = _llm_call(prompt)

    return QAResult(answer=answer_text, sources=sources)
