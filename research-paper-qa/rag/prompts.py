"""System prompt and preset questions for the Research Paper Q&A app."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a research assistant helping a student understand an academic paper.
Answer ONLY using the context excerpts provided below. Do not use outside knowledge.
If the context does not contain the answer, say: "This information was not found in the paper."
Be concise and structured (short paragraphs or bullet points). Cite page numbers like (p. 4)
after each claim. Preserve numbers, metric names, and dataset names exactly as written."""

PRESET_QUESTIONS: dict[str, str] = {
    "Method": (
        "What method, model, or approach does this paper propose? "
        "Explain the key idea, the architecture or algorithm steps, and what is novel compared to prior work."
    ),
    "Dataset": (
        "Which datasets are used in this paper? For each, give the name, size, "
        "source, train/validation/test split, and any preprocessing or annotation details."
    ),
    "Results": (
        "What are the main experimental results? Include key metrics, comparisons with "
        "baselines, and any ablation findings. Mention the numbers reported."
    ),
    "Limitations": (
        "What limitations, weaknesses, or failure cases do the authors mention? "
        "Also note any assumptions and future work they suggest."
    ),
    "Summarize": (
        "Give a TL;DR of this paper in 3-5 sentences covering: the problem it solves, "
        "the proposed approach, the main results, and the key contributions."
    ),
}

PRESET_K = 8  # Larger k for broad preset questions
