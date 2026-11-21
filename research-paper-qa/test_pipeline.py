"""Quick smoke test for the RAG pipeline — run with: python test_pipeline.py"""
import sys
import pymupdf

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_pages
from rag.embedder import embed_query
from rag.vector_store import build_index
from rag.prompts import PRESET_QUESTIONS, SYSTEM_PROMPT, PRESET_K

print("All RAG modules imported OK")
print("Preset questions:", list(PRESET_QUESTIONS.keys()))
print("PRESET_K:", PRESET_K)
print()

# ── Build a minimal in-memory PDF ────────────────────────────────────────────
doc = pymupdf.open()
page = doc.new_page()
page.insert_text(
    (50, 72),
    "Abstract: This paper proposes FlashAttention, a novel IO-aware exact attention "
    "algorithm using tiling to reduce memory reads/writes. It achieves 15% wall-clock "
    "speedup on BERT-large training compared to the baseline.",
)
page2 = doc.new_page()
page2.insert_text(
    (50, 72),
    "Datasets: We evaluate on ImageNet (1.28M images) and CIFAR-10 (60k images, "
    "10 classes). Train/val split is 90/10. No special preprocessing was applied.",
)
page3 = doc.new_page()
page3.insert_text(
    (50, 72),
    "Limitations: FlashAttention requires custom CUDA kernels, making it harder to "
    "port to non-NVIDIA hardware. Future work includes extension to TPUs.",
)
pdf_bytes = doc.tobytes()
doc.close()

# ── PDF loading ───────────────────────────────────────────────────────────────
pages = load_pdf(pdf_bytes)
print(f"Pages extracted: {len(pages)}")
assert len(pages) == 3, "Expected 3 pages"

# ── Chunking ──────────────────────────────────────────────────────────────────
chunks = chunk_pages(pages)
print(f"Chunks created: {len(chunks)}")
assert len(chunks) >= 1

# ── Indexing ──────────────────────────────────────────────────────────────────
store = build_index(chunks)
print("FAISS index built OK")

# ── Search ────────────────────────────────────────────────────────────────────
results = store.search("What method does the paper propose?", k=3)
print(f"Search returned {len(results)} results")
for r in results:
    page_num = r.chunk["page"]
    score = r.score
    snippet = r.chunk["text"][:70]
    print(f"  Page {page_num} | L2={score:.4f} | {snippet!r}")

assert len(results) > 0, "Expected at least one search result"

results2 = store.search("What datasets are used?", k=2)
assert any("ImageNet" in r.chunk["text"] for r in results2), "Dataset result missing"
print()
print("END-TO-END RAG PIPELINE: ALL CHECKS PASSED ✓")
