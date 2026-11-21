"""Comprehensive integration test for Research Paper Q&A RAG app."""
import os
import sys
import pymupdf
from dotenv import load_dotenv

load_dotenv()

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_pages
from rag.vector_store import build_index
from rag.prompts import PRESET_QUESTIONS, PRESET_K
from rag.qa_chain import answer

print("=== 1. Checking Environment ===")
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("LLM_MODEL")
print(f"LLM_MODEL: {model}")
print(f"GEMINI_API_KEY configured: {bool(api_key)}")

print("\n=== 2. Creating Synthetic Research Paper PDF ===")
doc = pymupdf.open()

# Page 1: Method
p1 = doc.new_page()
p1.insert_text(
    (50, 72),
    "Title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\n\n"
    "1. Introduction & Method\n"
    "Transformers are slow and memory-hungry on long sequences. Standard attention has O(N^2) memory complexity "
    "because it materializes the full attention matrix in High Bandwidth Memory (HBM). "
    "We propose FlashAttention, an IO-aware exact attention algorithm. The key idea is tiling: "
    "we load blocks of Q, K, and V from slow HBM into fast SRAM, compute attention incrementally with online softmax, "
    "and write back results without ever storing the large N x N intermediate matrix. "
    "Compared to standard attention, FlashAttention reduces memory reads/writes by 2x to 4x."
)

# Page 2: Datasets & Experimental Setup
p2 = doc.new_page()
p2.insert_text(
    (50, 72),
    "2. Experiments and Datasets\n"
    "We evaluate FlashAttention on two primary benchmarks:\n"
    "1) BERT-large pre-training on Wikipedia and BookCorpus (total 3.3B tokens). "
    "We use standard train/val splits of 95% train and 5% validation.\n"
    "2) Long-range Arena (LRA) benchmark across sequence lengths 1K to 4K for ListOps, Text, Retrieval, Image, and Pathfinder.\n"
    "No data augmentation or extra preprocessing was applied beyond standard Byte-Pair Encoding (BPE) tokenization."
)

# Page 3: Results
p3 = doc.new_page()
p3.insert_text(
    (50, 72),
    "3. Results\n"
    "On BERT-large training, FlashAttention achieves 15% faster wall-clock training time compared to MLPerf 1.1 baselines "
    "(training time drops from 10.2 hours to 8.7 hours on 8x A100 GPUs). "
    "On GPT-2 with 1K context length, it speeds up forward/backward pass by 3.0x. "
    "On Long-Range Arena, FlashAttention is 2.4x faster than standard Transformer and matches exact attention accuracy (58.4% average score)."
)

# Page 4: Limitations
p4 = doc.new_page()
p4.insert_text(
    (50, 72),
    "4. Limitations & Future Work\n"
    "A main limitation is that FlashAttention requires custom low-level CUDA kernels written specifically for NVIDIA GPUs, "
    "making compilation and porting to non-NVIDIA accelerators (like Google TPUs or AMD GPUs) non-trivial. "
    "Additionally, performance depends on SRAM size per streaming multiprocessor. "
    "Future work includes extending IO-aware tiling to multi-query attention, linear RNNs, and graph transformers."
)

pdf_bytes = doc.tobytes()
doc.close()

print("\n=== 3. Running PDF Loader & Chunker ===")
pages = load_pdf(pdf_bytes)
print(f"Loaded {len(pages)} pages.")
assert len(pages) == 4

chunks = chunk_pages(pages)
print(f"Generated {len(chunks)} chunks.")
assert len(chunks) >= 4

print("\n=== 4. Building Vector Store ===")
store = build_index(chunks)
print("Vector store indexed successfully.")

print("\n=== 5. Testing Preset Questions with Live LLM ===")
for preset_name in ["Method", "Dataset", "Results", "Limitations"]:
    q = PRESET_QUESTIONS[preset_name]
    print(f"\n--- Testing Preset: [{preset_name}] ---")
    result = answer(q, store, k=PRESET_K)
    ans = result["answer"]
    sources = result["sources"]
    print(f"Answer snippet: {ans[:200]}...")
    print(f"Retrieved {len(sources)} sources.")
    assert len(sources) > 0
    assert len(ans.strip()) > 0
    print(f"✓ Preset '{preset_name}' answered successfully.")

print("\n=== 6. Testing Out-of-Scope Question (Grounding Check) ===")
unrelated_q = "What is the recipe for baking chocolate chip cookies?"
result_unrelated = answer(unrelated_q, store, k=3)
ans_unrelated = result_unrelated["answer"]
print(f"Question: {unrelated_q}")
print(f"Answer: {ans_unrelated}")
assert "not found" in ans_unrelated.lower() or "not mentioned" in ans_unrelated.lower() or "information was not found" in ans_unrelated.lower()
print("✓ Out-of-scope question handled correctly.")

print("\n==============================================")
print("🎉 ALL TESTS PASSED! THE SYSTEM WORKS PROPERLY.")
print("==============================================")
