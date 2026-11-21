"""Research Paper Q&A — Streamlit app (RAG over PDF)."""

from __future__ import annotations

import io
import time
from datetime import datetime

import streamlit as st

from rag.chunker import chunk_pages
from rag.pdf_loader import load_pdf
from rag.prompts import PRESET_QUESTIONS, PRESET_K
from rag.qa_chain import answer
from rag.vector_store import build_index, VectorStore

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📄",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .preset-row button { margin-right: 6px; }
    .stChatMessage { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

MAX_FILE_SIZE_MB = 30

# ─── Session state defaults ──────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict = {
        "messages": [],         # list[dict] with role + content + sources
        "store": None,          # VectorStore | None
        "chunks": [],
        "page_count": 0,
        "chunk_count": 0,
        "current_file": None,   # name of currently indexed PDF
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Paper")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

    st.divider()
    top_k = st.slider("Retrieval chunks (k)", min_value=3, max_value=10, value=5)

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    # Download chat as Markdown
    if st.session_state.messages:
        lines: list[str] = [f"# Research Paper Q&A — {datetime.now().strftime('%Y-%m-%d')}\n"]
        for msg in st.session_state.messages:
            role = "**You**" if msg["role"] == "user" else "**Assistant**"
            lines.append(f"{role}: {msg['content']}\n")
        md_bytes = "\n".join(lines).encode("utf-8")
        st.download_button(
            "⬇️ Download chat (.md)",
            data=md_bytes,
            file_name="chat_export.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    if st.session_state.store:
        st.success(
            f"**Indexed:** {st.session_state.current_file}\n\n"
            f"Pages: {st.session_state.page_count} | Chunks: {st.session_state.chunk_count}"
        )

# ─── PDF Indexing ─────────────────────────────────────────────────────────────
if uploaded is not None:
    # Only reindex when a new file arrives
    if uploaded.name != st.session_state.current_file:
        file_bytes = uploaded.read()

        # Size check
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ File too large ({size_mb:.1f} MB). Limit is {MAX_FILE_SIZE_MB} MB.")
            st.stop()

        with st.spinner(f"🔍 Indexing **{uploaded.name}** …"):
            try:
                pages = load_pdf(file_bytes)
                chunks = chunk_pages(pages)
                store = build_index(chunks)
            except ValueError as exc:
                st.error(f"❌ {exc}")
                st.stop()
            except RuntimeError as exc:
                st.error(f"❌ {exc}")
                st.stop()

        # Persist to session
        st.session_state.store = store
        st.session_state.chunks = chunks
        st.session_state.page_count = len(pages)
        st.session_state.chunk_count = len(chunks)
        st.session_state.current_file = uploaded.name
        st.session_state.messages = []  # reset chat on new paper
        st.success(
            f"✅ Indexed **{uploaded.name}** — {len(pages)} pages, {len(chunks)} chunks."
        )

# ─── Main area ────────────────────────────────────────────────────────────────
st.title("📚 Research Paper Q&A")
st.caption("Upload a research paper PDF and ask questions grounded in its content.")

indexed = st.session_state.store is not None


# ─── Preset buttons ───────────────────────────────────────────────────────────
st.markdown("**Quick questions:**")
preset_cols = st.columns(len(PRESET_QUESTIONS))
preset_clicked: str | None = None

button_icons = {
    "Method": "🔬",
    "Dataset": "🗂️",
    "Results": "📊",
    "Limitations": "⚠️",
    "Summarize": "📝",
}

for col, (label, _) in zip(preset_cols, PRESET_QUESTIONS.items()):
    icon = button_icons.get(label, "")
    if col.button(f"{icon} {label}", disabled=not indexed, use_container_width=True):
        preset_clicked = label

st.divider()

# ─── Render existing chat ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📌 Sources"):
                for src in msg["sources"]:
                    chunk = src.chunk
                    st.markdown(
                        f"**Page {chunk['page']}** _(chunk #{chunk['chunk_id']})_"
                    )
                    st.caption(chunk["text"])
                    st.divider()


# ─── Helper: run RAG and append to chat ───────────────────────────────────────
def _run_qa(question: str, k: int) -> None:
    """Append user message, run RAG, append assistant reply."""
    # User message
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    # Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            try:
                result = answer(question, st.session_state.store, k=k)
            except RuntimeError as exc:
                st.error(f"❌ {exc}")
                return
            except EnvironmentError as exc:
                st.error(f"❌ {exc}")
                return

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📌 Sources"):
                for src in result["sources"]:
                    chunk = src.chunk
                    st.markdown(
                        f"**Page {chunk['page']}** _(chunk #{chunk['chunk_id']})_"
                    )
                    st.caption(chunk["text"])
                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )


# ─── Handle preset button clicks ─────────────────────────────────────────────
if preset_clicked and indexed:
    question_text = PRESET_QUESTIONS[preset_clicked]
    _run_qa(question_text, k=PRESET_K)

# ─── Handle free-form chat input ─────────────────────────────────────────────
user_input = st.chat_input(
    "Ask anything about the paper…",
    disabled=not indexed,
)

if not indexed:
    st.info("⬆️ Upload a PDF in the sidebar to get started.")

if user_input and indexed:
    _run_qa(user_input, k=top_k)
