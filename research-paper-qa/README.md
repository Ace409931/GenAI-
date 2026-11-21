# Research Paper Q&A 📚

A Streamlit web app that lets students upload a research paper PDF and ask questions about it using **RAG (Retrieval-Augmented Generation)**. Answers are grounded in the paper and cite page numbers.

---

## Features

- 📄 **PDF Upload** — drag & drop or browse; extracts text page by page with PyMuPDF
- 🔍 **Semantic Search** — FAISS vector index with `all-MiniLM-L6-v2` embeddings (runs locally, no API needed for retrieval)
- 🤖 **Gemini-powered answers** — grounded strictly in the paper; refuses to hallucinate
- 🔬 **Preset buttons** — one-click questions: **Method**, **Dataset**, **Results**, **Limitations**, **Summarize**
- 📌 **Sources expander** — shows retrieved chunks with page numbers under every answer
- 💬 **Chat history** — full conversation retained per session; clears on new PDF upload
- ⬇️ **Download chat** — export conversation as Markdown from the sidebar

---

## Project Structure

```
research-paper-qa/
├── app.py                 # Streamlit UI
├── rag/
│   ├── __init__.py
│   ├── pdf_loader.py      # Extract text per page (PyMuPDF)
│   ├── chunker.py         # Recursive splitter with page metadata
│   ├── embedder.py        # sentence-transformers wrapper
│   ├── vector_store.py    # FAISS index build + search
│   ├── prompts.py         # System prompt + preset questions
│   └── qa_chain.py        # Retrieve → build prompt → call LLM
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone / download the project

```bash
cd research-paper-qa
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `faiss-cpu` and `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) on first run.

### 4. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

`.env` contents:
```
GEMINI_API_KEY=your_actual_key_here
LLM_MODEL=gemini-2.5-flash
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com/).

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. **Upload a PDF** in the left sidebar (max 30 MB, must have extractable text).
2. Wait for the spinner — indexing a 15-page paper takes ~5–15 seconds.
3. Click a **preset button** (Method / Dataset / Results / Limitations / Summarize) or type a free-form question.
4. Each answer cites page numbers and shows a **Sources** expander.
5. Use the **k** slider to control how many chunks are retrieved (3–10).
6. **Clear chat** resets the conversation; uploading a new PDF also resets it.
7. **Download chat** exports the conversation as a `.md` file.

---

## Configuration

| `.env` variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | _(required)_ | Your Google Gemini API key |
| `LLM_MODEL` | `gemini-2.5-flash` | Any Gemini model name (e.g. `gemini-2.0-flash`) |

The LLM call is isolated in `rag/qa_chain.py::_llm_call()` — swap the provider by editing that single function.

---

## Architecture

```
PDF bytes
  └─► pdf_loader.py   → [{page, text}, ...]
        └─► chunker.py    → [{chunk_id, page, text}, ...]
              └─► embedder.py   → float32 numpy arrays
                    └─► vector_store.py  → FAISS index
                                            │
User question ────────────────────────────►│ search(k)
                                            │
                                   top-k chunks + scores
                                            │
                                   qa_chain.py
                                            │  build prompt
                                            │  call Gemini
                                            ▼
                                   {answer, sources}
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| _"No extractable text found"_ | PDF is scanned/image-based; use a text-layer PDF or run OCR first |
| _"GEMINI_API_KEY is not set"_ | Create `.env` from `.env.example` and add your key |
| Rate limit errors | The app auto-retries 3×; if persists, check your Gemini quota |
| Slow first run | `all-MiniLM-L6-v2` model is downloaded (~90 MB) on first use |

---

## License

MIT
