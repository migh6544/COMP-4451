# COMP4451 Project 4/5 — Agentic RAG Course Assistant

## Contributors
- Michael Ghattas — project implementation, document selection, RAG design, interface, and documentation.

## Project Topic
This RAG app answers questions about Agentic RAG, ReAct, RAG pipeline evaluation, and RAG metric interpretation using a small cohesive document corpus in `data/documents/`.

The included corpus contains four text-based class-note documents:
1. `01_react_reasoning_acting.md`
2. `02_agentic_rag_taxonomy.md`
3. `03_week10_pipeline_results.md`
4. `04_rag_metrics_interpretation.md`

These documents are cohesive and designed for meaningful questions about ReAct, Agentic RAG, pipeline comparison, retrieval failure modes, and evaluation metrics.

## Features Mapped to the Rubric

| Requirement | Implementation |
|---|---|
| Well-structured executable code | Modular files in `src/` plus `app.py` |
| Document loading and preprocessing | `src/document_loader.py` supports `.txt`, `.md`, `.pdf`, `.docx` |
| Optimized chunking | `src/chunking.py` splits on headings/sections first, then sentence windows with overlap |
| Embedding and indexing | `src/vector_store.py` stores IDs, text, and metadata in persistent Chroma |
| OpenAI/OpenRouter embeddings | `src/embeddings.py` uses `text-embedding-3-small` with 512 dimensions |
| Similarity search | Chroma cosine search returns top-k chunks |
| Prompt restriction | `src/rag_pipeline.py` instructs the model to answer only from retrieved context |
| Memory/history | Gradio app keeps prior turns in `gr.State` and injects recent history |
| Gradio UI | `app.py` provides chat, source chunks, index rebuild button, and top-k control |
| 3-D embedding visualization | `src/visualization.py` uses PCA + Plotly 3-D scatter with hover previews |

## Setup

Create and activate a virtual environment in the project folder.

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and set:

```text
OPENROUTER_API_KEY=your_openrouter_key_here
```

## Run the App

From the project folder:

```bash
python app.py
```

Open the local Gradio URL shown in the terminal.

First click **Rebuild Chroma Index**. This parses the documents, creates structure-aware overlapping chunks, embeds them, and stores them in persistent Chroma. Then ask questions in the chat.

## Example Questions

- What is ReAct and how is it different from chain-of-thought prompting?
- Why did the ReAct pipeline perform better than hybrid retrieval on the Week 10 benchmark?
- When is query decomposition useful?
- Why can faithfulness be high while correctness is low?
- Why is ROUGE weak as a standalone RAG metric?

## Source Files

```text
app.py
src/config.py
src/document_loader.py
src/chunking.py
src/embeddings.py
src/vector_store.py
src/rag_pipeline.py
src/safety.py
src/visualization.py
src/build_index.py
data/documents/*.md
requirements.txt
.env.example
```

## Notes

- The app uses `gpt-4o-mini` for answer generation.
- The app uses `text-embedding-3-small` with 512 dimensions for embeddings.
- Chroma is persistent in `chroma_db/` after indexing.
- The model is instructed to abstain if retrieved chunks do not support the answer.
- The 3-D plot visualizes the returned chunk embeddings, not the full corpus.
