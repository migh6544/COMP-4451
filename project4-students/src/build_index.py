from __future__ import annotations

from .config import DOCS_DIR
from .document_loader import load_documents
from .chunking import chunk_documents
from .vector_store import index_chunks


def build_or_rebuild_index(reset: bool = True) -> str:
    docs = load_documents(DOCS_DIR)
    if not docs:
        return f"No supported documents found in {DOCS_DIR}. Add .txt, .md, .pdf, or .docx files."
    chunks = chunk_documents(docs)
    collection = index_chunks(chunks, reset=reset)
    return f"Indexed {len(chunks)} chunks from {len(docs)} documents into Chroma collection '{collection.name}'."


if __name__ == "__main__":
    print(build_or_rebuild_index(reset=True))
