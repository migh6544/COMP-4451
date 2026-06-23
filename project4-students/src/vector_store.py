from __future__ import annotations

import chromadb
from chromadb.config import Settings

from .chunking import TextChunk
from .config import CHROMA_DIR, COLLECTION_NAME
from .embeddings import OpenRouterEmbeddingFunction


def get_chroma_collection(reset: bool = False):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    embedding_fn = OpenRouterEmbeddingFunction()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[TextChunk], reset: bool = False):
    collection = get_chroma_collection(reset=reset)
    if collection.count() > 0 and not reset:
        return collection

    ids = [c.chunk_id for c in chunks]
    docs = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metadatas[start:end],
        )
    return collection


def query_collection(query: str, top_k: int = 6):
    collection = get_chroma_collection(reset=False)
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    rows = []
    for i, doc in enumerate(results.get("documents", [[]])[0]):
        rows.append({
            "rank": i + 1,
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "embedding": results.get("embeddings", [[]])[0][i],
        })
    return rows
