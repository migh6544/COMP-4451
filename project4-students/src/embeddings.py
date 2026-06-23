from __future__ import annotations

import os
from typing import List

from chromadb import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI

from .config import EMBED_DIMENSIONS, EMBED_MODEL, OPENROUTER_BASE_URL


class OpenRouterEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function backed by OpenAI-compatible OpenRouter client."""

    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
        )

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
            dimensions=EMBED_DIMENSIONS,
        )
        return [item.embedding for item in response.data]
