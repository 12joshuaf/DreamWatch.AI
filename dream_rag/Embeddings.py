"""
embeddings.py

Thin wrapper around the OpenAI embeddings endpoint. Batches requests and
returns plain numpy arrays.
"""

from __future__ import annotations

import os

import numpy as np
from openai import OpenAI

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims, cheap & solid


class EmbeddingClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_EMBEDDING_MODEL):
        # Falls back to OPENAI_API_KEY env var if api_key isn't passed.
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of strings, returns shape (len(texts), embedding_dim)."""
        if not texts:
            return np.zeros((0, 0))

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            # response.data is returned in the same order as the input
            all_vectors.extend([item.embedding for item in response.data])

        return np.array(all_vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]