"""
vector_store.py

A minimal "vector DB" built on top of scikit-learn's PCA instead of a
dedicated vector database service. The idea:

1. Collect raw OpenAI embeddings for every document (1536-dim for
   text-embedding-3-small).
2. Fit a PCA model on the full corpus to compress those vectors down to a
   much smaller number of components (e.g. 50) while retaining most of the
   variance. This shrinks memory/storage and speeds up similarity search,
   at the cost of a small amount of retrieval accuracy.
3. Store the reduced vectors + metadata in memory (and optionally persist
   to disk as .npz/.json).
4. At query time, project the query embedding through the same PCA
   transform and do cosine similarity search in the reduced space.

This is appropriate for small-to-medium corpora (dozens to a few thousand
papers). For very large corpora, swap this out for a real vector DB
(pgvector, Pinecone, Chroma, etc.) — the interface below is intentionally
small so that swap is easy later.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA



@dataclass
class DocRecord:
    doc_id: str
    title: str
    source_path: str
    text: str  # the abstract+conclusion text that was embedded
    metadata: dict = field(default_factory=dict)


class PCAVectorStore:
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca: PCA | None = None
        self.raw_embeddings: np.ndarray | None = None      # (n_docs, orig_dim)
        self.reduced_embeddings: np.ndarray | None = None  # (n_docs, n_components)
        self.records: list[DocRecord] = []

    def build(self, embeddings: np.ndarray, records: list[DocRecord]) -> None:
        """Fit PCA on the full corpus and store everything. Call once per corpus,
        or use `add_and_rebuild` to incrementally add docs then refit."""
        if len(records) != embeddings.shape[0]:
            raise ValueError("embeddings and records must have the same length")

        n_components = min(self.n_components, embeddings.shape[0], embeddings.shape[1])
        if n_components < self.n_components:
            print(
                f"[PCAVectorStore] Reducing n_components from {self.n_components} "
                f"to {n_components} (not enough documents/dims yet)."
            )

        self.pca = PCA(n_components=n_components)
        self.raw_embeddings = embeddings
        self.reduced_embeddings = self.pca.fit_transform(embeddings)
        self.records = records

        explained = self.pca.explained_variance_ratio_.sum()
        print(f"[PCAVectorStore] Fit PCA with {n_components} components, "
              f"explaining {explained:.1%} of variance.")

    def add_and_rebuild(self, new_embeddings: np.ndarray, new_records: list[DocRecord]) -> None:
        """Add more documents and refit PCA on the combined corpus.
        Simple approach, fine for small/medium corpora that don't change often."""
        if self.raw_embeddings is None:
            self.build(new_embeddings, new_records)
            return
        combined_embeddings = np.vstack([self.raw_embeddings, new_embeddings])
        combined_records = self.records + new_records
        self.build(combined_embeddings, combined_records)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[DocRecord, float]]:
        """Project the query into PCA space and return top_k docs by cosine similarity."""
        if self.pca is None or self.reduced_embeddings is None:
            raise RuntimeError("Vector store has not been built yet. Call .build() first.")

        query_reduced = self.pca.transform(query_embedding.reshape(1, -1))[0]

        # cosine similarity in reduced space
        norm_query = query_reduced / (np.linalg.norm(query_reduced) + 1e-10)
        norms_docs = self.reduced_embeddings / (
            np.linalg.norm(self.reduced_embeddings, axis=1, keepdims=True) + 1e-10
        )
        similarities = norms_docs @ norm_query

        top_indices = np.argsort(-similarities)[:top_k]
        return [(self.records[i], float(similarities[i])) for i in top_indices]

    # --- persistence -----------------------------------------------------

    def save(self, directory: str) -> None:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            out_dir / "vectors.npz",
            raw_embeddings=self.raw_embeddings,
            reduced_embeddings=self.reduced_embeddings,
        )
        # Pickle the fitted PCA object directly rather than hand-picking
        # attributes — sklearn's PCA needs several internal fields
        # (explained_variance_, etc.) for transform() to work correctly.
        with open(out_dir / "pca.pkl", "wb") as f:
            pickle.dump(self.pca, f)

        meta = [
            {
                "doc_id": r.doc_id,
                "title": r.title,
                "source_path": r.source_path,
                "text": r.text,
                "metadata": r.metadata,
            }
            for r in self.records
        ]
        with open(out_dir / "records.json", "w") as f:
            json.dump({"n_components": self.n_components, "records": meta}, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "PCAVectorStore":
        in_dir = Path(directory)
        data = np.load(in_dir / "vectors.npz")
        with open(in_dir / "records.json") as f:
            meta = json.load(f)
        with open(in_dir / "pca.pkl", "rb") as f:
            pca = pickle.load(f)

        store = cls(n_components=meta["n_components"])
        store.raw_embeddings = data["raw_embeddings"]
        store.reduced_embeddings = data["reduced_embeddings"]
        store.pca = pca

        store.records = [
            DocRecord(
                doc_id=r["doc_id"],
                title=r["title"],
                source_path=r["source_path"],
                text=r["text"],
                metadata=r["metadata"],
            )
            for r in meta["records"]
        ]
        return store