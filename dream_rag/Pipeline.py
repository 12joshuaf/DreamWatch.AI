"""
pipeline.py

Ties everything together:
  PDFs -> parser (abstract+conclusion only) -> OpenAI embeddings
       -> PCA vector store -> retrieval -> OpenAI chat completion

Usage:
    from dream_rag.pipeline import DreamRAGPipeline

    pipeline = DreamRAGPipeline(openai_api_key="sk-...")
    pipeline.ingest_directory("./papers")
    answer = pipeline.query("What do recurring falling dreams tend to symbolize?")
    print(answer)
"""

from __future__ import annotations

from openai import OpenAI

from .Embeddings import EmbeddingClient
from .Parser import parse_directory, parse_paper
from .Vector_Store import DocRecord, PCAVectorStore

DEFAULT_CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a research assistant specializing in dream interpretation "
    "literature. Answer the user's question using ONLY the excerpts "
    "provided below, which are the abstracts and conclusions of relevant "
    "papers. If the excerpts don't contain enough information to answer, "
    "say so plainly instead of guessing. Cite papers by title when you use "
    "them."
)


class DreamRAGPipeline:
    def __init__(
        self,
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = DEFAULT_CHAT_MODEL,
        pca_components: int = 50,
    ):
        self.embedder = EmbeddingClient(api_key=openai_api_key, model=embedding_model)
        self.chat_client = OpenAI(api_key=openai_api_key)
        self.chat_model = chat_model
        self.store = PCAVectorStore(n_components=pca_components)
        self._next_id = 0

    # --- ingestion ---------------------------------------------------

    def ingest_directory(self, directory: str, skip_empty: bool = True) -> int:
        """Parse every PDF in a directory, embed the abstract+conclusion text,
        and (re)build the PCA index. Returns the number of papers ingested."""
        parsed_papers = parse_directory(directory)

        records = []
        texts_to_embed = []
        for paper in parsed_papers:
            if skip_empty and not paper.has_content:
                print(f"[ingest] Skipping (no abstract/conclusion found): {paper.source_path}")
                continue
            records.append(
                DocRecord(
                    doc_id=str(self._next_id),
                    title=paper.title,
                    source_path=paper.source_path,
                    text=paper.combined_text,
                )
            )
            texts_to_embed.append(paper.combined_text)
            self._next_id += 1

        if not records:
            print("[ingest] No usable papers found.")
            return 0

        embeddings = self.embedder.embed(texts_to_embed)

        if self.store.raw_embeddings is None:
            self.store.build(embeddings, records)
        else:
            self.store.add_and_rebuild(embeddings, records)

        return len(records)

    def ingest_file(self, pdf_path: str, skip_empty: bool = True) -> bool:
        """Parse and embed a single PDF, adding it to the index."""
        paper = parse_paper(pdf_path)
        if skip_empty and not paper.has_content:
            print(f"[ingest] Skipping (no abstract/conclusion found): {pdf_path}")
            return False

        record = DocRecord(
            doc_id=str(self._next_id),
            title=paper.title,
            source_path=paper.source_path,
            text=paper.combined_text,
        )
        self._next_id += 1

        embedding = self.embedder.embed([paper.combined_text])
        if self.store.raw_embeddings is None:
            self.store.build(embedding, [record])
        else:
            self.store.add_and_rebuild(embedding, [record])
        return True

    # --- retrieval + generation ---------------------------------------

    def retrieve(self, question: str, top_k: int = 5) -> list[tuple[DocRecord, float]]:
        query_embedding = self.embedder.embed_one(question)
        return self.store.search(query_embedding, top_k=top_k)

    def query(self, question: str, top_k: int = 5) -> str:
        results = self.retrieve(question, top_k=top_k)

        if not results:
            return "No papers have been ingested yet, so I have nothing to search."

        context_blocks = []
        for record, score in results:
            context_blocks.append(
                f"### {record.title} (similarity: {score:.2f})\n{record.text}"
            )
        context = "\n\n".join(context_blocks)

        user_message = (
            f"Question: {question}\n\n"
            f"Relevant excerpts:\n\n{context}"
        )

        response = self.chat_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    # --- persistence ---------------------------------------------------

    def save_index(self, directory: str) -> None:
        self.store.save(directory)

    def load_index(self, directory: str) -> None:
        self.store = PCAVectorStore.load(directory)
        self._next_id = len(self.store.records)