"""
example_usage.py

Drop your dream-interpretation PDFs into a folder (e.g. ./papers) and run:

    export OPENAI_API_KEY=sk-...
    python example_usage.py ./papers "What do teeth-falling-out dreams usually mean?"
"""

import sys

from dream_rag import DreamRAGPipeline


def main():
    if len(sys.argv) < 3:
        print("Usage: python example_usage.py <papers_directory> <question>")
        sys.exit(1)

    papers_dir = sys.argv[1]
    question = " ".join(sys.argv[2:])

    # Uses OPENAI_API_KEY from the environment by default.
    pipeline = DreamRAGPipeline(pca_components=50)

    n = pipeline.ingest_directory(papers_dir)
    print(f"Ingested {n} paper(s).\n")

    if n == 0:
        return

    # Optional: persist the index so you don't have to re-embed every run
    pipeline.save_index("./dream_rag_index")

    answer = pipeline.query(question, top_k=5)
    print("=== Answer ===")
    print(answer)



if __name__ == "__main__":
    main()