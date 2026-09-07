# Dream-Interpretation RAG Pipeline

A small RAG pipeline purpose-built for papers on dream interpretation:

- **Parser** (`parser.py`) — opens each PDF and extracts *only* the Abstract
  and Conclusion sections (via heading detection), ignoring methods, results,
  references, etc. This keeps what gets embedded short and on-topic.
- **Embeddings** (`embeddings.py`) — calls OpenAI's embeddings API
  (`text-embedding-3-small` by default) on the extracted text.
- **Vector store** (`vector_store.py`) — instead of a hosted vector DB, fits
  PCA (a small numpy/SVD implementation, no sklearn/scipy dependency) on
  your corpus's embeddings and stores the dimensionality-reduced vectors in
  memory (with save/load to disk). Search is cosine similarity in the
  reduced space.
- **Pipeline** (`pipeline.py`) — wires it together: ingest a folder of PDFs,
  then ask questions; retrieved excerpts are passed to an OpenAI chat model
  to generate the final answer.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

## Usage

```python
from dream_rag import DreamRAGPipeline

pipeline = DreamRAGPipeline(pca_components=50)

# Point this at a folder of your dream-interpretation PDFs
pipeline.ingest_directory("./papers")

answer = pipeline.query("What do recurring falling dreams tend to symbolize?")
print(answer)

# Optional: persist the index so you don't have to re-embed next time
pipeline.save_index("./dream_rag_index")

# Later, in a new session:
pipeline2 = DreamRAGPipeline()
pipeline2.load_index("./dream_rag_index")
print(pipeline2.query("Are nightmares linked to trauma recovery?"))
```

Or from the command line:

```bash
python example_usage.py ./papers "What do teeth-falling-out dreams usually mean?"
```

## Fetching papers

`fetch_papers.py` pulls open-access papers on dream interpretation from
PubMed Central (via NCBI's E-utilities API + BeautifulSoup to locate each
article's PDF link) and saves them straight into `./papers`, ready for
`ingest_directory`. It only pulls from PMC's "open access" filter — genuinely
free-to-download peer-reviewed literature — rather than scraping paywalled
publisher sites, which would violate their terms of service.

```bash
python fetch_papers.py --count 100 --outdir ./papers
```

Options:
- `--query` — override the default search terms
- `--api-key` — optional NCBI API key (raises rate limit from 3 to 10 req/sec; get one free at ncbi.nlm.nih.gov/account/settings)
- `--email` — optional contact email, recommended by NCBI for API usage
- `--delay` — seconds between requests (default ~3/sec, NCBI's unauthenticated limit)

Not every PMC article yields a parsable PDF link (page templates vary), so
the script over-fetches candidate IDs and skips ones it can't resolve —
check the `[skip]` lines in the output if you end up with fewer than
requested.



- **`pca_components`**: defaults to 50. With very few papers (fewer than
  `pca_components`), the store automatically shrinks the number of
  components to fit — you'll see a printed warning. PCA needs a reasonably
  sized corpus (dozens of papers+) before the reduced space is meaningful;
  with only a handful of papers, consider skipping PCA and comparing raw
  embeddings directly (swap `PCAVectorStore` for a simpler cosine-similarity
  list — the interface is small on purpose).
- **Section detection**: `parser.py` uses heading regexes tuned for common
  journal formats ("Abstract", "1. Introduction", "Conclusion",
  "References", etc.). Papers with unusual formatting (scanned images
  without OCR, non-standard headings) may need `_ABSTRACT_END_HEADINGS` /
  `_CONCLUSION_START_HEADINGS` / `_CONCLUSION_END_HEADINGS` extended — add
  any headings you see failing to parse correctly.
- **Skipped papers**: if a paper's abstract/conclusion can't be found,
  `ingest_directory` logs and skips it rather than embedding the whole
  paper (which would defeat the point of the abstract/conclusion-only
  filter). Check the printed `[ingest] Skipping...` lines after a run.
- **Cost**: embeddings are cheap (`text-embedding-3-small`), and only
  abstract+conclusion text gets sent, so ingesting even hundreds of papers
  should cost well under a dollar. The chat completion step is the
  per-query cost.
- **Swapping the vector store later**: if your corpus grows large enough
  that PCA+in-memory search stops being practical, `PCAVectorStore` can be
  swapped for pgvector/Pinecone/Chroma by keeping the same
  `build()` / `search()` / `save()` / `load()` method signatures.