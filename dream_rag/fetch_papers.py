"""
fetch_papers.py



Usage:
    python fetch_papers.py --count 100 --outdir ./papers
    python fetch_papers.py --count 50 --query "dream symbolism"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_ARTICLE_HTML = "https://europepmc.org/article/PMC/{pmcid_number}"

DEFAULT_QUERY = (
    '(TITLE:"dream interpretation" OR TITLE:"dream analysis" OR '
    'TITLE:"dream content" OR TITLE:"dream symbolism" OR ABSTRACT:"dream interpretation" '
    'OR ABSTRACT:"dream analysis" OR ABSTRACT:"dream symbolism") AND OPEN_ACCESS:Y'
)

HEADERS = {
    # Polite automated-tool identification.
    "User-Agent": "dream-rag-paper-fetcher/2.0 (research script)"
}

PDF_MAGIC = b"%PDF-"


def polite_get(url: str, params: dict | None = None, delay: float = 0.34, **kwargs) -> requests.Response:
    """GET with a small delay before it to stay polite to the API."""
    time.sleep(delay)
    response = requests.get(url, params=params, headers=HEADERS, timeout=30, **kwargs)
    response.raise_for_status()
    return response


def search_europe_pmc(query: str, count: int, delay: float) -> list[dict]:
    """Query Europe PMC and return result dicts (with fullTextUrlList) for
    articles that have a PMC ID."""
    results: list[dict] = []
    cursor_mark = "*"
    page_size = min(100, max(count * 2, 25))  # over-fetch since some won't have a PDF

    while len(results) < count * 2:
        params = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor_mark,
            "resultType": "core",  # needed to get fullTextUrlList
        }
        response = polite_get(EUROPE_PMC_SEARCH_URL, params=params, delay=delay)
        data = response.json()

        page_results = data.get("resultList", {}).get("result", [])
        if not page_results:
            break

        for r in page_results:
            if r.get("pmcid"):
                results.append(r)

        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor_mark:
            break
        cursor_mark = next_cursor

    return results


def find_pdf_url_from_metadata(result: dict) -> str | None:
    """Look for a documentStyle == 'pdf' entry in the article's fullTextUrlList."""
    url_list = result.get("fullTextUrlList", {}).get("fullTextUrl", [])
    for entry in url_list:
        if entry.get("documentStyle", "").lower() == "pdf":
            return entry.get("url")
    return None


def find_html_fulltext_url(result: dict) -> str | None:
    """Fallback: find an HTML full-text URL from the metadata to scrape for a PDF link."""
    url_list = result.get("fullTextUrlList", {}).get("fullTextUrl", [])
    for entry in url_list:
        if entry.get("documentStyle", "").lower() == "html":
            return entry.get("url")
    return None


def scrape_pdf_link_from_html(html: str, base_url: str) -> str | None:
    """Fallback PDF discovery: parse an HTML full-text page with BeautifulSoup
    and look for a genuine PDF download link (not just any '.pdf'-looking href,
    which is what produced false positives before)."""
    soup = BeautifulSoup(html, "html.parser")

    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        rel = " ".join(a.get("rel", [])).lower()
        # Look for hrefs that clearly point at a PDF, or link text/attrs that say so.
        if href.lower().endswith(".pdf") or "pdf" in rel or "download pdf" in text or "view pdf" in text:
            candidates.append(href)

    if not candidates:
        return None

    href = candidates[0]
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        # resolve against the scheme+host of base_url
        from urllib.parse import urlsplit
        parts = urlsplit(base_url)
        return f"{parts.scheme}://{parts.netloc}{href}"
    return base_url.rstrip("/") + "/" + href


def is_real_pdf(content: bytes) -> bool:
    """Strict validation: only trust the file's magic bytes, never the
    Content-Type header (servers sometimes mislabel HTML error pages as
    'application/pdf', which is what caused garbled/wrong-charset files)."""
    return content[:5] == PDF_MAGIC


def download_paper(result: dict, outdir: Path, delay: float) -> bool:
    """Fetch a single article's PDF using metadata first, HTML-scrape as
    fallback. Returns True on success."""
    pmcid = result["pmcid"]
    title = result.get("title", "")[:80]

    pdf_url = find_pdf_url_from_metadata(result)

    if not pdf_url:
        html_url = find_html_fulltext_url(result)
        if not html_url:
            print(f"  [skip] {pmcid}: no full-text URL of any kind in metadata")
            return False
        try:
            page = polite_get(html_url, delay=delay)
        except requests.RequestException as e:
            print(f"  [skip] {pmcid}: couldn't load HTML full text ({e})")
            return False
        pdf_url = scrape_pdf_link_from_html(page.text, html_url)
        if not pdf_url:
            print(f"  [skip] {pmcid}: no PDF link found even after scraping HTML full text")
            return False

    try:
        pdf_response = polite_get(pdf_url, delay=delay)
    except requests.RequestException as e:
        print(f"  [skip] {pmcid}: PDF download failed ({e})")
        return False

    if not is_real_pdf(pdf_response.content):
        print(f"  [skip] {pmcid}: downloaded file isn't actually a PDF "
              f"(got content-type: {pdf_response.headers.get('Content-Type', '?')}) — "
              f"'{title}'")
        return False

    out_path = outdir / f"{pmcid}.pdf"
    out_path.write_bytes(pdf_response.content)  # binary write — no charset involved
    print(f"  [ok]   {pmcid} -> {out_path.name}  ({title})")
    return True


def fetch_papers(query: str, count: int, outdir: str, delay: float = 0.34) -> int:
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching Europe PMC for: {query}")
    candidates = search_europe_pmc(query, count=count, delay=delay)
    print(f"Found {len(candidates)} candidate articles with a PMC ID. Downloading up to {count}...\n")

    downloaded = 0
    for result in candidates:
        if downloaded >= count:
            break

        pmcid = result["pmcid"]
        out_path = out_dir / f"{pmcid}.pdf"
        if out_path.exists():
            print(f"  [skip] {pmcid}: already downloaded")
            downloaded += 1
            continue

        if download_paper(result, out_dir, delay=delay):
            downloaded += 1

    print(f"\nDone. {downloaded}/{count} papers saved to {out_dir.resolve()}")
    if downloaded < count:
        print(
            "Fewer papers than requested were saved — some candidates had no "
            "resolvable open-access PDF. Try a broader --query or increase "
            "--count to compensate."
        )
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Fetch open-access dream-interpretation papers via Europe PMC.")
    parser.add_argument("--count", type=int, default=1000, help="Number of papers to fetch (default: 1000)")
    parser.add_argument("--outdir", type=str, default="./papers", help="Directory to save PDFs into (default: ./papers)")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Europe PMC search query")
    parser.add_argument("--delay", type=float, default=0.34, help="Delay in seconds between requests (default: 0.34)")
    args = parser.parse_args()

    try:
        fetch_papers(query=args.query, count=args.count, outdir=args.outdir, delay=args.delay)
    except requests.RequestException as e:
        print(f"Network error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()