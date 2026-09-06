"""
parser.py

Extracts ONLY the Abstract and Conclusion/Discussion sections from academic
PDFs (built with dream-interpretation papers in mind, but generic enough for
most journal-formatted PDFs).

Strategy:
1. Pull raw text per page with pdfplumber.
2. Join into one big string.
3. Use heading-based regex to find where "Abstract" starts and where it ends
   (next heading like "Introduction", "Keywords", "1.", etc).
4. Do the same for "Conclusion" / "Conclusions" / "Discussion and Conclusion"
   through to the next heading ("References", "Acknowledgments", "Bibliography").
5. Return a dict with both chunks. If a section can't be found, it's left
   empty (and the caller should decide whether to skip the paper).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# Headings that commonly follow the Abstract in a paper
_ABSTRACT_END_HEADINGS = [
    "keywords", "key words", "index terms", "introduction",
    "1. introduction", "1 introduction", "background",
]

# Headings that can introduce the conclusion
_CONCLUSION_START_HEADINGS = [
    "conclusion", "conclusions", "discussion and conclusion",
    "conclusions and future work", "summary and conclusion",
    "general discussion", "concluding remarks",
]

# Headings that commonly follow the Conclusion
_CONCLUSION_END_HEADINGS = [
    "references", "acknowledgments", "acknowledgements", "bibliography",
    "appendix", "supplementary material", "conflict of interest",
    "funding", "author contributions",
]


@dataclass
class PaperSections:
    source_path: str
    title: str
    abstract: str
    conclusion: str

    @property
    def has_content(self) -> bool:
        return bool(self.abstract.strip() or self.conclusion.strip())

    @property
    def combined_text(self) -> str:
        """What actually gets embedded — abstract + conclusion, labeled."""
        parts = []
        if self.abstract.strip():
            parts.append(f"Abstract: {self.abstract.strip()}")
        if self.conclusion.strip():
            parts.append(f"Conclusion: {self.conclusion.strip()}")
        return "\n\n".join(parts)


def _extract_full_text(pdf_path: str) -> str:
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    # Normalize whitespace but keep paragraph breaks
    full_text = "\n".join(text_chunks)
    full_text = re.sub(r"[ \t]+", " ", full_text)
    return full_text


def _find_section(
    text: str,
    start_headings: list[str],
    end_headings: list[str],
    search_from: int = 0,
) -> tuple[str, int, int]:
    """
    Find the first occurrence of any start_heading after search_from, then
    capture everything up to the first end_heading (or a hard length cap).
    Returns (section_text, start_index, end_index). Empty string if not found.
    """
    lower = text.lower()

    start_pos = -1
    matched_heading_len = 0
    for heading in start_headings:
        # heading must appear at a line start-ish position (preceded by
        # newline/whitespace, optionally with section numbering like "4.")
        # to avoid matching it mid-sentence
        pattern = re.compile(
            r"(?:^|\n)\s*\d*\.?\s*" + re.escape(heading) + r"\s*[:\-\n]",
            re.IGNORECASE,
        )
        match = pattern.search(lower, search_from)
        if match and (start_pos == -1 or match.start() < start_pos):
            start_pos = match.start()
            matched_heading_len = match.end() - match.start()

    if start_pos == -1:
        return "", -1, -1

    content_start = start_pos + matched_heading_len

    end_pos = len(text)
    for heading in end_headings:
        pattern = re.compile(
            r"(?:^|\n)\s*\d*\.?\s*" + re.escape(heading) + r"\s*[:\-\n]?",
            re.IGNORECASE,
        )
        match = pattern.search(lower, content_start)
        if match and match.start() < end_pos:
            end_pos = match.start()

    # Safety cap: an abstract/conclusion shouldn't realistically exceed
    # ~6000 characters. If no end heading was found and the remaining text
    # is huge, we cap it rather than swallowing the whole paper.
    if end_pos - content_start > 6000:
        end_pos = content_start + 6000

    section_text = text[content_start:end_pos].strip()
    return section_text, start_pos, end_pos


def _guess_title(text: str, pdf_path: str) -> str:
    # crude heuristic: first non-empty line of reasonable length
    for line in text.splitlines():
        clean = line.strip()
        if 15 < len(clean) < 200 and not clean.lower().startswith("abstract"):
            return clean
    return Path(pdf_path).stem


def parse_paper(pdf_path: str) -> PaperSections:
    """Parse a single PDF and return only its Abstract + Conclusion."""
    full_text = _extract_full_text(pdf_path)
    title = _guess_title(full_text, pdf_path)

    abstract, _, abstract_end = _find_section(
        full_text, ["abstract"], _ABSTRACT_END_HEADINGS
    )

    # Search for the conclusion starting after the abstract to avoid
    # accidentally matching something inside it.
    search_from = max(abstract_end, 0)
    conclusion, _, _ = _find_section(
        full_text, _CONCLUSION_START_HEADINGS, _CONCLUSION_END_HEADINGS,
        search_from=search_from,
    )

    return PaperSections(
        source_path=pdf_path,
        title=title,
        abstract=abstract,
        conclusion=conclusion,
    )


def parse_directory(directory: str) -> list[PaperSections]:
    """Parse every PDF in a directory (non-recursive)."""
    results = []
    for path in sorted(Path(directory).glob("*.pdf")):
        parsed = parse_paper(str(path))
        results.append(parsed)
    return results