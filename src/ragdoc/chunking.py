"""Sentence-aware chunking with overlap and exact source offsets.

Chunks never cut a sentence in half (unless a single sentence is longer than the limit), consecutive
chunks share ``overlap`` sentences so an answer that straddles a boundary is still retrievable, and
every chunk records the Markdown section it came from.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragdoc.documents import Document
from ragdoc.text import Span, heading_text, sentence_spans


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str | None
    text: str
    start: int
    end: int

    @property
    def search_text(self) -> str:
        """Text used for indexing: the chunk plus its document title and section heading."""
        parts = [self.title, self.section or "", self.text]
        return "\n".join(p for p in parts if p)

    @property
    def label(self) -> str:
        return f"{self.title} › {self.section}" if self.section else self.title


def _split_long(span: Span, text: str, max_chars: int) -> list[Span]:
    """Hard-wrap a single over-long sentence at whitespace."""
    pieces: list[Span] = []
    start = span.start
    while span.end - start > max_chars:
        cut = text.rfind(" ", start, start + max_chars)
        if cut <= start:
            cut = start + max_chars
        pieces.append(Span(start, cut))
        start = cut
        while start < span.end and text[start].isspace():
            start += 1
    if start < span.end:
        pieces.append(Span(start, span.end))
    return pieces


def chunk_document(doc: Document, *, max_chars: int = 700, overlap: int = 1) -> list[Chunk]:
    if max_chars < 50:
        raise ValueError("max_chars must be at least 50")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    units: list[tuple[Span, str | None]] = []  # (sentence span, section it belongs to)
    section: str | None = None
    for span in sentence_spans(doc.text):
        piece = doc.text[span.start : span.end]
        heading = heading_text(piece)
        if heading is not None:
            # A top-level "# Title" is the document title, not a section.
            section = None if piece.lstrip().startswith("# ") else heading
            continue
        for sub in _split_long(span, doc.text, max_chars):
            units.append((sub, section))

    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        j = i
        # Greedily pack sentences from the same section until the size limit.
        while j < len(units) and units[j][1] == units[i][1]:
            if j > i and units[j][0].end - units[i][0].start > max_chars:
                break
            j += 1
        start, end = units[i][0].start, units[j - 1][0].end
        chunks.append(
            Chunk(
                id=f"{doc.id}#{len(chunks)}",
                doc_id=doc.id,
                title=doc.title,
                section=units[i][1],
                text=doc.text[start:end],
                start=start,
                end=end,
            )
        )
        if j >= len(units) or units[j][1] != units[i][1]:
            i = j  # new section: don't overlap across headings
        else:
            i = max(j - overlap, i + 1)
    return chunks


def chunk_documents(docs: list[Document], *, max_chars: int = 700, overlap: int = 1) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, max_chars=max_chars, overlap=overlap)]
