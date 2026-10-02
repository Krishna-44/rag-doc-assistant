"""Text normalisation shared by the lexical and dense retrievers.

Both retrievers must see exactly the same tokens for a query and a document, so tokenising,
stop-word removal and stemming live in one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")

STOPWORDS = frozenset(
    """
    a about above after again all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had
    has have having he her here hers him his how i if in into is it its itself just me more most
    my no nor not now of off on once only or other our ours out over own same she should so some
    such than that the their theirs them then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you your
    yours i'm i've i'd can't don't isn't also any anything each every get gets got many much
    may might must shall please tell know us like want
    """.split()
)


def stem(token: str) -> str:
    """A deliberately small suffix stripper ("deploys" -> "deploy", "policies" -> "policy").

    A full Porter stemmer is overkill for short policy documents; this keeps behaviour easy to
    reason about while still matching the most common inflections.
    """
    if token.endswith("'s") or token.endswith("’s"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed") and not token.endswith("eed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lower-case, split on non-alphanumerics, drop stop words and stem."""
    tokens = _TOKEN_RE.findall(text.lower())
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) for t in tokens]


@dataclass(frozen=True)
class Span:
    """A half-open character range ``[start, end)`` into some source text."""

    start: int
    end: int


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")


def heading_text(line: str) -> str | None:
    """Return the heading text if ``line`` is a Markdown ATX heading, else ``None``."""
    match = _HEADING_RE.match(line)
    return match.group(1) if match else None


def sentence_spans(text: str) -> list[Span]:
    """Split ``text`` into sentence spans, treating line-level Markdown structure as boundaries.

    Headings and list items are their own units; wrapped lines inside a paragraph are joined.
    Spans always index into the original string, so callers can slice the exact source text.
    """
    spans: list[Span] = []
    block_start: int | None = None
    block_end = 0
    offset = 0

    def flush() -> None:
        nonlocal block_start
        if block_start is None:
            return
        block = text[block_start:block_end]
        cursor = 0
        for match in _SENTENCE_END_RE.finditer(block):
            piece_end = match.start()
            if block[cursor:piece_end].strip():
                spans.append(_trimmed(text, block_start + cursor, block_start + piece_end))
            cursor = match.end()
        if block[cursor:].strip():
            spans.append(_trimmed(text, block_start + cursor, block_end))
        block_start = None

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        line_start, line_end = offset, offset + len(line.rstrip("\r\n"))
        offset += len(line)
        is_structural = heading_text(line) is not None or stripped.startswith(("- ", "* ", "+ "))
        if not stripped:
            flush()
        elif is_structural:
            flush()
            block_start, block_end = line_start, line_end
            flush()
        else:
            if block_start is None:
                block_start = line_start
            block_end = line_end
    flush()
    return spans


def _trimmed(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start, end)
