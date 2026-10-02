from itertools import pairwise

import pytest

from ragdoc.chunking import chunk_document
from ragdoc.documents import document_from_text

DOC = document_from_text(
    "policy",
    "# Policy\n\nIntro sentence one. Intro sentence two.\n\n"
    "## Section A\n\n" + " ".join(f"Alpha sentence number {i}." for i in range(30)) + "\n\n"
    "## Section B\n\nBeta only sentence.\n",
)


def test_chunks_respect_size_and_map_back_to_source():
    chunks = chunk_document(DOC, max_chars=200, overlap=1)
    assert len(chunks) > 3
    for c in chunks:
        assert c.text == DOC.text[c.start : c.end]
        assert len(c.text) <= 200
        assert c.text.strip()


def test_chunks_track_sections_and_never_cross_headings():
    chunks = chunk_document(DOC, max_chars=200, overlap=1)
    assert chunks[0].section is None  # text under the H1 belongs to the document itself
    assert {c.section for c in chunks} == {None, "Section A", "Section B"}
    assert all("Beta" not in c.text for c in chunks if c.section == "Section A")
    assert chunks[-1].label == "Policy › Section B"


def test_consecutive_chunks_overlap_by_one_sentence():
    chunks = [c for c in chunk_document(DOC, max_chars=200, overlap=1) if c.section == "Section A"]
    for prev, nxt in pairwise(chunks):
        last_sentence = prev.text.rsplit(". ", 1)[-1]
        assert nxt.text.startswith(last_sentence.rstrip("."))


def test_overlong_sentence_is_hard_wrapped():
    doc = document_from_text("long", "word " * 400)
    chunks = chunk_document(doc, max_chars=120, overlap=0)
    assert all(len(c.text) <= 120 for c in chunks)
    assert sum(len(c.text.split()) for c in chunks) == 400


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_document(DOC, max_chars=10)
    with pytest.raises(ValueError):
        chunk_document(DOC, overlap=-1)
