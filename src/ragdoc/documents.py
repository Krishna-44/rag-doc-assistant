"""Loading source documents (Markdown, plain text and PDF) from disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragdoc.text import heading_text

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str


def _title_from(text: str, fallback: str) -> str:
    for line in text.splitlines():
        title = heading_text(line)
        if title and line.lstrip().startswith("# "):
            return title
        if line.strip():
            break
    return fallback.replace("-", " ").replace("_", " ").strip().title()


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise RuntimeError("PDF support needs the optional extra: pip install 'ragdoc[pdf]'") from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)


def document_from_text(doc_id: str, text: str, source: str = "") -> Document:
    text = text.replace("\r\n", "\n")
    return Document(id=doc_id, title=_title_from(text, doc_id), text=text, source=source or doc_id)


def load_document(path: str | Path) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type {suffix!r}; expected one of {SUPPORTED_SUFFIXES}")
    text = _read_pdf(path) if suffix == ".pdf" else path.read_text(encoding="utf-8")
    return document_from_text(path.stem, text, source=str(path))


def load_directory(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Document directory not found: {directory}")
    paths = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return [load_document(p) for p in paths]
