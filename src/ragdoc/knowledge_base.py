"""A small in-memory knowledge base: documents → chunks → retriever.

The index is rebuilt from source documents on start-up (well under a second for hundreds of
pages). For large corpora, swap :class:`~ragdoc.retrieval.Retriever` for a vector database; the rest
of the pipeline only depends on ``search``.
"""

from __future__ import annotations

from pathlib import Path

from ragdoc.chunking import Chunk, chunk_documents
from ragdoc.documents import Document, load_directory
from ragdoc.retrieval import Embedder, Mode, Retriever, SearchHit


class KnowledgeBase:
    def __init__(
        self,
        documents: list[Document],
        *,
        chunk_chars: int = 700,
        overlap: int = 1,
        embedder: Embedder | None = None,
    ) -> None:
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self._embedder = embedder
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self._retriever: Retriever | None = None
        self.add_documents(documents)

    @classmethod
    def from_directory(cls, directory: str | Path, **kwargs) -> KnowledgeBase:
        return cls(load_directory(directory), **kwargs)

    def add_documents(self, documents: list[Document]) -> None:
        """Add or replace documents (matched by id) and rebuild the index."""
        by_id = {d.id: d for d in self.documents}
        for doc in documents:
            by_id[doc.id] = doc
        self.documents = sorted(by_id.values(), key=lambda d: d.id)
        self.chunks = chunk_documents(self.documents, max_chars=self.chunk_chars, overlap=self.overlap)
        self._retriever = Retriever(self.chunks, self._embedder) if self.chunks else None

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[SearchHit]:
        if self._retriever is None:
            return []
        return self._retriever.search(query, k=k, mode=mode)

    def idf(self, term: str) -> float:
        return self._retriever.bm25.idf(term) if self._retriever else 0.0

    def in_vocabulary(self, term: str) -> bool:
        return self._retriever is not None and term in self._retriever.bm25

    def stats(self) -> dict[str, int]:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}
