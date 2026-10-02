"""Lexical (BM25), dense (LSA embeddings) and hybrid (reciprocal rank fusion) retrieval.

* **BM25** is implemented from scratch: it is strong on exact terms such as numbers and names.
* **Dense** retrieval embeds chunks with TF-IDF + truncated SVD (latent semantic analysis). It runs
  fully offline with no model download and captures co-occurrence ("vacation" ~ "annual leave").
  Any object with ``fit``/``embed`` can be swapped in, e.g. a sentence-transformers model.
* **Hybrid** retrieval fuses both rankings with reciprocal rank fusion, which needs no score
  calibration between the two retrievers.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ragdoc.chunking import Chunk
from ragdoc.text import tokenize

Mode = Literal["bm25", "dense", "hybrid"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid")


class BM25:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._doc_len = np.zeros(0)
        self._avgdl = 0.0

    def fit(self, corpus: Sequence[Sequence[str]]) -> BM25:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, tokens in enumerate(corpus):
            for term, tf in Counter(tokens).items():
                postings[term].append((doc_idx, tf))
        n = len(corpus)
        self._postings = dict(postings)
        # Lucene's non-negative IDF variant.
        self._idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()
        }
        self._doc_len = np.array([len(tokens) for tokens in corpus], dtype=float)
        self._avgdl = float(self._doc_len.mean()) if n else 0.0
        return self

    def __contains__(self, term: str) -> bool:
        return term in self._idf

    def idf(self, term: str) -> float:
        """IDF of ``term``; unseen terms get the maximum (they are maximally specific)."""
        known = self._idf.get(term)
        if known is not None:
            return known
        n = len(self._doc_len)
        return math.log(1 + (n + 0.5) / 0.5)

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(len(self._doc_len))
        if not len(self._doc_len):
            return scores
        norm = self.k1 * (1 - self.b + self.b * self._doc_len / max(self._avgdl, 1e-9))
        for term in set(query_tokens):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_idx, tf in self._postings[term]:
                scores[doc_idx] += idf * tf * (self.k1 + 1) / (tf + norm[doc_idx])
        return scores


class Embedder(Protocol):
    def fit(self, texts: Sequence[str]) -> Embedder: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class LsaEmbedder:
    """TF-IDF (unigrams + bigrams) projected to a low-rank semantic space with truncated SVD."""

    def __init__(self, dims: int = 128, seed: int = 0) -> None:
        self.dims = dims
        self.seed = seed
        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
            sublinear_tf=True, min_df=1,
        )
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: Sequence[str]) -> LsaEmbedder:
        tfidf = self._vectorizer.fit_transform(texts)
        rank = min(self.dims, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=rank, random_state=self.seed) if rank >= 2 else None
        if self._svd is not None:
            self._svd.fit(tfidf)
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        tfidf = self._vectorizer.transform(texts)
        vectors = self._svd.transform(tfidf) if self._svd is not None else tfidf.toarray()
        return normalize(vectors)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Fuse several best-first rankings: ``score(d) = sum_r 1 / (k + rank_r(d))``."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] += 1.0 / (k + rank)
    return dict(fused)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    rank: int
    score: float
    bm25: float
    dense: float


class Retriever:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder | None = None) -> None:
        if not chunks:
            raise ValueError("Cannot build a retriever over zero chunks")
        self.chunks = list(chunks)
        texts = [c.search_text for c in self.chunks]
        self.bm25 = BM25().fit([tokenize(t) for t in texts])
        self.embedder = (embedder or LsaEmbedder()).fit(texts)
        self._vectors = self.embedder.embed(texts)

    def _scores(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        bm25 = self.bm25.scores(tokenize(query))
        dense = self._vectors @ self.embedder.embed([query])[0]
        return bm25, dense

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[SearchHit]:
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {MODES}")
        k = max(1, min(k, len(self.chunks)))
        bm25, dense = self._scores(query)
        if mode == "bm25":
            final = bm25
        elif mode == "dense":
            final = dense
        else:
            # Only fuse the head of each ranking; the tail of a BM25 ranking with zero scores is
            # arbitrary and would add noise.
            depth = min(len(self.chunks), 20)
            bm25_rank = [i for i in np.argsort(-bm25)[:depth] if bm25[i] > 0]
            dense_rank = list(np.argsort(-dense)[:depth])
            fused = reciprocal_rank_fusion([bm25_rank, dense_rank])
            final = np.array([fused.get(i, 0.0) for i in range(len(self.chunks))])
        order = np.argsort(-final, kind="stable")[:k]
        return [
            SearchHit(
                chunk=self.chunks[i], rank=r, score=float(final[i]),
                bm25=float(bm25[i]), dense=float(dense[i]),
            )
            for r, i in enumerate(order, start=1)
        ]
