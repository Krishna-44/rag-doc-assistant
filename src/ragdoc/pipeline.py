"""The end-to-end question-answering pipeline: retrieve → (maybe abstain) → generate."""

from __future__ import annotations

import os
from time import perf_counter

from ragdoc.generation import ClaudeGenerator, ExtractiveGenerator, Generator, not_found
from ragdoc.knowledge_base import KnowledgeBase
from ragdoc.retrieval import Mode, SearchHit
from ragdoc.text import tokenize


def _weighted_share(terms: set[str], idf, keep) -> float:
    total = sum(idf(t) for t in terms)
    return sum(idf(t) for t in terms if keep(t)) / total if total else 0.0


def lexical_coverage(question: str, hit: SearchHit, idf) -> float:
    """IDF-weighted share of the question's terms that appear in a chunk.

    Weighting by IDF means a match on a rare, specific term ("parental") counts for much more than
    a match on a common one ("company"), so a question about something the corpus never mentions
    scores close to zero even if it shares a few generic words with it.
    """
    chunk_terms = set(tokenize(hit.chunk.search_text))
    return _weighted_share(set(tokenize(question)), idf, chunk_terms.__contains__)


class RAGPipeline:
    def __init__(
        self,
        kb: KnowledgeBase,
        generator: Generator,
        *,
        k: int = 4,
        mode: Mode = "hybrid",
        min_coverage: float = 0.3,
        min_similarity: float = 0.2,
    ) -> None:
        self.kb = kb
        self.generator = generator
        self.k = k
        self.mode = mode
        self.min_coverage = min_coverage
        self.min_similarity = min_similarity

    def should_abstain(self, question: str, hits: list[SearchHit]) -> bool:
        """Skip generation when neither retriever found anything that plausibly answers.

        The dense score is discounted by the share of the question the corpus vocabulary covers:
        LSA silently drops unknown words, so "Explain quantum chromodynamics" would otherwise look
        like a strong match for any chunk containing "explain". Thresholds were chosen on the
        bundled sample set; re-check them on your own corpus.
        """
        if not hits:
            return True
        idf = self.kb.idf
        lexical = max(lexical_coverage(question, h, idf) for h in hits)
        known = _weighted_share(set(tokenize(question)), idf, self.kb.in_vocabulary)
        semantic = max(h.dense for h in hits) * known
        return lexical < self.min_coverage and semantic < self.min_similarity

    def ask(self, question: str, *, k: int | None = None, mode: Mode | None = None):
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        start = perf_counter()
        hits = self.kb.search(question, k=k or self.k, mode=mode or self.mode)
        if self.should_abstain(question, hits):
            answer = not_found(question, hits, self.generator.name)
        else:
            answer = self.generator.generate(question, hits)
        answer.latency_ms = (perf_counter() - start) * 1000
        return answer


def default_generator(kind: str = "auto", *, model: str | None = None) -> Generator:
    """``auto`` uses Claude when Anthropic credentials are configured, else the offline generator."""
    kind = kind.lower()
    has_credentials = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if kind == "claude" or (kind == "auto" and has_credentials):
        return ClaudeGenerator(model=model or os.environ.get("RAGDOC_MODEL", "claude-opus-5-5"))
    if kind in ("auto", "extractive"):
        return ExtractiveGenerator()
    raise ValueError(f"Unknown generator {kind!r}; expected auto, claude or extractive")
