"""Answer generation from retrieved chunks.

Two interchangeable generators share one interface:

* :class:`ClaudeGenerator` passes each retrieved chunk to Claude as a ``document`` block with
  native citations enabled, so every claim in the answer is tied to the exact text it came from.
* :class:`ExtractiveGenerator` needs no API key: it selects the sentences that best match the
  question and cites them. It keeps the project fully runnable offline and in CI.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ragdoc.retrieval import SearchHit
from ragdoc.text import heading_text, sentence_spans, tokenize

DEFAULT_MODEL = "claude-opus-5-5"
NOT_FOUND = "I couldn't find that in the knowledge base."

SYSTEM_PROMPT = (
    "You answer questions about an organisation's internal documents. Use only the documents "
    "provided in the user's message. If they don't contain the answer, say that you couldn't find "
    "it in the knowledge base instead of guessing. Answer in two to four sentences and quote "
    "figures such as amounts, limits and deadlines exactly as the documents state them."
)


@dataclass(frozen=True)
class Citation:
    source: int  # 1-based position in Answer.sources, matching the [n] markers in the text
    chunk_id: str
    label: str
    quote: str


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    sources: list[SearchHit]
    abstained: bool
    generator: str
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "abstained": self.abstained,
            "generator": self.generator,
            "latency_ms": round(self.latency_ms, 1),
            "citations": [asdict(c) for c in self.citations],
            "sources": [
                {
                    "n": i,
                    "chunk_id": h.chunk.id,
                    "doc_id": h.chunk.doc_id,
                    "label": h.chunk.label,
                    "text": h.chunk.text,
                    "score": round(h.score, 4),
                    "bm25": round(h.bm25, 3),
                    "dense": round(h.dense, 3),
                }
                for i, h in enumerate(self.sources, start=1)
            ],
            **({"meta": self.meta} if self.meta else {}),
        }


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: Sequence[SearchHit]) -> Answer: ...


def not_found(question: str, hits: Sequence[SearchHit], generator: str) -> Answer:
    return Answer(question, NOT_FOUND, [], list(hits), abstained=True, generator=generator)


def _clean(sentence: str) -> str:
    sentence = re.sub(r"\*\*|__", "", sentence)
    sentence = re.sub(r"^\s*[-*+]\s+", "", sentence)
    return " ".join(sentence.split())


class ExtractiveGenerator:
    """Pick the sentences that best cover the question's terms and cite where they came from."""

    name = "extractive"

    def __init__(self, max_sentences: int = 2, max_sources: int = 3) -> None:
        self.max_sentences = max_sentences
        self.max_sources = max_sources

    def generate(self, question: str, hits: Sequence[SearchHit]) -> Answer:
        q_terms = set(tokenize(question))
        candidates: list[tuple[float, int, int, str]] = []
        for n, hit in enumerate(hits[: self.max_sources], start=1):
            text = hit.chunk.text
            for pos, span in enumerate(sentence_spans(text)):
                sentence = text[span.start : span.end]
                terms = set(tokenize(sentence))
                if heading_text(sentence) is not None or not terms:
                    continue
                overlap = len(q_terms & terms)
                if overlap:
                    # Normalise by sentence length and slightly prefer higher-ranked sources.
                    score = overlap / math.sqrt(len(terms)) + 0.3 / n
                    candidates.append((score, n, pos, sentence))
        if not candidates:
            return not_found(question, hits, self.name)

        candidates.sort(reverse=True)
        best_score = candidates[0][0]
        chosen = [c for c in candidates[: self.max_sentences] if c[0] >= 0.5 * best_score]
        chosen.sort(key=lambda c: (c[1], c[2]))  # restore reading order

        parts, citations = [], []
        for _, n, _, sentence in chosen:
            hit = hits[n - 1]
            parts.append(f"{_clean(sentence)} [{n}]")
            # The quote stays verbatim so it can be located in the source chunk.
            citations.append(Citation(n, hit.chunk.id, hit.chunk.label, sentence))
        return Answer(question, " ".join(parts), citations, list(hits), False, self.name)


class ClaudeGenerator:
    """Grounded answers from Claude using document blocks with native citations."""

    name = "claude"

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_tokens: int = 16000,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def build_request(self, question: str, hits: Sequence[SearchHit]) -> dict[str, Any]:
        documents = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": hit.chunk.text},
                "title": hit.chunk.label,
                "citations": {"enabled": True},
            }
            for hit in hits
        ]
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "output_config": {"effort": self.effort},
            # If a safety classifier declines the request, let the API retry it on a
            # fallback model inside the same call.
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
            "messages": [
                {"role": "user", "content": [*documents, {"type": "text", "text": question}]}
            ],
        }

    def generate(self, question: str, hits: Sequence[SearchHit]) -> Answer:
        if not hits:
            return not_found(question, hits, self.name)
        response = self.client.beta.messages.create(**self.build_request(question, hits))
        return self.parse_response(question, hits, response)

    def parse_response(self, question: str, hits: Sequence[SearchHit], response: Any) -> Answer:
        meta = {"model": getattr(response, "model", self.model), "stop_reason": response.stop_reason}
        if response.stop_reason == "refusal":
            answer = Answer(
                question, "The model declined to answer this question.", [], list(hits),
                abstained=True, generator=self.name, meta=meta,
            )
            return answer

        parts: list[str] = []
        citations: list[Citation] = []
        seen: set[tuple[int, str]] = set()
        for block in response.content:
            if block.type != "text":
                continue  # thinking, fallback markers, etc.
            markers: list[int] = []
            for cite in getattr(block, "citations", None) or []:
                idx = getattr(cite, "document_index", None)
                if idx is None or not 0 <= idx < len(hits):
                    continue
                n = idx + 1
                if n not in markers:
                    markers.append(n)
                quote = (cite.cited_text or "").strip()
                if (n, quote) not in seen:
                    seen.add((n, quote))
                    citations.append(Citation(n, hits[idx].chunk.id, hits[idx].chunk.label, quote))
            body = block.text.rstrip()
            trailing = block.text[len(body) :]
            parts.append(body + "".join(f" [{n}]" for n in markers) + trailing)

        text = "".join(parts).strip() or NOT_FOUND
        return Answer(
            question, text, citations, list(hits), abstained=not citations,
            generator=self.name, meta=meta,
        )
