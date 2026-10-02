"""Retrieval evaluation: Recall@k and MRR against a labelled question set.

Each question names the document that answers it and a short *evidence* span copied from that
document. A retrieved chunk counts as relevant when it comes from that document and contains the
evidence, so labels stay valid when chunking parameters change.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragdoc.knowledge_base import KnowledgeBase
from ragdoc.retrieval import MODES, Mode, SearchHit


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    doc: str
    evidence: str


def load_questions(path: str | Path) -> list[EvalQuestion]:
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(EvalQuestion(**json.loads(line)))
    return questions


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(hit: SearchHit, q: EvalQuestion) -> bool:
    return hit.chunk.doc_id == q.doc and _norm(q.evidence) in _norm(hit.chunk.text)


def first_relevant_rank(hits: Sequence[SearchHit], q: EvalQuestion) -> int | None:
    for hit in hits:
        if is_relevant(hit, q):
            return hit.rank
    return None


def recall_at_k(ranks: Iterable[int | None], k: int) -> float:
    ranks = list(ranks)
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mean_reciprocal_rank(ranks: Iterable[int | None]) -> float:
    ranks = list(ranks)
    return sum(1 / r for r in ranks if r is not None) / len(ranks) if ranks else 0.0


def evaluate(
    kb: KnowledgeBase,
    questions: Sequence[EvalQuestion],
    *,
    modes: Sequence[Mode] = MODES,
    ks: Sequence[int] = (1, 3, 5),
    depth: int = 10,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for mode in modes:
        ranks = [first_relevant_rank(kb.search(q.question, k=depth, mode=mode), q) for q in questions]
        metrics = {f"recall@{k}": recall_at_k(ranks, k) for k in ks}
        metrics[f"mrr@{depth}"] = mean_reciprocal_rank(ranks)
        results[mode] = metrics
    return results


def format_table(results: dict[str, dict[str, float]]) -> str:
    metrics = list(next(iter(results.values())).keys())
    headers = [m.replace("recall", "Recall").replace("mrr", "MRR") for m in metrics]
    lines = ["| Retriever | " + " | ".join(headers) + " |", "|---|" + "---:|" * len(metrics)]
    for mode, values in results.items():
        lines.append(f"| {mode} | " + " | ".join(f"{values[m]:.3f}" for m in metrics) + " |")
    return "\n".join(lines)
