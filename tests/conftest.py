from pathlib import Path

import pytest

from ragdoc.knowledge_base import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "data" / "sample_docs"
QUESTIONS = ROOT / "eval" / "questions.jsonl"


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_directory(SAMPLE_DOCS)
