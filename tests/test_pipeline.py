import pytest

from ragdoc.generation import NOT_FOUND, ExtractiveGenerator
from ragdoc.pipeline import RAGPipeline


@pytest.fixture()
def pipeline(kb):
    return RAGPipeline(kb, ExtractiveGenerator())


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Can I deploy to production on a Friday?", "no production deploys on Fridays"),
        ("Who approves an expense above 1,000 USD?", "approval from a director"),
        ("How long is the probation period?", "probation period is three months"),
    ],
)
def test_answers_contain_the_policy_text(pipeline, question, expected):
    answer = pipeline.ask(question)
    assert not answer.abstained
    assert expected.lower() in " ".join(answer.text.split()).lower()
    assert answer.latency_ms > 0


@pytest.mark.parametrize(
    "question",
    ["What is the capital of France?", "Explain quantum chromodynamics", "What's the company's stock price?"],
)
def test_out_of_scope_questions_abstain(pipeline, question):
    answer = pipeline.ask(question)
    assert answer.abstained
    assert answer.text == NOT_FOUND


def test_empty_question_is_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.ask("   ")
