from conftest import QUESTIONS

from ragdoc.evaluation import evaluate, format_table, load_questions, mean_reciprocal_rank, recall_at_k


def test_metric_math():
    ranks = [1, 3, None, 2]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.75
    assert mean_reciprocal_rank(ranks) == (1 + 1 / 3 + 1 / 2) / 4


def test_retrieval_quality_does_not_regress(kb):
    """Guard rail: recall@5 on the bundled question set must stay at or above 0.85."""
    results = evaluate(kb, load_questions(QUESTIONS))
    for mode, metrics in results.items():
        assert metrics["recall@5"] >= 0.85, (mode, metrics)
    table = format_table(results)
    assert table.splitlines()[0].startswith("| Retriever | Recall@1")
