import math

import pytest

from ragdoc.retrieval import BM25, reciprocal_rank_fusion


def test_bm25_prefers_documents_with_rarer_matching_terms():
    corpus = [["deploy", "friday"], ["deploy", "monday"], ["lunch", "menu"]]
    scores = BM25().fit(corpus).scores(["friday", "deploy"])
    assert scores[0] > scores[1] > scores[2] == 0


def test_bm25_idf_for_unseen_terms_is_maximal():
    bm25 = BM25().fit([["a"], ["a", "b"], ["c"]])
    assert bm25.idf("zzz") > bm25.idf("b") > bm25.idf("a")
    assert math.isclose(bm25.idf("zzz"), math.log(1 + 3.5 / 0.5))


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]], k=60)
    assert fused[1] == fused[2] > fused[3] > 0
    assert math.isclose(fused[1], 1 / 61 + 1 / 62)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_every_mode_finds_the_deployment_policy(kb, mode):
    hits = kb.search("Are production deploys allowed on Fridays?", k=3, mode=mode)
    assert hits[0].chunk.doc_id == "07-deployment"
    assert [h.rank for h in hits] == [1, 2, 3]


def test_unknown_mode_is_rejected(kb):
    with pytest.raises(ValueError):
        kb.search("anything", mode="magic")


def test_k_is_clamped_to_corpus_size(kb):
    assert len(kb.search("leave", k=10_000)) == len(kb.chunks)
