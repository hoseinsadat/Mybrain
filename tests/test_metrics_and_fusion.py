from mybrain.evalx.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from mybrain.retrieval.hybrid import rrf_fuse


def test_rrf_prefers_items_in_both_lists():
    fused = rrf_fuse([("vector0", ["a", "b", "c"], 1.0), ("lexical0", ["c", "d"], 1.0)], k=60)
    ranked = sorted(fused.values(), key=lambda c: -c.score)
    assert ranked[0].chunk_id == "c"
    assert fused["c"].vector_rank == 3 and fused["c"].lexical_rank == 1
    assert fused["a"].lexical_rank is None


def test_metrics():
    ranked = ["x", "g1", "y", "g2"]
    gold = ["g1", "g2"]
    assert hit_at_k(ranked, gold, 1) == 0.0 and hit_at_k(ranked, gold, 2) == 1.0
    assert recall_at_k(ranked, gold, 2) == 0.5 and recall_at_k(ranked, gold, 4) == 1.0
    assert abs(mrr(ranked, gold) - 0.5) < 1e-9
    assert 0 < ndcg_at_k(ranked, gold, 10) < 1
    assert ndcg_at_k(["g1", "g2"], gold, 10) == 1.0
    assert ndcg_at_k(ranked, [], 10) == 0.0
