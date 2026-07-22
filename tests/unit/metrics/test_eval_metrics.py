from minionerec.evaluate_official import hr_at_k, ndcg_at_k, compute_metrics


def test_hr_ndcg_handcrafted():
    # rank 0 (1st)
    assert hr_at_k(0, 3) == 1.0
    assert abs(ndcg_at_k(0, 3) - 1.0) < 1e-9
    # rank 2 (3rd)
    assert hr_at_k(2, 3) == 1.0
    assert hr_at_k(2, 1) == 0.0
    # rank 9 (10th)
    assert hr_at_k(9, 10) == 1.0
    assert hr_at_k(9, 5) == 0.0
    # miss
    assert hr_at_k(None, 10) == 0.0
    assert ndcg_at_k(None, 10) == 0.0

    preds = [["t", "a", "b"], ["a", "b", "t"], ["a", "b", "c"]]
    targets = ["t", "t", "t"]
    m = compute_metrics(preds, targets, ks=(3,))
    assert abs(m["HR@3"] - 2 / 3) < 1e-9
