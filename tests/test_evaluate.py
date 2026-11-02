"""Hand-calculated tests for rmse, precision_at_k, recall_at_k, ndcg_at_k."""

import math

import pandas as pd

from rec_engine.evaluate import (
    evaluate_recommendations,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


def test_rmse_perfect():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmse_known_value():
    # constant error of 2 on every prediction -> rmse == 2
    assert math.isclose(rmse([3, 3, 3], [1, 1, 1]), 2.0)


def test_precision_at_k_known_value():
    recommended = [1, 2, 3, 4, 5]
    relevant = {1, 3, 5, 7}
    assert math.isclose(precision_at_k(recommended, relevant, k=5), 3 / 5)
    assert math.isclose(precision_at_k(recommended, relevant, k=3), 2 / 3)


def test_precision_at_k_no_recommendations():
    assert precision_at_k([], {1, 2}, k=5) == 0.0


def test_recall_at_k_known_value():
    recommended = [1, 2, 3, 4, 5]
    relevant = {1, 3, 5, 7}
    assert math.isclose(recall_at_k(recommended, relevant, k=5), 3 / 4)
    assert math.isclose(recall_at_k(recommended, relevant, k=3), 2 / 4)


def test_recall_at_k_no_relevant_items():
    assert recall_at_k([1, 2, 3], set(), k=5) == 0.0


def test_ndcg_at_k_perfect_ranking():
    # both relevant items at the top, in any order -> NDCG == 1.0
    assert math.isclose(ndcg_at_k([1, 2], {1, 2}, k=2), 1.0)


def test_ndcg_at_k_known_value():
    # one relevant item (2) at rank 2 (0-indexed position 1) out of 3
    # dcg = 1/log2(1+2); ideal dcg (hit at rank 1) = 1/log2(0+2) = 1
    recommended, relevant, k = [1, 2, 3], {2}, 3
    expected = (1 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k(recommended, relevant, k), expected)


def test_ndcg_at_k_no_hits():
    assert ndcg_at_k([1, 2, 3], {9}, k=3) == 0.0


def test_ndcg_at_k_no_relevant_items():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_ndcg_at_k_no_recommendations():
    assert ndcg_at_k([], {1, 2}, k=3) == 0.0


def test_catalogue_coverage_catches_narrow_recommendations():
    # a model that recommends the SAME 2 items to every user, regardless of
    # who they are, should score low coverage even if precision/recall look
    # fine -- this is exactly the failure mode that slipped through once
    # (see CLAUDE.md) before this metric existed.
    train_df = pd.DataFrame({"user_id": [1, 2], "item_id": [100, 100], "rating": [5, 5]})
    test_df = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 11], "rating": [5, 5]})

    def narrow_recommend_fn(user_id, k, exclude):
        return [10, 11]

    metrics = evaluate_recommendations(narrow_recommend_fn, train_df, test_df, k=10, threshold=4, n_items=1000)

    assert metrics["n_unique_items_recommended"] == 2
    assert math.isclose(metrics["catalogue_coverage"], 2 / 1000)
