"""Hand-calculated tests for rmse, precision_at_k, recall_at_k."""

import math

from src.evaluate import precision_at_k, recall_at_k, rmse


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
