"""Tests for the BPR triplet generation used by the ranking-loss training path."""

import numpy as np

from src.data_loader import load_ratings, map_ids
from src.train import generate_bpr_triplets

SAMPLE_COLUMNS = {"user": "user_id", "item": "item_id", "rating": "rating", "timestamp": "timestamp"}


def _load_sample():
    df = load_ratings("data/sample/ratings_sample.csv", columns=SAMPLE_COLUMNS)
    return map_ids(df)


def test_triplet_shapes_and_positive_count():
    df, _user2idx, item2idx = _load_sample()
    n_negatives = 3
    threshold = 4
    user_items = df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    users, pos_items, neg_items = generate_bpr_triplets(
        df, user_items, len(item2idx), n_negatives, threshold, np.random.default_rng(42)
    )

    n_positives = (df["rating"] >= threshold).sum()
    assert len(users) == len(pos_items) == len(neg_items) == n_positives * n_negatives


def test_negatives_are_never_items_the_user_rated():
    df, _user2idx, item2idx = _load_sample()
    user_items = df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    users, _pos_items, neg_items = generate_bpr_triplets(
        df, user_items, len(item2idx), n_negatives=4, threshold=4, rng=np.random.default_rng(42)
    )

    for user_idx, neg_item in zip(users, neg_items):
        assert neg_item not in user_items[user_idx]


def test_positives_are_real_high_ratings():
    df, _user2idx, item2idx = _load_sample()
    user_items = df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    threshold = 4

    users, pos_items, _ = generate_bpr_triplets(
        df, user_items, len(item2idx), n_negatives=2, threshold=threshold, rng=np.random.default_rng(42)
    )

    rated = df.set_index(["user_idx", "item_idx"])["rating"]
    for user_idx, pos_item in zip(users, pos_items):
        assert rated.loc[(user_idx, pos_item)] >= threshold
