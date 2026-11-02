"""Tests for the data loading pipeline. Run against both sample datasets to
prove the loader works on any user-item-rating CSV shape, not just MovieLens.
"""

import pandas as pd
import pytest

from src.data_loader import dataset_stats, load_ratings, map_ids, split_train_test

MOVIE_COLUMNS = {"user": "user_id", "item": "item_id", "rating": "rating", "timestamp": "timestamp"}
ECOMMERCE_COLUMNS = {"user": "user_id", "item": "product_id", "rating": "rating", "timestamp": "timestamp"}

CASES = [
    ("data/sample/ratings_sample.csv", MOVIE_COLUMNS),
    ("data/sample/ecommerce_sample.csv", ECOMMERCE_COLUMNS),
]


@pytest.mark.parametrize("path,columns", CASES)
def test_load_ratings_shape(path, columns):
    df = load_ratings(path, columns=columns)
    assert list(df.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert len(df) == 200


@pytest.mark.parametrize("path,columns", CASES)
def test_id_mapping_round_trip(path, columns):
    df = load_ratings(path, columns=columns)
    mapped, user2idx, item2idx = map_ids(df)

    assert mapped["user_idx"].min() == 0
    assert mapped["user_idx"].max() == len(user2idx) - 1
    assert mapped["item_idx"].min() == 0
    assert mapped["item_idx"].max() == len(item2idx) - 1

    idx2user = {idx: uid for uid, idx in user2idx.items()}
    assert all(idx2user[idx] == uid for uid, idx in user2idx.items())


@pytest.mark.parametrize("path,columns", CASES)
def test_split_no_leakage_and_reproducible(path, columns):
    df = load_ratings(path, columns=columns)
    train, test = split_train_test(df, test_size=0.2, seed=42, min_ratings_per_user=5)

    # train and test are a clean partition, no shared rows
    merged = pd.merge(train, test, how="inner")
    assert len(merged) == 0

    # every user with >=5 ratings appears in both train and test
    counts = df.groupby("user_id").size()
    heavy_users = set(counts[counts >= 5].index)
    assert heavy_users.issubset(set(train["user_id"].unique()))
    assert heavy_users.issubset(set(test["user_id"].unique()))

    # reproducible for the same seed
    train2, test2 = split_train_test(df, test_size=0.2, seed=42, min_ratings_per_user=5)
    pd.testing.assert_frame_equal(train.reset_index(drop=True), train2.reset_index(drop=True))
    pd.testing.assert_frame_equal(test.reset_index(drop=True), test2.reset_index(drop=True))


@pytest.mark.parametrize("path,columns", CASES)
def test_dataset_stats(path, columns):
    df = load_ratings(path, columns=columns)
    stats = dataset_stats(df)
    assert stats["n_ratings"] == len(df)
    assert 0 <= stats["sparsity"] <= 1


def test_load_item_genres_reads_flags_and_skips_unknown(tmp_path):
    from src.data_loader import MOVIELENS_GENRES, load_item_genres

    flags = ["0"] * len(MOVIELENS_GENRES)
    flags[MOVIELENS_GENRES.index("Comedy")] = "1"
    flags[MOVIELENS_GENRES.index("Romance")] = "1"
    line = "|".join(["7", "Test Film (1999)", "01-Jan-1999", "", "http://x"] + flags)
    path = tmp_path / "u.item"
    path.write_text(line + "\n", encoding="latin-1")

    assert load_item_genres(path) == {7: ["Comedy", "Romance"]}
    assert load_item_genres(tmp_path / "missing") == {}


def test_empty_titles_path_means_no_titles_not_a_crash():
    from src.data_loader import load_item_genres, load_item_titles

    assert load_item_titles("") == {}
    assert load_item_genres("") == {}
