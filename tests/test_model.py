"""Tests for the TensorFlow matrix factorisation model."""

import tensorflow as tf

from rec_engine.data_loader import load_ratings, map_ids
from rec_engine.model import build_model, recommend

SAMPLE_COLUMNS = {"user": "user_id", "item": "item_id", "rating": "rating", "timestamp": "timestamp"}


def _load_sample():
    df = load_ratings("data/sample/ratings_sample.csv", columns=SAMPLE_COLUMNS)
    return map_ids(df)


def test_model_builds():
    df, user2idx, item2idx = _load_sample()
    model = build_model(len(user2idx), len(item2idx), embedding_dim=8, global_mean=df["rating"].mean())

    output = model.predict([df["user_idx"].values[:5], df["item_idx"].values[:5]], verbose=0)
    assert output.shape == (5,)


def test_training_step_reduces_loss():
    df, user2idx, item2idx = _load_sample()
    tf.random.set_seed(42)
    model = build_model(len(user2idx), len(item2idx), embedding_dim=8, global_mean=df["rating"].mean())
    model.compile(optimizer=tf.keras.optimizers.Adam(0.05), loss="mse")

    inputs = [df["user_idx"].values, df["item_idx"].values]
    targets = df["rating"].values

    history = model.fit(inputs, targets, epochs=5, batch_size=32, verbose=0)
    losses = history.history["loss"]
    assert losses[-1] < losses[0]


def test_recommend_returns_k_unique_unseen_items():
    df, user2idx, item2idx = _load_sample()
    model = build_model(len(user2idx), len(item2idx), embedding_dim=8, global_mean=df["rating"].mean())

    user_index = 0
    seen = set(df.loc[df["user_idx"] == user_index, "item_idx"])
    k = 5

    recommended = recommend(model, user_index, k, exclude_seen=seen, n_items=len(item2idx))

    assert len(recommended) == k
    assert len(set(recommended)) == k
    assert seen.isdisjoint(recommended)
