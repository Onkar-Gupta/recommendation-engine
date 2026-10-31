"""Train the TensorFlow matrix factorisation model, save it plus ID mappings,
then evaluate against the same metrics as the baselines.

Usage:
    python -m src.train --config config.yaml
    python -m src.train --data path/to/custom.csv --config config.yaml
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data_loader import load_config, load_ratings, map_ids, split_train_test
from src.evaluate import evaluate_recommendations, rmse, write_metrics
from src.model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TensorFlow matrix factorisation model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None, help="Path to a custom ratings CSV. Defaults to MovieLens 100K.")
    return parser.parse_args()


def train(config_path="config.yaml", data_path=None):
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    if data_path is None:
        df = load_ratings("data/raw/ml-100k/u.data")
    else:
        df = load_ratings(data_path, columns=config["data"]["columns"])

    df, user2idx, item2idx = map_ids(df)
    n_users, n_items = len(user2idx), len(item2idx)

    train_df, test_df = split_train_test(
        df,
        test_size=config["training"]["test_split"],
        seed=config["training"]["seed"],
        min_ratings_per_user=config["data"]["min_ratings_per_user"],
    )
    # small validation split carved out of train, for early stopping only
    fit_df, val_df = split_train_test(
        train_df,
        test_size=0.1,
        seed=config["training"]["seed"],
        min_ratings_per_user=config["data"]["min_ratings_per_user"],
    )

    global_mean = fit_df["rating"].mean()
    model = build_model(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=config["training"]["embedding_dim"],
        global_mean=global_mean,
        l2_reg=config["training"].get("l2_reg", 0.0),
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(config["training"]["learning_rate"]), loss="mse")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    start = time.time()
    model.fit(
        [fit_df["user_idx"].values, fit_df["item_idx"].values],
        fit_df["rating"].values,
        validation_data=([val_df["user_idx"].values, val_df["item_idx"].values], val_df["rating"].values),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        callbacks=[early_stopping],
        verbose=0,
    )
    train_seconds = time.time() - start

    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "tf_mf.keras")
    (models_dir / "id_mappings.json").write_text(
        json.dumps(
            {
                "user2idx": {str(k): v for k, v in user2idx.items()},
                "item2idx": {str(k): v for k, v in item2idx.items()},
            }
        ),
        encoding="utf-8",
    )

    idx2item = {v: k for k, v in item2idx.items()}

    # Score every (user, item) pair in one batched predict() call. Calling
    # recommend() (which calls predict()) once per test user instead would
    # mean ~900 separate predict() calls, each paying TF's per-call overhead.
    all_user_idx = np.repeat(np.arange(n_users), n_items)
    all_item_idx = np.tile(np.arange(n_items), n_users)
    score_matrix = model.predict([all_user_idx, all_item_idx], batch_size=8192, verbose=0).reshape(n_users, n_items)

    def recommend_fn(user_id, k, exclude_item_ids):
        if user_id not in user2idx:
            return []
        scores = score_matrix[user2idx[user_id]].copy()
        for item_id in exclude_item_ids:
            if item_id in item2idx:
                scores[item2idx[item_id]] = -np.inf
        top_idx = np.argsort(-scores)[:k]
        return [idx2item[i] for i in top_idx]

    y_pred = model.predict([test_df["user_idx"].values, test_df["item_idx"].values], verbose=0).flatten()

    metrics = {
        "rmse": rmse(test_df["rating"].values, y_pred),
        **evaluate_recommendations(
            recommend_fn, train_df, test_df, config["evaluation"]["top_k"], config["evaluation"]["relevant_rating_threshold"]
        ),
        "train_seconds": round(train_seconds, 1),
    }
    write_metrics("tf_mf", metrics, path=f"{config['paths']['reports_dir']}/metrics.json")
    return metrics


if __name__ == "__main__":
    args = parse_args()
    result = train(config_path=args.config, data_path=args.data)
    print(
        f"tf_mf: rmse={result['rmse']:.3f} precision@k={result['precision_at_k']:.3f} "
        f"recall@k={result['recall_at_k']:.3f} train_seconds={result['train_seconds']}"
    )
