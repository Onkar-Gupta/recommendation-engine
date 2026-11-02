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
from src.evaluate import (
    evaluate_recommendations,
    make_recommend_fn,
    rmse,
    write_metrics,
)
from src.model import build_model
from src.weights import WEIGHTS_FILE, save_weights


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TensorFlow matrix factorisation model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None, help="Path to a custom ratings CSV. Defaults to MovieLens 100K.")
    return parser.parse_args()


def generate_bpr_triplets(train_df, user_items, n_items, n_negatives, threshold, rng):
    """One (user, positive_item, negative_item) triplet per positive rating
    per negative sample. A positive is a train rating >= threshold; a
    negative is an item that user never rated in train. Training the model
    to score positives above negatives is what makes it learn to rank well,
    instead of just predicting a plausible star rating.
    """
    positives = train_df[train_df["rating"] >= threshold]
    users = np.repeat(positives["user_idx"].values, n_negatives)
    pos_items = np.repeat(positives["item_idx"].values, n_negatives)

    neg_items = np.empty(len(users), dtype=np.int64)
    for i, user_idx in enumerate(users):
        seen = user_items[user_idx]
        neg = rng.integers(0, n_items)
        while neg in seen:
            neg = rng.integers(0, n_items)
        neg_items[i] = neg

    return users, pos_items, neg_items


def score_all(model, n_users, n_items, batch_size=8192):
    """Predict every (user, item) pair in one batched call. Looping recommend()
    per user instead would mean hundreds of separate predict() calls, each
    paying TF's per-call overhead (this is what made Prompt 4's first pass
    take 15+ minutes instead of seconds)."""
    all_user_idx = np.repeat(np.arange(n_users), n_items)
    all_item_idx = np.tile(np.arange(n_items), n_users)
    return model.predict([all_user_idx, all_item_idx], batch_size=batch_size, verbose=0).reshape(n_users, n_items)


def train_bpr(model, fit_df, val_df, n_users, n_items, user2idx, item2idx, idx2item, config, seed):
    """Train with a pairwise ranking loss (BPR): push the score of items a
    user liked above items they never interacted with. Early-stops on
    Precision@k measured on the validation split, since ranking quality is
    the actual goal, not training loss.
    """
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    n_negatives = train_cfg.get("bpr_negatives", 4)
    threshold = eval_cfg["relevant_rating_threshold"]
    k = eval_cfg["top_k"]
    patience = train_cfg.get("bpr_patience", 5)
    max_epochs = train_cfg.get("bpr_epochs", 30)
    batch_size = train_cfg["batch_size"]

    rng = np.random.default_rng(seed)
    optimizer = tf.keras.optimizers.Adam(train_cfg.get("bpr_learning_rate", 0.01))
    user_items = fit_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    # Compiled into a graph so each of the ~hundreds of mini-batches per
    # epoch doesn't pay Python/eager dispatch overhead. Without this, a
    # full training run took 3m40s; this is what makes a hyperparameter
    # sweep (many full runs) actually practical.
    @tf.function
    def train_step(u_batch, pos_batch, neg_batch):
        with tf.GradientTape() as tape:
            pos_scores = model([u_batch, pos_batch])
            neg_scores = model([u_batch, neg_batch])
            loss = -tf.reduce_mean(tf.math.log_sigmoid(pos_scores - neg_scores))
            if model.losses:
                loss += tf.add_n(model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    best_score, best_weights, wait = -1.0, None, 0
    for _epoch in range(max_epochs):
        users, pos_items, neg_items = generate_bpr_triplets(fit_df, user_items, n_items, n_negatives, threshold, rng)
        perm = rng.permutation(len(users))
        users, pos_items, neg_items = users[perm], pos_items[perm], neg_items[perm]

        n_batches = max(1, len(users) // batch_size)
        for b in range(n_batches):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            train_step(
                tf.constant(users[sl], dtype=tf.int32),
                tf.constant(pos_items[sl], dtype=tf.int32),
                tf.constant(neg_items[sl], dtype=tf.int32),
            )

        score_matrix = score_all(model, n_users, n_items)
        recommend_fn = make_recommend_fn(score_matrix, user2idx, item2idx, idx2item)
        val_metrics = evaluate_recommendations(recommend_fn, fit_df, val_df, k, threshold)
        score = val_metrics["precision_at_k"]

        if score > best_score:
            best_score, wait = score, 0
            best_weights = [v.numpy() for v in model.trainable_variables]
        else:
            wait += 1
            if wait >= patience:
                break

    if best_weights is not None:
        for var, weights in zip(model.trainable_variables, best_weights):
            var.assign(weights)
    return model


def train(config_path="config.yaml", data_path=None):
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    if data_path is None:
        df = load_ratings("data/raw/ml-100k/u.data")
    else:
        df = load_ratings(data_path, columns=config["data"]["columns"])

    df, user2idx, item2idx = map_ids(df)
    idx2item = {v: k for k, v in item2idx.items()}
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

    objective = config["training"].get("objective", "mse")
    start = time.time()

    if objective == "bpr":
        train_bpr(model, fit_df, val_df, n_users, n_items, user2idx, item2idx, idx2item, config, config["training"]["seed"])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(config["training"]["learning_rate"]), loss="mse")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
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
    save_weights(model, models_dir / WEIGHTS_FILE)  # the server reads this, so it never needs TensorFlow
    (models_dir / "id_mappings.json").write_text(
        json.dumps(
            {
                "user2idx": {str(k): v for k, v in user2idx.items()},
                "item2idx": {str(k): v for k, v in item2idx.items()},
            }
        ),
        encoding="utf-8",
    )

    score_matrix = score_all(model, n_users, n_items)

    # Items with very few training ratings have under-trained, noisy
    # embeddings; keep them out of the recommendable pool (same idea the
    # popularity baseline uses). Mostly relevant to the mse objective; bpr's
    # negative sampling already pushes unsupported items down on its own.
    min_support = config["evaluation"].get("min_item_support", 0)
    item_support = fit_df["item_idx"].value_counts()
    low_support_idx = {i for i in range(n_items) if item_support.get(i, 0) < min_support}

    recommend_fn = make_recommend_fn(score_matrix, user2idx, item2idx, idx2item, low_support_idx)

    y_pred = model.predict([test_df["user_idx"].values, test_df["item_idx"].values], verbose=0).flatten()

    metrics = {
        "rmse": rmse(test_df["rating"].values, y_pred),
        **evaluate_recommendations(
            recommend_fn,
            train_df,
            test_df,
            config["evaluation"]["top_k"],
            config["evaluation"]["relevant_rating_threshold"],
            n_items,
        ),
        "train_seconds": round(train_seconds, 1),
        "objective": objective,
    }
    write_metrics("tf_mf", metrics, path=f"{config['paths']['reports_dir']}/metrics.json")
    return metrics


if __name__ == "__main__":
    args = parse_args()
    result = train(config_path=args.config, data_path=args.data)
    print(
        f"tf_mf ({result['objective']}): rmse={result['rmse']:.3f} precision@k={result['precision_at_k']:.3f} "
        f"recall@k={result['recall_at_k']:.3f} ndcg@k={result['ndcg_at_k']:.3f} "
        f"coverage={result['catalogue_coverage']:.1%} train_seconds={result['train_seconds']}"
    )
