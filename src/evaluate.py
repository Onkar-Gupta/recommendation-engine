"""Evaluation metrics: RMSE, Precision@K, Recall@K."""

import json
from pathlib import Path

import numpy as np

DEFAULT_RELEVANT_THRESHOLD = 4


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_at_k(recommended, relevant, k):
    """recommended: ranked list of item ids. relevant: iterable of relevant item ids."""
    top_k = list(recommended)[:k]
    if not top_k:
        return 0.0
    relevant = set(relevant)
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(recommended, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top_k = list(recommended)[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    """Normalised Discounted Cumulative Gain. Unlike precision/recall, this
    cares about ORDER within the top-k: a relevant item at rank 1 counts for
    more than the same item at rank 10. Binary relevance (in `relevant` or
    not), the standard choice for this metric in recommender literature.
    """
    relevant = set(relevant)
    top_k = list(recommended)[:k]
    if not relevant or not top_k:
        return 0.0

    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(recommend_fn, train_df, test_df, k, threshold=DEFAULT_RELEVANT_THRESHOLD, n_items=None):
    """Average Precision@k, Recall@k and NDCG@k across test users who have at
    least one relevant item (test rating >= threshold). Also reports
    catalogue coverage: how many distinct items ever get recommended across
    all evaluated users. A model can score well on precision/recall while
    recommending the same narrow handful of items to everyone (this
    happened in this project — see CLAUDE.md) and coverage is what catches
    that; precision/recall alone don't.

    `recommend_fn(user_id, k, exclude)` must return a ranked list of item ids
    with items in `exclude` (what the user already rated in train) left out.
    Pass `n_items` (total catalogue size) to get coverage as a fraction.
    """
    precisions, recalls, ndcgs = [], [], []
    recommended_items_seen = set()
    for user_id, group in test_df.groupby("user_id"):
        relevant = set(group.loc[group["rating"] >= threshold, "item_id"])
        if not relevant:
            continue
        seen = set(train_df.loc[train_df["user_id"] == user_id, "item_id"])
        recommended = recommend_fn(user_id, k, seen)
        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        recommended_items_seen.update(recommended)

    metrics = {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_evaluated": len(precisions),
        "n_unique_items_recommended": len(recommended_items_seen),
    }
    if n_items:
        metrics["catalogue_coverage"] = len(recommended_items_seen) / n_items
    return metrics


def write_metrics(name, metrics, path="reports/metrics.json"):
    """Record results for `name` (a model/baseline) into the shared metrics file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    all_metrics[name] = metrics
    path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    return all_metrics
