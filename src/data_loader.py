"""Load user-item rating data, map IDs to contiguous indices, and split train/test."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STANDARD_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
_KEY_MAP = {"user": "user_id", "item": "item_id", "rating": "rating", "timestamp": "timestamp"}


def load_config(path="config.yaml"):
    """Load config.yaml. All paths and hyperparameters live here, not in code."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ratings(path, columns=None):
    """Load ratings from either MovieLens's u.data (tab-separated, no header)
    or any CSV with user/item/rating/timestamp columns.

    `columns` maps the short names (user, item, rating, timestamp) to the
    actual column names in a custom CSV, e.g. config.yaml's data.columns
    section: {"user": "user_id", "item": "product_id", ...}. This is what
    lets people swap in their own data without editing any code.
    """
    path = Path(path)

    if path.name == "u.data":
        df = pd.read_csv(path, sep="\t", header=None, names=STANDARD_COLUMNS)
    else:
        df = pd.read_csv(path)
        if columns:
            rename = {actual: _KEY_MAP.get(short, short) for short, actual in columns.items()}
            df = df.rename(columns=rename)

    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}. Check config.yaml's data.columns mapping.")
    return df[STANDARD_COLUMNS].copy()


def map_ids(df):
    """Map raw user_id/item_id to contiguous integer indices starting at 0.
    Returns the dataframe with user_idx/item_idx columns added, plus both
    id-to-index mappings.
    """
    user_ids = sorted(df["user_id"].unique())
    item_ids = sorted(df["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["item_idx"] = df["item_id"].map(item2idx)
    return df, user2idx, item2idx


def split_train_test(df, test_size=0.2, seed=42, min_ratings_per_user=5):
    """Per-user train/test split. A user needs at least `min_ratings_per_user`
    ratings to appear in both train and test; users below that threshold go
    entirely to train. Reproducible for a fixed seed.
    """
    train_parts, test_parts = [], []
    for _, group in df.groupby("user_id"):
        group = group.sample(frac=1, random_state=seed)
        if len(group) >= min_ratings_per_user:
            n_test = max(1, int(round(len(group) * test_size)))
            test_parts.append(group.iloc[:n_test])
            train_parts.append(group.iloc[n_test:])
        else:
            train_parts.append(group)

    train = pd.concat(train_parts).reset_index(drop=True) if train_parts else df.iloc[0:0].copy()
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else df.iloc[0:0].copy()
    return train, test


def load_item_titles(path):
    """Load MovieLens u.item (pipe-separated, latin-1) into {item_id: title}.
    Optional: returns {} if the file does not exist, since custom datasets
    may have no title file.
    """
    path = Path(path)
    if not path.exists():
        return {}

    titles = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 2:
                titles[int(parts[0])] = parts[1]
    return titles


def dataset_stats(df):
    """Basic stats used in the README and as a sanity check after loading."""
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_ratings = len(df)
    sparsity = 1 - n_ratings / (n_users * n_items)
    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
    }
