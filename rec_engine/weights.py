"""Save a trained model's numbers to a small .npz file and score from them with plain NumPy.

The model is just a table of numbers per user and per item, so serving does not need
TensorFlow: score(user, item) = global_mean + user_bias + item_bias + user_vector . item_vector.
This module deliberately imports nothing but NumPy.
"""

from pathlib import Path

import numpy as np

WEIGHTS_FILE = "mf_weights.npz"


def save_weights(model, path):
    """Write the embeddings and biases of a trained MatrixFactorizationModel."""
    np.savez(
        path,
        user_vectors=model.user_embedding.get_weights()[0],
        item_vectors=model.item_embedding.get_weights()[0],
        user_bias=model.user_bias.get_weights()[0][:, 0],
        item_bias=model.item_bias.get_weights()[0][:, 0],
        global_mean=np.float32(model.global_mean.numpy()),
    )


def load_weights(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Train the model first: rec-engine train (or python -m rec_engine.train)")
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def score_matrix(weights):
    """Predicted score for every (user, item) pair, shape (n_users, n_items)."""
    scores = weights["user_vectors"] @ weights["item_vectors"].T
    scores += weights["user_bias"][:, None] + weights["item_bias"][None, :]
    return scores + weights["global_mean"]


def unit_vectors(vectors):
    """Item vectors scaled to length 1, so a dot product is the cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
