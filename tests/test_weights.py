"""The NumPy scoring used by the server must match the TensorFlow model exactly."""

import numpy as np

from src.model import build_model
from src.train import score_all
from src.weights import load_weights, save_weights, score_matrix, unit_vectors


def _random_model(n_users=7, n_items=11, dim=5):
    model = build_model(n_users, n_items, embedding_dim=dim, global_mean=3.4)
    model([np.array([0]), np.array([0])])  # build the layers
    # Fresh biases start at zero, which would hide a bias bug, so give every table real values.
    rng = np.random.default_rng(0)
    for layer in (model.user_embedding, model.item_embedding, model.user_bias, model.item_bias):
        layer.set_weights([rng.normal(size=layer.get_weights()[0].shape).astype("float32")])
    return model


def test_numpy_scores_match_tensorflow(tmp_path):
    model = _random_model()
    save_weights(model, tmp_path / "w.npz")

    from_numpy = score_matrix(load_weights(tmp_path / "w.npz"))
    from_tensorflow = score_all(model, 7, 11)

    assert from_numpy.shape == (7, 11)
    assert np.allclose(from_numpy, from_tensorflow, atol=1e-5)


def test_missing_weights_file_says_how_to_fix_it(tmp_path):
    try:
        load_weights(tmp_path / "nope.npz")
    except FileNotFoundError as error:
        assert "python -m src.train" in str(error)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_unit_vectors_have_length_one():
    lengths = np.linalg.norm(unit_vectors(np.array([[3.0, 4.0], [0.0, 2.0]])), axis=1)
    assert np.allclose(lengths, 1.0)
