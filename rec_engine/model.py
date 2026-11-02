"""TensorFlow embedding-based matrix factorisation (SVD-style) recommender model."""

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="rec_engine")
class MatrixFactorizationModel(tf.keras.Model):
    """User embedding, item embedding, user bias, item bias, global mean,
    combined with a dot product. L2 regularisation on every embedding table.
    """

    def __init__(self, n_users, n_items, embedding_dim, global_mean=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.global_mean = tf.constant(global_mean, dtype=tf.float32)
        self._l2_reg = l2_reg
        reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None

        self.user_embedding = tf.keras.layers.Embedding(
            n_users, embedding_dim, embeddings_regularizer=reg, name="user_embedding"
        )
        self.item_embedding = tf.keras.layers.Embedding(
            n_items, embedding_dim, embeddings_regularizer=reg, name="item_embedding"
        )
        self.user_bias = tf.keras.layers.Embedding(n_users, 1, embeddings_regularizer=reg, name="user_bias")
        self.item_bias = tf.keras.layers.Embedding(n_items, 1, embeddings_regularizer=reg, name="item_bias")

    def call(self, inputs):
        user_idx, item_idx = inputs
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        dot = tf.reduce_sum(user_vec * item_vec, axis=-1)
        bias = tf.squeeze(self.user_bias(user_idx), axis=-1) + tf.squeeze(self.item_bias(item_idx), axis=-1)
        return self.global_mean + bias + dot

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_users": self.user_embedding.input_dim,
                "n_items": self.item_embedding.input_dim,
                "embedding_dim": self.user_embedding.output_dim,
                "global_mean": float(self.global_mean.numpy()),
                "l2_reg": self._l2_reg,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_model(n_users, n_items, embedding_dim, global_mean=0.0, l2_reg=0.0):
    return MatrixFactorizationModel(n_users, n_items, embedding_dim, global_mean, l2_reg)


def recommend(model, user_index, k, exclude_seen=None, n_items=None):
    """Rank all items for one user, excluding items in exclude_seen (item
    indices), return the top k item indices by predicted rating.
    """
    exclude_seen = exclude_seen or set()
    if n_items is None:
        n_items = model.item_embedding.input_dim

    candidates = [i for i in range(n_items) if i not in exclude_seen]
    if not candidates:
        return []

    user_batch = np.full(len(candidates), user_index, dtype=np.int32)
    item_batch = np.array(candidates, dtype=np.int32)
    scores = model.predict([user_batch, item_batch], verbose=0).flatten()

    ranked = [item for item, _ in sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)]
    return ranked[:k]
