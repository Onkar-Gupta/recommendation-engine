"""Popularity and TruncatedSVD baselines. Neither is tuned; they exist to give
the TensorFlow model (Prompt 4) something real to beat.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.data_loader import load_config, load_ratings, split_train_test
from src.evaluate import evaluate_recommendations, rmse, write_metrics


class PopularityBaseline:
    """Recommends the most-rated items with a mean rating at or above a
    threshold. Same list for every user, no personalisation: the simplest
    possible baseline.
    """

    def __init__(self, min_mean_rating=3.5):
        self.min_mean_rating = min_mean_rating
        self.ranked_items_ = []
        self.global_mean_ = 0.0

    def fit(self, train_df):
        self.global_mean_ = train_df["rating"].mean()
        stats = train_df.groupby("item_id")["rating"].agg(["mean", "count"])
        qualifying = stats[stats["mean"] >= self.min_mean_rating]
        self.ranked_items_ = qualifying.sort_values("count", ascending=False).index.tolist()
        return self

    def predict_rating(self, user_id, item_id):
        return self.global_mean_

    def recommend(self, user_id, k, exclude=None):
        exclude = exclude or set()
        return [item for item in self.ranked_items_ if item not in exclude][:k]


class TruncatedSVDBaseline:
    """SVD over the user-item rating matrix (missing entries filled with the
    global mean), used to predict ratings and rank unseen items per user.
    """

    def __init__(self, n_components=20, seed=42):
        self.n_components = n_components
        self.seed = seed

    def fit(self, train_df):
        self.user_ids_ = sorted(train_df["user_id"].unique())
        self.item_ids_ = sorted(train_df["item_id"].unique())
        self.user_pos_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_pos_ = {it: i for i, it in enumerate(self.item_ids_)}

        self.global_mean_ = train_df["rating"].mean()
        matrix = np.full((len(self.user_ids_), len(self.item_ids_)), self.global_mean_)
        for user_id, item_id, rating in zip(train_df["user_id"], train_df["item_id"], train_df["rating"]):
            matrix[self.user_pos_[user_id], self.item_pos_[item_id]] = rating

        n_components = min(self.n_components, min(matrix.shape) - 1)
        self.svd_ = TruncatedSVD(n_components=n_components, random_state=self.seed)
        user_factors = self.svd_.fit_transform(matrix)
        self.predicted_ = user_factors @ self.svd_.components_
        return self

    def predict_rating(self, user_id, item_id):
        if user_id not in self.user_pos_ or item_id not in self.item_pos_:
            return self.global_mean_
        value = self.predicted_[self.user_pos_[user_id], self.item_pos_[item_id]]
        return float(np.clip(value, 1, 5))

    def recommend(self, user_id, k, exclude=None):
        exclude = exclude or set()
        if user_id not in self.user_pos_:
            return []
        scores = self.predicted_[self.user_pos_[user_id]]
        ranked = sorted(self.item_ids_, key=lambda item: scores[self.item_pos_[item]], reverse=True)
        return [item for item in ranked if item not in exclude][:k]


def _rmse_for(model, test_df):
    y_true = test_df["rating"].tolist()
    y_pred = [model.predict_rating(u, i) for u, i in zip(test_df["user_id"], test_df["item_id"])]
    return rmse(y_true, y_pred)


def run_baselines(config_path="config.yaml"):
    config = load_config(config_path)
    df = load_ratings("data/raw/ml-100k/u.data")
    train, test = split_train_test(
        df,
        test_size=config["training"]["test_split"],
        seed=config["training"]["seed"],
        min_ratings_per_user=config["data"]["min_ratings_per_user"],
    )
    k = config["evaluation"]["top_k"]
    threshold = config["evaluation"]["relevant_rating_threshold"]
    n_items = df["item_id"].nunique()

    results = {}

    popularity = PopularityBaseline(min_mean_rating=config["baselines"]["popularity_min_rating"]).fit(train)
    results["popularity"] = {
        "rmse": _rmse_for(popularity, test),
        **evaluate_recommendations(popularity.recommend, train, test, k, threshold, n_items),
    }

    svd = TruncatedSVDBaseline(
        n_components=config["baselines"]["svd_n_components"], seed=config["training"]["seed"]
    ).fit(train)
    results["truncated_svd"] = {
        "rmse": _rmse_for(svd, test),
        **evaluate_recommendations(svd.recommend, train, test, k, threshold, n_items),
    }

    for name, metrics in results.items():
        write_metrics(name, metrics, path=f"{config['paths']['reports_dir']}/metrics.json")

    return results


if __name__ == "__main__":
    results = run_baselines()
    print(f"{'model':<15}{'rmse':>8}{'precision@k':>14}{'recall@k':>12}{'ndcg@k':>10}{'coverage':>10}")
    for name, metrics in results.items():
        print(
            f"{name:<15}{metrics['rmse']:>8.3f}{metrics['precision_at_k']:>14.3f}"
            f"{metrics['recall_at_k']:>12.3f}{metrics['ndcg_at_k']:>10.3f}{metrics['catalogue_coverage']:>10.1%}"
        )
