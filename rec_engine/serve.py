"""FastAPI server for the trained recommender. Runs on NumPy and pandas only: no TensorFlow.

Usage:
    uvicorn rec_engine.serve:create_app --factory
"""

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rec_engine.baselines import PopularityBaseline
from rec_engine.data_loader import (
    load_config,
    load_item_genres,
    load_item_titles,
    load_ratings,
)
from rec_engine.evaluate import make_recommend_fn
from rec_engine.middleware import make_auth_middleware, make_usage_tracking_middleware
from rec_engine.weights import WEIGHTS_FILE, load_weights, unit_vectors
from rec_engine.weights import score_matrix as compute_scores


def create_app(config_path="config.yaml", config=None):
    """Build the app from a config file, or from an already-loaded settings dict (used by the CLI)."""
    config = config if config is not None else load_config(config_path)
    serving = config["serving"]
    models_dir = Path(config["paths"]["models_dir"])

    # Missing model files fail loudly here on purpose: no silent empty-model fallback.
    weights = load_weights(models_dir / WEIGHTS_FILE)
    mappings = json.loads((models_dir / "id_mappings.json").read_text(encoding="utf-8"))
    user2idx = {int(k): v for k, v in mappings["user2idx"].items()}
    item2idx = {int(k): v for k, v in mappings["item2idx"].items()}
    idx2item = {v: k for k, v in item2idx.items()}

    ratings = load_ratings(serving["ratings_path"], columns=config["data"]["columns"])
    seen = ratings.groupby("user_id")["item_id"].apply(set).to_dict()
    popularity = PopularityBaseline(config["baselines"]["popularity_min_rating"]).fit(ratings)
    support = ratings["item_id"].value_counts()
    low_support = {
        idx for item_id, idx in item2idx.items() if support.get(item_id, 0) < config["evaluation"]["min_item_support"]
    }
    titles = load_item_titles(serving["item_titles_path"])
    genres = load_item_genres(serving["item_titles_path"])  # same u.item file holds titles and genres
    liked = ratings.sort_values(["rating", "timestamp"], ascending=False).groupby("user_id")[["item_id", "rating"]].apply(
        lambda g: list(zip(g["item_id"], g["rating"]))
    )

    score_matrix = compute_scores(weights)
    recommend_fn = make_recommend_fn(score_matrix, user2idx, item2idx, idx2item, low_support)
    item_unit = unit_vectors(weights["item_vectors"])

    def to_item(item_id, score):
        return {"item_id": int(item_id), "title": titles.get(item_id), "genres": genres.get(item_id, []), "score": float(score)}

    app = FastAPI(title="Recommendation Engine")
    # add_middleware puts the LAST one added outermost, so usage tracking wraps auth and logs rejected requests too.
    app.middleware("http")(make_auth_middleware(config["auth"]))
    app.middleware("http")(make_usage_tracking_middleware(serving["usage_log_path"]))

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    def check_k(k):
        if not 1 <= k <= serving["max_k"]:
            raise HTTPException(422, f"k must be between 1 and {serving['max_k']}")

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": serving["model_version"], "dataset_name": serving["dataset_name"]}

    @app.get("/history/{user_id}")
    def history(user_id: int, k: int = Query(serving["default_k"])):
        check_k(k)
        # "score" is the user's own rating here. Unknown users simply have no history.
        return {"user_id": user_id, "history": [to_item(i, r) for i, r in liked.get(user_id, [])[:k]]}

    @app.get("/recommend/{user_id}")
    def recommend(user_id: int, request: Request, k: int = Query(serving["default_k"])):
        check_k(k)
        if user_id in user2idx:
            strategy = "personalized"
            ids = recommend_fn(user_id, k, seen.get(user_id, set()))
            scores = [score_matrix[user2idx[user_id], item2idx[i]] for i in ids]
        else:
            # Popularity has no per-item score, so report the dataset mean rating for every item.
            strategy = "popular"
            ids = popularity.recommend(user_id, k)
            scores = [popularity.global_mean_] * len(ids)
        return {
            "user_id": user_id,
            "strategy": strategy,
            "recommendations": [to_item(i, s) for i, s in zip(ids, scores)],
        }

    @app.get("/similar/{item_id}")
    def similar(item_id: int, k: int = Query(serving["default_k"])):
        check_k(k)
        if item_id not in item2idx:
            raise HTTPException(404, f"Unknown item_id {item_id}")
        query = item2idx[item_id]
        sims = item_unit @ item_unit[query]
        sims[query] = -np.inf
        if low_support:
            sims[list(low_support)] = -np.inf  # candidates only; the query item itself is still allowed
        top = np.argsort(-sims)[:k]
        return {"item_id": item_id, "similar": [to_item(idx2item[i], sims[i]) for i in top]}

    return app

