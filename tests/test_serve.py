"""Tests for the FastAPI server, using a tiny model trained on the sample CSV."""

import json

import pytest
import yaml
from fastapi.testclient import TestClient

from src.data_loader import load_config
from src.serve import create_app
from src.train import train

SAMPLE = "data/sample/ratings_sample.csv"


def _write_config(tmp_path, auth_enabled):
    config = load_config("config.yaml")
    config["paths"]["models_dir"] = str(tmp_path / "models")
    config["paths"]["reports_dir"] = str(tmp_path / "reports")
    config["training"].update({"embedding_dim": 8, "bpr_epochs": 2, "batch_size": 32})
    config["evaluation"]["min_item_support"] = 1
    config["serving"].update(
        {
            "ratings_path": SAMPLE,
            "item_titles_path": str(tmp_path / "no_titles.item"),
            "usage_log_path": str(tmp_path / "usage.jsonl"),
        }
    )
    config["auth"]["enabled"] = auth_enabled
    path = tmp_path / f"config_auth_{auth_enabled}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("serve")


@pytest.fixture(scope="module")
def client(tmp_dir):
    config_path = _write_config(tmp_dir, auth_enabled=False)
    train(str(config_path), data_path=SAMPLE)
    return TestClient(create_app(str(config_path)))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_root_serves_the_web_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Recommendation Engine" in response.text


def test_history_lists_highest_rated_first_and_unknown_user_is_empty(client):
    history = client.get("/history/1?k=5").json()["history"]
    assert 0 < len(history) <= 5
    assert history[0]["score"] >= history[-1]["score"]
    assert client.get("/history/999999").json()["history"] == []


def test_known_user_gets_personalised_recommendations(client):
    body = client.get("/recommend/1?k=5").json()
    assert body["strategy"] == "personalized"
    assert 0 < len(body["recommendations"]) <= 5


def test_unknown_user_falls_back_to_popularity(client):
    body = client.get("/recommend/999999?k=5").json()
    assert body["strategy"] == "popular"
    assert len(body["recommendations"]) > 0


def test_k_above_max_is_rejected(client):
    assert client.get("/recommend/1?k=1000").status_code == 422


def test_similar_excludes_query_item_and_404s_on_unknown(client):
    item_id = client.get("/recommend/1?k=1").json()["recommendations"][0]["item_id"]
    similar = client.get(f"/similar/{item_id}?k=5").json()["similar"]
    assert item_id not in [s["item_id"] for s in similar]
    assert client.get("/similar/999999").status_code == 404


def test_usage_is_logged(client, tmp_dir):
    client.get("/recommend/1")
    lines = (tmp_dir / "usage.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["user_id"] == "1"


def test_auth_requires_header_when_enabled(client, tmp_dir):
    secured = TestClient(create_app(str(_write_config(tmp_dir, auth_enabled=True))))
    assert secured.get("/recommend/1").status_code == 401
    assert secured.get("/recommend/1", headers={"X-API-Key": "anything"}).status_code == 200
    assert secured.get("/health").status_code == 200
