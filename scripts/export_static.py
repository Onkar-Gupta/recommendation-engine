"""Export the API's answers for every user and film as a static website, so the demo can
be hosted for free with no server (for example a Hugging Face static Space).

It calls the real API through a test client, so the saved answers are exactly what the
live server would return. Output: <out>/index.html, <out>/data.json, <out>/static/genres/.

Run with: python scripts/export_static.py --out site
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from rec_engine.data_loader import load_config
from rec_engine.serve import create_app

UNKNOWN_USER = 0  # any id the model has never seen returns the popular list
REPO_API_DOCS = "https://github.com/Onkar-Gupta/recommendation-engine#api"


def export(config_path, out_dir):
    config = load_config(config_path)
    mappings = json.loads((Path(config["paths"]["models_dir"]) / "id_mappings.json").read_text(encoding="utf-8"))
    client = TestClient(create_app(config_path))

    items = {}

    def remember(entries):
        for e in entries:
            items[str(e["item_id"])] = {"title": e["title"], "genres": e["genres"]}
        return [e["item_id"] for e in entries]

    data = {"recommend": {}, "history": {}, "similar": {}}
    for user in mappings["user2idx"]:
        data["recommend"][user] = remember(client.get(f"/recommend/{user}?k=10").json()["recommendations"])
        history = client.get(f"/history/{user}?k=5").json()["history"]
        remember(history)
        data["history"][user] = [[h["item_id"], h["score"]] for h in history]
    for item in mappings["item2idx"]:
        data["similar"][item] = remember(client.get(f"/similar/{item}?k=10").json()["similar"])
    data["popular"] = remember(client.get(f"/recommend/{UNKNOWN_USER}?k=10").json()["recommendations"])
    data["items"] = items

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    shutil.copytree(ROOT / "rec_engine/static/genres", out / "static/genres")

    page = (ROOT / "rec_engine/static/index.html").read_text(encoding="utf-8")
    page = page.replace('<link rel="icon"', '<meta name="static-data" content="data.json">\n<link rel="icon"', 1)
    page = page.replace('<a href="/docs">API docs</a>', f'<a href="{REPO_API_DOCS}">API docs</a>', 1)
    (out / "index.html").write_text(page, encoding="utf-8")
    return len(data["recommend"]), len(data["similar"]), (out / "data.json").stat().st_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a static copy of the demo.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="site")
    args = parser.parse_args()
    users, films, size = export(args.config, args.out)
    print(f"Exported {users} users and {films} films to {args.out}/ (data.json is {size / 1e6:.2f} MB)")
