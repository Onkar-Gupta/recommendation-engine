"""Look up a poster link for each film from TMDB (themoviedb.org). Only the
link is saved (data/posters.json); no images are downloaded or committed.

Needs a free TMDB "API Read Access Token" in the TMDB_TOKEN environment
variable. It is never read from or written to a file in this repo.

Usage:
    python -m src.posters --config config.yaml
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from src.data_loader import load_config, load_item_titles

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
ARTICLES = "The|A|An|Il|La|Le|Les|Das|Der|Die|El|Los|Las"


def clean_title(raw):
    """'Close Shave, A (1995)' -> ('A Close Shave', '1995'). Drops alias
    brackets like '(The Postman)' and moves a trailing article to the front."""
    year = ""
    match = re.search(r"\((\d{4})\)\s*$", raw)
    if match:
        year, raw = match.group(1), raw[: match.start()]
    name = re.sub(r"\s*\([^)]*\)", "", raw).strip()
    moved = re.match(rf"^(.*), ({ARTICLES})$", name)
    if moved:
        name = f"{moved.group(2)} {moved.group(1)}"
    return name, year


def pick_poster(results, year):
    """First result that has a poster and a release year within one year of
    the MovieLens year (release dates differ by region). Without a year, the
    first result with a poster."""
    for movie in results:
        path = movie.get("poster_path")
        released = (movie.get("release_date") or "")[:4]
        if not path:
            continue
        if not year or (released.isdigit() and abs(int(released) - int(year)) <= 1):
            return IMAGE_BASE + path
    return None


def find_poster(name, year, token, get=None):
    query = urllib.parse.urlencode({"query": name, "include_adult": "false"})
    data = (get or _http_get)(f"{SEARCH_URL}?{query}", token)
    return pick_poster(data.get("results", []), year)


def _http_get(url, token):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)
        except OSError:
            time.sleep(1 + attempt)
    return {}


def fetch_all(config_path="config.yaml", delay=0.05):
    token = os.environ.get("TMDB_TOKEN")
    if not token:
        raise SystemExit("Set the TMDB_TOKEN environment variable to your TMDB API Read Access Token first.")
    config = load_config(config_path)
    titles = load_item_titles(config["serving"]["item_titles_path"])
    out_path = Path(config["serving"]["posters_path"])
    posters = {}
    for n, (item_id, raw) in enumerate(titles.items(), 1):
        name, year = clean_title(raw)
        url = find_poster(name, year, token)
        if url:
            posters[item_id] = url
        if n % 100 == 0:
            print(f"{n}/{len(titles)} done, {len(posters)} posters found", flush=True)
        time.sleep(delay)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(posters), encoding="utf-8")
    print(f"Saved {len(posters)} of {len(titles)} to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch poster links from TMDB.")
    parser.add_argument("--config", default="config.yaml")
    fetch_all(parser.parse_args().config)
