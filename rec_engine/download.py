"""Download MovieLens 100K from GroupLens into a data folder (git-ignored). Skips if already there."""

import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download(raw_dir="data/raw"):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    target = raw_dir / "ml-100k" / "u.data"
    if target.exists():
        print(f"Already downloaded: {target}")
        return target

    # Windows python.org builds don't always pick up the OS cert store; point
    # urllib at certifi's bundle so the GroupLens download verifies cleanly.
    context = ssl.create_default_context(cafile=certifi.where())
    zip_path = raw_dir / "ml-100k.zip"
    print(f"Downloading MovieLens 100K from {URL} ...")
    with urllib.request.urlopen(URL, context=context) as response, open(zip_path, "wb") as f:
        f.write(response.read())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()

    print(f"Done. Files are in {target.parent}")
    return target
