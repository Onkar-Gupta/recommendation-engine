"""Download MovieLens 100K from GroupLens into data/raw/ (git-ignored). Skips
if already present. Run with: python scripts/download_data.py
"""

import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import certifi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_config

# Windows python.org builds don't always pick up the OS cert store; point
# urllib at certifi's bundle so the GroupLens download verifies cleanly.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download(raw_dir=None):
    if raw_dir is None:
        raw_dir = load_config()["paths"]["raw_data_dir"]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    target = raw_dir / "ml-100k" / "u.data"
    if target.exists():
        print(f"Already downloaded: {target}")
        return target

    zip_path = raw_dir / "ml-100k.zip"
    print(f"Downloading MovieLens 100K from {URL} ...")
    with urllib.request.urlopen(URL, context=_SSL_CONTEXT) as response, open(zip_path, "wb") as f:
        f.write(response.read())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()

    print(f"Done. Files are in {target.parent}")
    return target


if __name__ == "__main__":
    download()
