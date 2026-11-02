"""Download MovieLens 100K into the folder named in config.yaml. Run with: python scripts/download_data.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rec_engine.data_loader import load_config
from rec_engine.download import download

if __name__ == "__main__":
    download(load_config()["paths"]["raw_data_dir"])
