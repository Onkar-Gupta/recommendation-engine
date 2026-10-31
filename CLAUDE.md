# Project memory: recommendation-engine

Read this file at the start of every session. Update the Progress log at the end of every task.

## Project brief

**Problem:** People browse platforms with thousands of items (movies, products, courses) and have no way to find what they'd actually like. Most recommendation tutorials online stop at a notebook that prints an RMSE score; they never show how to serve recommendations to a real user. Enterprise tools (AWS Personalize, Google Recommendations AI) are expensive and overkill for small to mid-size platforms.

**Solution:** A complete, plug-and-play recommendation system. Give it user-item interaction data (ratings, clicks, purchases), it learns preferences using SVD-style matrix factorisation (TensorFlow embedding layers), and serves personalised recommendations through a FastAPI endpoint. Clone it, swap in your own CSV, run one command, get working recommendations.

**Who it's for:** Developers building a platform with a catalogue who need recommendations without paying for enterprise tools. Students learning recommender systems who want working code, not just theory.

## Locked decisions

- **Stack:** Python 3.11, TensorFlow 2.x (CPU), scikit-learn, pandas, numpy, PyYAML, FastAPI, uvicorn, pytest, ruff, httpx (test client).
- **Dataset rule:** MovieLens 100K downloaded by script at run time. NEVER commit MovieLens files (licence forbids redistribution). Ship only small synthetic sample CSVs (`data/sample/`) for tests and demos.
- **Second sample dataset:** a synthetic e-commerce CSV, different ID ranges, to prove the engine works on any user-item-rating data shape.
- **Folder layout:** see repo structure below.
- **Metrics:** RMSE, Precision@10, Recall@10.
- **Random seed:** 42 everywhere (data split, model init).

## Architecture notes

- Config-driven: all paths, hyperparameters and dataset column names live in `config.yaml`. No hardcoded values, so a future customer can swap in their own config.
- Middleware pattern: `src/middleware.py` has placeholder `auth_middleware` and `usage_tracking_middleware` functions. The free version ships open (no auth, logs to file). The paid hosted tier plugs real auth and billing into these same slots later, no API rewrite.
- Model and data paths are relative to a configurable root, so multi-tenant model folders (one per paid customer) work later without restructuring.
- Do NOT build any paid features now.

## Repo structure

```
recommendation-engine/
├── README.md
├── LICENSE (MIT)
├── .gitignore
├── requirements.txt
├── config.yaml
├── data/
│   ├── README.md
│   └── sample/
│       ├── ratings_sample.csv
│       └── ecommerce_sample.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── serve.py
│   └── middleware.py
├── scripts/
│   └── download_data.py
├── tests/
│   └── __init__.py
└── notebooks/
```

## Dataset stats

| Dataset | Users | Items | Ratings | Sparsity |
|---------|-------|-------|---------|----------|
| MovieLens 100K | 943 | 1,682 | 100,000 | 0.9370 |
| ratings_sample.csv | 30 | 39 | 200 | 0.8291 |
| ecommerce_sample.csv | 25 | 49 | 200 | 0.8367 |

## Progress log

| Date | Prompt | What was done | Commit |
|------|--------|----------------|--------|
| 2026-10-31 | Prompt 1 | Scaffolded structure, venv (Python 3.11), confirmed tensorflow/sklearn/fastapi import, created and pushed github.com/Onkar-Gupta/recommendation-engine | `chore: scaffold project structure` |
| 2026-10-31 | Prompt 2 | Built data pipeline: download_data.py (MovieLens, certifi-backed SSL), data_loader.py (load_ratings, map_ids, split_train_test, load_item_titles, dataset_stats), 8 tests passing on both sample CSVs | `feat: add data download and loading pipeline`, `test: add data loader tests` |
| 2026-10-31 | Prompt 3 | Added evaluate.py (rmse, precision_at_k, recall_at_k, write_metrics) and baselines.py (Popularity, TruncatedSVD). Ran on MovieLens 100K: popularity rmse=1.126 P@10=0.135 R@10=0.132; truncated_svd rmse=1.040 P@10=0.171 R@10=0.153. 6 metric tests passing. No tuning done. | `feat: add evaluation metrics`, `feat: add popularity and TruncatedSVD baselines`, `test: add metric tests` |

## Next step

Prompt 4: the TensorFlow matrix factorisation model (`src/model.py`, `src/train.py`) — must beat the TruncatedSVD baseline above.
