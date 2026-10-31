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

## Prompt 4 tuning tries (max 3, no further tuning after this)

| Try | embedding_dim | l2_reg | RMSE | P@10 | R@10 | Verdict |
|-----|---------------|--------|------|------|------|---------|
| 1 | 32 | 0.0001 | 0.928 | 0.108 | 0.082 | Best overall (kept, final config) |
| 2 | 32 | 0.001 | 0.986 | 0.097 | 0.078 | Worse on everything, stronger L2 hurt |
| 3 | 16 | 0.0001 | 0.924 | 0.104 | 0.080 | Marginally better RMSE, worse ranking than try 1 |

**Final comparison (popularity vs truncated_svd vs tf_mf), MovieLens 100K:**

| Model | RMSE | Precision@10 | Recall@10 | Train time |
|-------|------|---------------|-----------|------------|
| Popularity | 1.126 | 0.135 | 0.132 | - |
| TruncatedSVD | 1.040 | 0.171 | 0.153 | - |
| tf_mf | 0.928 | 0.108 | 0.082 | 3.8s |

tf_mf has the lowest rating error of the three but loses to TruncatedSVD (and even to the naive Popularity baseline) on ranking quality. Likely cause: plain dot-product MF gives sparse/under-trained items noisy scores that can still outrank well-supported items, which pure RMSE doesn't penalise. Worth investigating in a later pass (e.g. popularity-aware score blending), but not required by this prompt, which only asks to report the numbers honestly.

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
| 2026-10-31 | Prompt 4 | Added model.py (TF subclassed matrix factorisation: user/item embeddings + biases + global mean, dot product, L2 reg) and train.py (CLI, early stopping, saves model + ID mappings, writes metrics). 3 model tests passing. Tried 3 configs (see below), kept the best. **Result: tf_mf beats both baselines on RMSE but does not beat TruncatedSVD on Precision@10/Recall@10 in any of the 3 tries** — reporting honestly, not tuning further. Also fixed a real perf bug: evaluating via recommend() per user meant ~900 separate model.predict() calls; batched into one predict() over the full user x item matrix (15min -> 4s). | `feat: add TensorFlow matrix factorisation model`, `feat: add training script with CLI`, `test: add model tests` |

## Next step

Prompt 5: FastAPI serving with the middleware pattern (`src/serve.py`).
