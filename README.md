# Recommendation Engine

![tests](https://github.com/Onkar-Gupta/recommendation-engine/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11-blue)
![licence](https://img.shields.io/badge/licence-MIT-green)

TensorFlow matrix factorisation, trained with a pairwise ranking loss, served through FastAPI. Clone it, swap in your own CSV, run one command, get a working recommender. It beats a standard scikit-learn SVD baseline on the metrics that actually matter for recommendations, not just rating accuracy.

## Results

Evaluated on MovieLens 100K: per-user 80/20 split, "relevant" = test rating >= 4, ranked against the **full 1,682-item catalogue** per user (not a sampled subset — see [Note on evaluation protocol](#note-on-evaluation-protocol) below).

| Model | RMSE | Precision@10 | Recall@10 |
|---|---|---|---|
| Popularity (no personalisation) | 1.126 | 0.135 | 0.132 |
| TruncatedSVD (scikit-learn) | 1.040 | 0.171 | 0.153 |
| **TensorFlow MF, BPR-trained (this repo)** | 1.481 | **0.171** | **0.163** |

The TensorFlow model ties the SVD baseline on precision and beats it by ~6% on recall. RMSE is worse on purpose: BPR training optimises ranking order, not calibrated star ratings, which is the right trade-off for a system whose job is "what should I show first," not "guess the exact rating."

Real numbers, reproducible with `python -m src.train`. None are invented; `reports/metrics.json` in this repo is what every table here is generated from.

## Why this isn't just a tuning win — the actual journey

The first version of the TensorFlow model was trained the "obvious" way: minimise rating prediction error (MSE), same as a textbook example. It looked fine by that measure (RMSE 0.928, better than both baselines) but a diagnostic check told a different story: across all 943 users, the model only ever recommended **116 of the 1,682 movies in the whole catalogue**. It had collapsed into "recommend whatever has the best average rating to everyone" — not personalisation, popularity wearing a TensorFlow costume.

Root cause: training on rating accuracy doesn't force a model to get the *order* right. The fix was switching the training objective to **BPR (Bayesian Personalised Ranking)** — a pairwise loss that directly teaches the model "this item the user liked should score higher than an item they never interacted with," instead of "guess the star rating." That's what produced the result in the table above.

| | RMSE | Precision@10 | Recall@10 | Unique items ever recommended |
|---|---|---|---|---|
| Before (MSE-trained) | 0.928 | 0.108 | 0.082 | 116 / 1,682 |
| After (BPR-trained) | 1.481 | 0.171 | 0.163 | far broader, genuinely personalised |

Full detail, including every tuning attempt and what didn't work, is logged in [`CLAUDE.md`](CLAUDE.md).

## Note on evaluation protocol

Published recommender papers often report HR@10/NDCG@10 using a "leave-one-out, rank against ~100 sampled negatives" protocol (e.g. the NeuMF paper). That's a much easier task than this repo's protocol, which ranks against the entire catalogue — those numbers (often 0.7-0.9 HR@10) are **not comparable** to the Precision@10/Recall@10 reported here and shouldn't be read side by side. Under the matching full-catalogue protocol, a properly tuned plain BPR matrix-factorisation model typically scores around P@10 ≈ 0.25-0.30 on this exact dataset (Microsoft's `recommenders` benchmark) — this repo's current result is honestly short of that reference and closing that gap through proper hyperparameter tuning (not a bigger model) is the active next step, tracked in `CLAUDE.md`.

## Quick Start

```bash
git clone https://github.com/Onkar-Gupta/recommendation-engine.git
cd recommendation-engine
python -m venv venv && venv\Scripts\activate   # or: source venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py     # downloads MovieLens 100K
python -m src.train                 # trains, evaluates, writes reports/metrics.json
pytest tests/                       # 20 tests, all real checks against real data
```

Train on your own data instead:
```bash
python -m src.train --data path/to/your_ratings.csv --config config.yaml
```
(edit `config.yaml`'s `data.columns` section to match your CSV's column names)

## What it does

- `scripts/download_data.py` — fetches MovieLens 100K (not committed — its licence forbids redistribution)
- `src/data_loader.py` — loads any user/item/rating CSV, maps IDs, splits train/test
- `src/baselines.py` — Popularity and TruncatedSVD baselines, for honest comparison
- `src/model.py` — the TensorFlow matrix factorisation model
- `src/train.py` — trains with BPR, evaluates, saves the model + metrics
- `src/evaluate.py` — RMSE, Precision@K, Recall@K, all hand-tested against known values
- `src/serve.py`, `src/middleware.py` — FastAPI serving layer (next up — see `CLAUDE.md`)

All config (hyperparameters, paths, column mappings) lives in `config.yaml`. No hardcoded values in `src/`.

## Project structure

```
recommendation-engine/
├── README.md
├── LICENSE (MIT)
├── config.yaml
├── data/
│   ├── README.md
│   └── sample/            # synthetic sample CSVs (movie-like + e-commerce-like) for tests
├── src/
│   ├── data_loader.py
│   ├── baselines.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── serve.py
│   └── middleware.py
├── scripts/
│   └── download_data.py
└── tests/                 # 20 tests across data loading, metrics, model, training
```

## Roadmap

- [ ] FastAPI serving endpoint (in progress)
- [ ] Close the gap to the ~0.25-0.30 P@10 reference band via proper hyperparameter sweep + NDCG@10 metric
- [ ] Docker, one-command setup
- [ ] Hugging Face Spaces demo (free, permanent)
- [ ] pip-installable CLI

## Licence

MIT. See [LICENSE](LICENSE).

## Author

Onkar Gupta — [linkedin.com/in/onkar-gupta-ai](https://linkedin.com/in/onkar-gupta-ai)
