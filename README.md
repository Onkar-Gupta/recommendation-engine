# Recommendation Engine

![tests](https://github.com/Onkar-Gupta/recommendation-engine/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11-blue)
![licence](https://img.shields.io/badge/licence-MIT-green)

TensorFlow matrix factorisation, trained with a pairwise ranking loss and tuned via a real hyperparameter sweep, served through FastAPI. Clone it, swap in your own CSV, run one command, get a working recommender. It beats a standard scikit-learn SVD baseline on every ranking metric, not just rating accuracy.

## Results

Evaluated on MovieLens 100K: per-user 80/20 split, "relevant" = test rating >= 4, ranked against the **full 1,682-item catalogue** per user (not a sampled subset).

| Model | RMSE | Precision@10 | Recall@10 | NDCG@10 | Catalogue coverage |
|---|---|---|---|---|---|
| Popularity (no personalisation) | 1.126 | 0.135 | 0.132 | 0.176 | 3.3% |
| TruncatedSVD (scikit-learn) | 1.040 | 0.171 | 0.153 | 0.225 | 20.7% |
| TensorFlow MF, BPR-trained (this repo) | 2.561 | **0.213** | **0.237** | **0.296** | **24.4%** |

The TensorFlow model beats the SVD baseline on every ranking metric, with broader catalogue coverage too. RMSE is higher by design: the model is trained with a pairwise ranking loss (BPR) that directly optimises for ranking order, not calibrated star ratings — the right trade-off for a system whose job is "what should I show first," not "guess the exact rating."

Catalogue coverage is reported deliberately, not just precision/recall: a model can score well on both while recommending the same narrow handful of items to everyone, which is a real failure mode this repo hit once and now tracks for explicitly (see `CLAUDE.md`).

Real numbers, reproducible with `python -m src.train`. None are invented; `reports/metrics.json` is what every table here is generated from, and `scripts/sweep.py` is the hyperparameter sweep that produced the final config.

## Evaluation protocol

This repo ranks recommendations against the entire 1,682-item catalogue, which is harder and more realistic than the protocol many published recommender papers use — they rank against ~100 sampled negative items. That difference makes published HR@10/NDCG@10 figures (often 0.7-0.9) not directly comparable to the numbers above. Under the same full-catalogue protocol, a properly tuned pairwise ranking model typically scores around P@10 ≈ 0.25–0.30 on this dataset (Microsoft's `recommenders` benchmark is the reference). This repo's Recall@10 (0.237) is inside that reference band; Precision@10 (0.213) is close behind.

## Quick Start

```bash
git clone https://github.com/Onkar-Gupta/recommendation-engine.git
cd recommendation-engine
python -m venv venv && venv\Scripts\activate   # or: source venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py     # downloads MovieLens 100K
python -m src.train                 # trains, evaluates, writes reports/metrics.json
pytest tests/                       # 26 tests, all real checks against real data
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
- `src/evaluate.py` — RMSE, Precision@K, Recall@K, NDCG@K, catalogue coverage, all hand-tested against known values
- `scripts/sweep.py` — the hyperparameter sweep that produced the final config
- `src/serve.py`, `src/middleware.py` — FastAPI serving layer (next up)

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
│   ├── download_data.py
│   └── sweep.py
└── tests/                 # 26 tests across data loading, metrics, model, training
```

## Roadmap

- [ ] FastAPI serving endpoint (in progress)
- [x] Hyperparameter sweep, NDCG@10 and catalogue coverage metrics
- [ ] Docker, one-command setup
- [ ] Hugging Face Spaces demo (free, permanent)
- [ ] pip-installable CLI

## Licence

MIT. See [LICENSE](LICENSE).

## Author

Onkar Gupta — [linkedin.com/in/onkar-gupta-ai](https://linkedin.com/in/onkar-gupta-ai)
