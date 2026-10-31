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

## Prompt 4: the MSE attempt, then the real fix (BPR)

**Tuning tries with the original plain-MSE training (max 3, per the plan's rule):**

| Try | embedding_dim | l2_reg | RMSE | P@10 | R@10 |
|-----|---------------|--------|------|------|------|
| 1 | 32 | 0.0001 | 0.928 | 0.108 | 0.082 |
| 2 | 32 | 0.001 | 0.986 | 0.097 | 0.078 |
| 3 | 16 | 0.0001 | 0.924 | 0.104 | 0.080 |

None beat TruncatedSVD (P@10=0.171, R@10=0.153) on ranking. Diagnosed why: checked how many unique items ever appeared in anyone's top-10 across all 943 users — only 116 out of 1,682. The model had collapsed into "recommend whatever has the best average rating to everyone", barely personalising at all. Root cause: training on plain MSE (predict the right star rating) doesn't force the model to get the *order* right, and the bias terms dominated the weak dot-product signal. Ruled out a different hypothesis first (noisy cold-start items polluting rankings) by testing a min-item-support filter — zero effect, so that wasn't it.

**The fix: switched the training objective to BPR (Bayesian Personalised Ranking)**, a pairwise ranking loss — for each item a user rated >= 4, sample an item they never rated, and train the model so the liked item scores higher than the random one (`-log(sigmoid(pos_score - neg_score))`). This directly optimises for ranking instead of rating accuracy. Implemented in `src/train.py` as `generate_bpr_triplets()` + a custom `tf.GradientTape` training loop (`train_bpr()`), early-stopping on validation Precision@10 (the metric that actually matters) instead of loss. Config-driven via `training.objective: bpr` (kept `mse` as a documented, selectable fallback).

**Final comparison (popularity vs truncated_svd vs tf_mf), MovieLens 100K:**

| Model | RMSE | Precision@10 | Recall@10 | Train time |
|-------|------|---------------|-----------|------------|
| Popularity | 1.126 | 0.135 | 0.132 | - |
| TruncatedSVD | 1.040 | 0.171 | 0.153 | - |
| tf_mf (bpr) | 1.481 | **0.171** | **0.163** | 220s |

tf_mf now beats TruncatedSVD on Recall@10 (+6%) and ties it on Precision@10 — the ranking-loss model genuinely recommends better than both baselines. RMSE got worse (1.48 vs 0.93 under plain MSE), which is expected and correct: BPR only learns relative order, not calibrated rating values, and ranking quality is what a recommendation engine is actually for. This is the honest, final, reported result — not cherry-picked, reproducible by re-running `python -m src.train`.

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
| 2026-10-31 | Post-Prompt-4 fix | Diagnosed why tf_mf lost to TruncatedSVD: only 116/1682 items ever appeared in anyone's top-10 (near-zero personalisation, model had collapsed to recommending by popularity). Switched training objective to BPR (pairwise ranking loss). **New result: tf_mf ties TruncatedSVD on P@10 (0.171) and beats it on R@10 by ~6% (0.163 vs 0.153)**, at the cost of worse RMSE (1.481) — correct trade-off since BPR optimises ranking, not rating accuracy. | `feat: switch tf_mf training to pairwise ranking loss (BPR)`, `test: add BPR triplet generation tests` |
| 2026-10-31 | Research pass (subagent) | Researched real published benchmarks (Microsoft `recommenders` repo, same full-catalogue-ranking protocol family). Finding: a properly-tuned plain BPR-MF on this exact dataset typically reaches P@10≈0.25-0.30, R@10≈0.20-0.25 — we're ~35-40% short, but it's an undertrained-config gap (current BPR hyperparameters were first-try values, never swept), not an architecture problem. Explicitly ruled out chasing the 0.35-0.47 band (NCF/LightGCN/BiVAE) — different architecture class, disproportionate effort, would dilute this repo's actual story. Plan: (A) add NDCG@10 metric, real hyperparameter sweep, multi-seed reporting, optionally a second `implicit`-ALS model reusing the same eval harness; (B) make the engineering visible — README was almost empty, now rewritten (see below). | - |
| 2026-10-31 | README rewrite + CI | Rewrote README.md: results table, the "journey" story (MSE→diagnosis→BPR fix) with the 116/1682 stat front and centre, an explicit evaluation-protocol honesty note (full-catalogue ranking vs sampled-negative, not comparable to most paper-quoted HR@10 numbers), real Quick Start, roadmap. Added `.github/workflows/tests.yml` (pytest on push) for an honest CI badge. | `docs: rewrite README with results, journey story, and honest benchmark context`, `ci: add pytest workflow` |

## Next step

Metric phase: add NDCG@10 to `src/evaluate.py`, run a real hyperparameter sweep on the BPR config (embedding_dim, bpr_negatives, bpr_learning_rate, bpr_epochs/patience, l2_reg), report multiple seeds. Target: close the gap toward P@10≈0.25-0.30. Then Prompt 5: FastAPI serving with the middleware pattern (`src/serve.py`).
