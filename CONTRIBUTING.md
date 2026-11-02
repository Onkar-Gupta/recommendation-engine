# Contributing

Thanks for taking a look. Small fixes, bug reports and new ideas are all welcome.

## Run it locally

```bash
python -m venv venv && venv\Scripts\activate    # or: source venv/bin/activate
pip install -r requirements.txt
pip install -e .                         # optional: gives you the rec-engine command
python -m pytest tests/
python -m ruff check .
```

Both commands must pass before you open a pull request. The tests use the small sample CSVs in `data/sample/`, so you do not need to download MovieLens to run them.

## Code style

- Lint with ruff: run `ruff check .` and fix what it reports.
- No hardcoded paths or hyperparameters in `rec_engine/`. They belong in `config.yaml`.
- Keep changes small and focused. One idea per pull request.
- Add or update a test for any behaviour you change.

## Commit messages

Conventional Commits, in the imperative, under 72 characters:

```
feat: add NDCG to the sweep report
fix: ignore blank titles path
docs: explain the evaluation split
test: cover empty ratings file
```

Prefixes: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`.

## Adding a new model

1. Put the model in `rec_engine/`. Follow `rec_engine/baselines.py`: a class with `fit(train_df)` and `recommend(user_id, k, exclude)`.
2. Score it with `evaluate_recommendations` from `rec_engine/evaluate.py`, so it is compared on the same split and the same five metrics as everything else.
3. If it changes the config, edit `config.yaml` and copy it to `rec_engine/default_config.yaml` (a test fails if they differ).
4. Add its row to the results table in the README, using the numbers from `reports/metrics.json`. Only real numbers, never estimates.
5. Add tests next to the existing ones in `tests/`.

## Reporting a bug or suggesting a feature

Open an issue and pick the matching template. For a bug, the most useful things are the command you ran, what you expected, and what happened instead.
