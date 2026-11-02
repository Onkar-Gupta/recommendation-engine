"""Hyperparameter sweep over the BPR training config. Logs every config's
full metrics (precision/recall/ndcg/coverage) to reports/sweep.json, so the
final config is picked from evidence, not another ad hoc try. Coverage is
tracked deliberately: a config can look great on precision/recall while
recommending the same narrow handful of items to everyone (this happened in
this project, see CLAUDE.md) and precision/recall alone don't catch it.

Run with: python scripts/sweep.py
"""

import copy
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_config
from src.train import train

TMP_CONFIG_PATH = Path("reports/_sweep_tmp_config.yaml")
SWEEP_OUT_PATH = Path("reports/sweep.json")


def build_grid(baseline):
    """A coordinate search around the current baseline (vary one knob at a
    time from the committed config), not a full grid. 36+ full-grid runs
    would mean 36x the wait for a payoff a sweep this size doesn't need;
    this covers the same hyperparameters the research pass flagged as
    unswept, at a fraction of the runs.
    """
    grid = [{"label": "baseline", "overrides": {}}]

    for embedding_dim in [16, 64]:
        grid.append({"label": f"embedding_dim={embedding_dim}", "overrides": {"embedding_dim": embedding_dim}})

    for bpr_negatives in [8, 15]:
        grid.append({"label": f"bpr_negatives={bpr_negatives}", "overrides": {"bpr_negatives": bpr_negatives}})

    for bpr_learning_rate in [0.005, 0.02]:
        grid.append(
            {"label": f"bpr_learning_rate={bpr_learning_rate}", "overrides": {"bpr_learning_rate": bpr_learning_rate}}
        )

    for l2_reg in [1e-5, 1e-3]:
        grid.append({"label": f"l2_reg={l2_reg}", "overrides": {"l2_reg": l2_reg}})

    grid.append(
        {"label": "bpr_epochs=80,patience=10", "overrides": {"bpr_epochs": 80, "bpr_patience": 10}}
    )
    return grid


def run_sweep(base_config_path="config.yaml"):
    base_config = load_config(base_config_path)
    grid = build_grid(base_config["training"])

    results = []
    for trial in grid:
        config = copy.deepcopy(base_config)
        config["training"].update(trial["overrides"])

        TMP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        TMP_CONFIG_PATH.write_text(yaml.safe_dump(config), encoding="utf-8")

        start = time.time()
        metrics = train(config_path=str(TMP_CONFIG_PATH))
        wall_seconds = round(time.time() - start, 1)

        result = {"label": trial["label"], "overrides": trial["overrides"], "wall_seconds": wall_seconds, **metrics}
        results.append(result)
        print(
            f"{trial['label']:<32} P@10={metrics['precision_at_k']:.3f} R@10={metrics['recall_at_k']:.3f} "
            f"NDCG@10={metrics['ndcg_at_k']:.3f} coverage={metrics['catalogue_coverage']:.1%} "
            f"({wall_seconds}s)"
        )

    TMP_CONFIG_PATH.unlink(missing_ok=True)
    SWEEP_OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    run_sweep()
