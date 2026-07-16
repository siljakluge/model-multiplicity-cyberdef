from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _entropy(votes: np.ndarray) -> float:
    _, counts = np.unique(votes, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _select_rashomon_models(metrics: pd.DataFrame, tolerance: float, metric: str) -> tuple[set[str], float]:
    if metric not in metrics.columns:
        raise ValueError(f"Unknown Rashomon metric {metric!r}; available columns: {sorted(metrics.columns)}")
    best = float(metrics[metric].max())
    # Rashomon set: all models within a small absolute tolerance of the best
    # observed score. Downstream disagreement is computed only inside this set.
    selected = set(metrics.loc[metrics[metric] >= best - tolerance, "model_id"])
    return selected, best


def compute_disagreement(
    run_dir: str | Path,
    rashomon_tolerance: float = 0.015,
    rashomon_metric: str = "accuracy",
) -> None:
    run_path = Path(run_dir)
    predictions = pd.read_csv(run_path / "test_predictions.csv")
    metrics = pd.read_csv(run_path / "metrics.csv")
    rashomon_models, best_score = _select_rashomon_models(metrics, rashomon_tolerance, rashomon_metric)
    predictions = predictions[predictions["model_id"].isin(rashomon_models)].copy()
    # Pivot to one row per test sample and one column per selected model; this
    # makes pairwise model disagreement and per-sample vote statistics direct.
    pivot = predictions.pivot(index="sample_id", columns="model_id", values="y_pred")

    rows = []
    for model_a, model_b in combinations(pivot.columns, 2):
        disagree = pivot[model_a].to_numpy() != pivot[model_b].to_numpy()
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "disagreement_rate": float(disagree.mean()),
                "agreement_rate": float(1.0 - disagree.mean()),
                "n_samples": int(len(disagree)),
            }
        )

    sample_rows = []
    y_true = predictions.drop_duplicates("sample_id").set_index("sample_id")["y_true"]
    for sample_id, row in pivot.iterrows():
        votes = row.to_numpy()
        values, counts = np.unique(votes, return_counts=True)
        majority_idx = int(np.argmax(counts))
        majority_label = int(values[majority_idx])
        majority_fraction = float(counts[majority_idx] / counts.sum())
        # Conflict ratio is zero when all Rashomon models agree and grows as the
        # majority vote becomes less dominant.
        conflict_ratio = float(1.0 - majority_fraction)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "y_true": int(y_true.loc[sample_id]),
                "majority_pred": majority_label,
                "vote_entropy": _entropy(votes),
                "unique_predictions": int(len(values)),
                "majority_fraction": majority_fraction,
                "conflict_ratio": conflict_ratio,
                "is_conflict": bool(conflict_ratio > 0),
            }
        )

    pd.DataFrame(rows).to_csv(run_path / "disagreement_summary.csv", index=False)
    sample_frame = pd.DataFrame(sample_rows)
    sample_frame.to_csv(run_path / "sample_disagreement.csv", index=False)

    disagreement_values = pd.DataFrame(rows)["disagreement_rate"] if rows else pd.Series(dtype=float)
    ambiguity = float(sample_frame["is_conflict"].mean()) if len(sample_frame) else 0.0
    summary = {
        "rashomon_metric": rashomon_metric,
        "rashomon_tolerance": rashomon_tolerance,
        "best_score": best_score,
        "n_models_total": int(metrics["model_id"].nunique()),
        "n_models_rashomon": int(len(rashomon_models)),
        "ambiguity": ambiguity,
        "mean_conflict_ratio": float(sample_frame["conflict_ratio"].mean()) if len(sample_frame) else 0.0,
        "max_conflict_ratio": float(sample_frame["conflict_ratio"].max()) if len(sample_frame) else 0.0,
        "mean_pairwise_disagreement": float(disagreement_values.mean()) if len(disagreement_values) else 0.0,
        "max_pairwise_disagreement": float(disagreement_values.max()) if len(disagreement_values) else 0.0,
    }
    pd.DataFrame([summary]).to_csv(run_path / "multiplicity_summary.csv", index=False)
