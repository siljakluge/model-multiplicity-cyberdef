from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _difference_signature(left: pd.Series, right: pd.Series) -> tuple[str, ...]:
    if (
        "comparison_block" in left.index
        and "comparison_block" in right.index
        and left["comparison_block"] == right["comparison_block"]
        and left["comparison_block"] not in {"", "baseline", "grid"}
    ):
        return (str(left["comparison_block"]),)
    factors = []
    for column in [
        "model_family",
        "seed",
        "train_seed",
        "data_seed",
        "subset_fraction",
        "subset_strategy",
        "architecture_id",
        "hidden_dims_label",
        "activation",
    ]:
        if column in left.index and column in right.index and left[column] != right[column]:
            if column == "hidden_dims_label":
                continue
            factors.append(column)
    return tuple(factors)


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


def _rashomon_source_summary(metrics: pd.DataFrame, rashomon_models: set[str]) -> pd.DataFrame:
    source_column = "comparison_block" if "comparison_block" in metrics.columns else "model_family"
    source_labels = metrics[source_column].fillna("unknown").astype(str)
    summary = (
        metrics.assign(
            multiplicity_source=source_labels,
            in_rashomon=metrics["model_id"].isin(rashomon_models),
        )
        .groupby("multiplicity_source", as_index=False)
        .agg(
            n_models_total=("model_id", "nunique"),
            n_models_rashomon=("in_rashomon", "sum"),
        )
        .sort_values(["n_models_rashomon", "n_models_total", "multiplicity_source"], ascending=[False, False, True])
    )
    summary["n_models_rashomon"] = summary["n_models_rashomon"].astype(int)
    summary["rashomon_fraction"] = np.where(
        summary["n_models_total"] > 0,
        summary["n_models_rashomon"] / summary["n_models_total"],
        0.0,
    )
    return summary


def compute_disagreement(
    run_dir: str | Path,
    rashomon_tolerance: float = 0.015,
    rashomon_metric: str = "accuracy",
) -> None:
    run_path = Path(run_dir)
    predictions = pd.read_csv(
        run_path / "test_predictions.csv",
        dtype={
            "model_id": "string",
            "architecture_id": "string",
            "hidden_dims_label": "string",
            "activation": "string",
            "comparison_block": "string",
            "model_family": "string",
            "subset_strategy": "string",
        },
        low_memory=False,
    )
    metrics = pd.read_csv(run_path / "metrics.csv")
    rashomon_models, best_score = _select_rashomon_models(metrics, rashomon_tolerance, rashomon_metric)
    predictions = predictions[predictions["model_id"].isin(rashomon_models)].copy()
    metric_lookup = metrics.set_index("model_id")
    # Pivot to one row per test sample and one column per selected model; this
    # makes pairwise model disagreement and per-sample vote statistics direct.
    pivot = predictions.pivot(index="sample_id", columns="model_id", values="y_pred")

    rows = []
    for model_a, model_b in combinations(pivot.columns, 2):
        disagree = pivot[model_a].to_numpy() != pivot[model_b].to_numpy()
        left = metric_lookup.loc[model_a]
        right = metric_lookup.loc[model_b]
        differing_factors = _difference_signature(left, right)
        source_factor = differing_factors[0] if len(differing_factors) == 1 else "combined"
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "seed_a": int(left["seed"]),
                "seed_b": int(right["seed"]),
                "origin_seed_a": int(left.get("origin_seed", left["seed"])),
                "origin_seed_b": int(right.get("origin_seed", right["seed"])),
                "train_seed_a": int(left.get("train_seed", left["seed"])),
                "train_seed_b": int(right.get("train_seed", right["seed"])),
                "data_seed_a": int(left.get("data_seed", left["seed"])),
                "data_seed_b": int(right.get("data_seed", right["seed"])),
                "model_family_a": left.get("model_family", "mlp"),
                "model_family_b": right.get("model_family", "mlp"),
                "comparison_block_a": left.get("comparison_block", ""),
                "comparison_block_b": right.get("comparison_block", ""),
                "subset_strategy_a": left.get("subset_strategy", "stratified"),
                "subset_strategy_b": right.get("subset_strategy", "stratified"),
                "subset_fraction_a": float(left["subset_fraction"]),
                "subset_fraction_b": float(right["subset_fraction"]),
                "architecture_id_a": left.get("architecture_id", ""),
                "architecture_id_b": right.get("architecture_id", ""),
                "hidden_dims_label_a": left.get("hidden_dims_label", ""),
                "hidden_dims_label_b": right.get("hidden_dims_label", ""),
                "activation_a": left.get("activation", ""),
                "activation_b": right.get("activation", ""),
                "source_factor": source_factor,
                "n_differing_factors": int(len(differing_factors)),
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
    pairwise_frame = pd.DataFrame(rows)
    if len(pairwise_frame):
        factor_summary = (
            pairwise_frame.groupby("source_factor")
            .agg(
                n_pairs=("model_a", "count"),
                mean_disagreement_rate=("disagreement_rate", "mean"),
                max_disagreement_rate=("disagreement_rate", "max"),
                mean_agreement_rate=("agreement_rate", "mean"),
            )
            .reset_index()
            .sort_values(["source_factor"])
        )
    else:
        factor_summary = pd.DataFrame(
            columns=[
                "source_factor",
                "n_pairs",
                "mean_disagreement_rate",
                "max_disagreement_rate",
                "mean_agreement_rate",
            ]
        )
    factor_summary.to_csv(run_path / "disagreement_by_factor.csv", index=False)
    sample_frame = pd.DataFrame(sample_rows)
    sample_frame.to_csv(run_path / "sample_disagreement.csv", index=False)

    disagreement_values = pairwise_frame["disagreement_rate"] if len(pairwise_frame) else pd.Series(dtype=float)
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

    rashomon_by_source = _rashomon_source_summary(metrics, rashomon_models)
    rashomon_by_source.insert(0, "rashomon_metric", rashomon_metric)
    rashomon_by_source.insert(1, "rashomon_tolerance", rashomon_tolerance)
    rashomon_by_source.insert(2, "best_score", best_score)
    rashomon_by_source.to_csv(run_path / "rashomon_by_source.csv", index=False)
