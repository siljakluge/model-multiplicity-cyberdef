from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _dataset_label(run_path: Path) -> str:
    config_path = run_path / "config.resolved.json"
    if not config_path.exists():
        return run_path.parent.name
    config = json.loads(config_path.read_text())
    dataset = config.get("dataset", {})
    name = str(dataset.get("name", run_path.parent.name))
    problem = dataset.get("problem") or dataset.get("task")
    return f"{name}:{problem}" if problem else name


def _one_row(path: Path) -> dict:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(frame)}.")
    return frame.iloc[0].to_dict()


def _bootstrap_ci(values: np.ndarray, n_resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def aggregate_runs(
    run_dirs: list[str | Path], out_dir: str | Path, n_bootstrap: int = 10_000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate repeated origins; never treats models, pairs, or samples as replicas."""
    records: list[dict] = []
    for raw_run_dir in run_dirs:
        run_path = Path(raw_run_dir)
        multiplicity = _one_row(run_path / "multiplicity_summary.csv")
        explanation = _one_row(run_path / "explanation_multiplicity_summary.csv")
        variability = _one_row(run_path / "shap_variability_summary.csv")
        membership = pd.read_csv(run_path / "rashomon_models.csv")
        if membership.empty:
            raise ValueError(f"Empty Rashomon membership in {run_path}.")
        if membership["rashomon_metric"].nunique() != 1 or membership["rashomon_metric"].iloc[0] != "val_accuracy":
            raise ValueError(f"{run_path} was not selected with val_accuracy. Re-run disagreement before final aggregation.")
        records.append(
            {
                "run_dir": str(run_path),
                "dataset": _dataset_label(run_path),
                "origin_seed": int(membership.get("origin_seed", pd.Series([0])).iloc[0]),
                "mean_rashomon_test_accuracy": float(membership["accuracy"].mean()),
                "mean_rashomon_test_macro_f1": float(membership["macro_f1"].mean()),
                **{key: value for key, value in multiplicity.items() if key not in {"rashomon_metric", "rashomon_tolerance"}},
                **{f"explanation_{key}": value for key, value in explanation.items()},
                **{f"variability_{key}": value for key, value in variability.items()},
            }
        )

    origin_table = pd.DataFrame(records).sort_values(["dataset", "origin_seed", "run_dir"])
    numeric_columns = origin_table.select_dtypes(include="number").columns
    rng = np.random.default_rng(seed)
    summaries: list[dict] = []
    for dataset, group in origin_table.groupby("dataset", sort=True):
        row = {"dataset": dataset, "n_origins": int(len(group))}
        for column in numeric_columns:
            values = group[column].dropna().to_numpy(dtype=float)
            lower, upper = _bootstrap_ci(values, n_bootstrap, rng)
            row[f"{column}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{column}_ci95_low"] = lower
            row[f"{column}_ci95_high"] = upper
        summaries.append(row)
    summary_table = pd.DataFrame(summaries)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    origin_table.to_csv(output_path / "origin_level_summary.csv", index=False)
    summary_table.to_csv(output_path / "dataset_summary.csv", index=False)
    return origin_table, summary_table
