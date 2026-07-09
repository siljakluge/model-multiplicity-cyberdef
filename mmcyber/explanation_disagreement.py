from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(1.0 - (np.dot(a, b) / denom))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(x.rank().corr(y.rank()))


def compute_explanation_disagreement(run_dir: str | Path, top_k: int = 20) -> None:
    run_path = Path(run_dir)
    summary = pd.read_csv(run_path / "shap_summary.csv")
    rows = []

    for class_name, class_frame in summary.groupby("class_name"):
        vectors = {}
        top_features = {}
        for model_id, model_frame in class_frame.groupby("model_id"):
            vector = model_frame.set_index("feature")["mean_abs_shap"]
            vectors[model_id] = vector
            top_features[model_id] = set(model_frame.nsmallest(top_k, "rank")["feature"])

        all_features = sorted(set().union(*(set(v.index) for v in vectors.values())))
        for model_a, model_b in combinations(sorted(vectors), 2):
            a = vectors[model_a].reindex(all_features, fill_value=0.0).to_numpy()
            b = vectors[model_b].reindex(all_features, fill_value=0.0).to_numpy()
            intersection = len(top_features[model_a] & top_features[model_b])
            union = len(top_features[model_a] | top_features[model_b])
            rows.append(
                {
                    "class_name": class_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "mean_abs_shap_cosine_distance": _cosine_distance(a, b),
                    "top_k": top_k,
                    "top_k_jaccard": float(intersection / union) if union else 1.0,
                }
            )

    pd.DataFrame(rows).to_csv(run_path / "explanation_disagreement.csv", index=False)

    shap_values_path = run_path / "shap_values_long.csv.gz"
    sample_disagreement_path = run_path / "sample_disagreement.csv"
    if shap_values_path.exists() and sample_disagreement_path.exists():
        compute_shap_variability(run_path)


def compute_shap_variability(run_dir: str | Path) -> None:
    run_path = Path(run_dir)
    shap_values = pd.read_csv(run_path / "shap_values_long.csv.gz")
    sample_disagreement = pd.read_csv(run_path / "sample_disagreement.csv")
    sample_cols = ["sample_id", "conflict_ratio", "is_conflict", "vote_entropy", "majority_fraction"]
    available_cols = [column for column in sample_cols if column in sample_disagreement.columns]

    grouped = shap_values.groupby(["sample_id", "class_name", "feature"])["shap_value"]
    variability = grouped.agg(
        shap_value_min="min",
        shap_value_max="max",
        shap_value_mean="mean",
        shap_value_std="std",
        shap_value_variance="var",
    ).reset_index()
    variability["shap_value_range"] = variability["shap_value_max"] - variability["shap_value_min"]

    positive_fraction = grouped.apply(lambda values: float((values > 0).mean())).reset_index(name="positive_fraction")
    variability = variability.merge(positive_fraction, on=["sample_id", "class_name", "feature"])
    variability["sign_instability"] = np.minimum(
        variability["positive_fraction"],
        1.0 - variability["positive_fraction"],
    )
    variability["shap_value_std"] = variability["shap_value_std"].fillna(0.0)
    variability["shap_value_variance"] = variability["shap_value_variance"].fillna(0.0)
    variability = variability.merge(sample_disagreement[available_cols], on="sample_id", how="left")
    variability.to_csv(run_path / "shap_variability.csv", index=False)

    correlation_rows = []
    for (class_name, feature), frame in variability.groupby(["class_name", "feature"]):
        correlation_rows.append(
            {
                "class_name": class_name,
                "feature": feature,
                "n_samples": int(len(frame)),
                "spearman_conflict_vs_shap_range": _spearman(frame["conflict_ratio"], frame["shap_value_range"]),
                "spearman_conflict_vs_shap_variance": _spearman(frame["conflict_ratio"], frame["shap_value_variance"]),
                "spearman_conflict_vs_sign_instability": _spearman(frame["conflict_ratio"], frame["sign_instability"]),
                "mean_shap_range": float(frame["shap_value_range"].mean()),
                "mean_shap_variance": float(frame["shap_value_variance"].mean()),
                "mean_sign_instability": float(frame["sign_instability"].mean()),
            }
        )
    pd.DataFrame(correlation_rows).to_csv(run_path / "shap_variability_correlations.csv", index=False)
