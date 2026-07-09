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
