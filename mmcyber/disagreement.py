from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _entropy(votes: np.ndarray) -> float:
    _, counts = np.unique(votes, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def compute_disagreement(run_dir: str | Path) -> None:
    run_path = Path(run_dir)
    predictions = pd.read_csv(run_path / "test_predictions.csv")
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
        sample_rows.append(
            {
                "sample_id": sample_id,
                "y_true": int(y_true.loc[sample_id]),
                "majority_pred": majority_label,
                "vote_entropy": _entropy(votes),
                "unique_predictions": int(len(values)),
                "majority_fraction": float(counts[majority_idx] / counts.sum()),
            }
        )

    pd.DataFrame(rows).to_csv(run_path / "disagreement_summary.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(run_path / "sample_disagreement.csv", index=False)
