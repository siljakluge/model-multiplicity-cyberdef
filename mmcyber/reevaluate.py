from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss

from mmcyber.model import MLPClassifier, resolve_device
from mmcyber.train import _predict
from mmcyber.data import prepare_dataset
from mmcyber.utils import load_config


def _load_cached_dataset(run_path: Path) -> dict:
    cache_path = run_path / "mlp_dataset.joblib"
    if not cache_path.exists():
        config_path = run_path / "config.resolved.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing {cache_path} and {config_path}. Download either the saved dataset cache or the resolved config "
                "to reconstruct the exact deterministic split."
            )
        prepared = prepare_dataset(load_config(config_path), run_path)
        return {
            "x_val": prepared.x_val,
            "y_val": prepared.y_val,
            "x_test": prepared.x_test,
            "y_test": prepared.y_test,
            "class_names": prepared.class_names,
        }
    dataset = joblib.load(cache_path)
    required = {"x_val", "y_val", "x_test", "y_test", "class_names"}
    missing = required - set(dataset)
    if missing:
        raise ValueError(f"{cache_path} lacks required fields: {sorted(missing)}")
    return dataset


def _evaluate_mlp(model_path: Path, dataset: dict, device: torch.device, batch_size: int) -> dict[str, float]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = MLPClassifier(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        activation=str(checkpoint["activation"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_pred, val_probs = _predict(model, dataset["x_val"], device, batch_size)
    test_pred, test_probs = _predict(model, dataset["x_test"], device, batch_size)
    labels = list(range(len(dataset["class_names"])))
    return {
        "val_accuracy": accuracy_score(dataset["y_val"], val_pred),
        "val_macro_f1": f1_score(dataset["y_val"], val_pred, average="macro"),
        "val_log_loss": log_loss(dataset["y_val"], val_probs, labels=labels),
        "recomputed_test_accuracy": accuracy_score(dataset["y_test"], test_pred),
        "recomputed_test_macro_f1": f1_score(dataset["y_test"], test_pred, average="macro"),
        "recomputed_test_log_loss": log_loss(dataset["y_test"], test_probs, labels=labels),
    }


def _evaluate_xgboost(model_path: Path, dataset: dict) -> dict[str, float]:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    val_pred, val_probs = model.predict(dataset["x_val"]), model.predict_proba(dataset["x_val"])
    test_pred, test_probs = model.predict(dataset["x_test"]), model.predict_proba(dataset["x_test"])
    labels = list(range(len(dataset["class_names"])))
    return {
        "val_accuracy": accuracy_score(dataset["y_val"], val_pred),
        "val_macro_f1": f1_score(dataset["y_val"], val_pred, average="macro"),
        "val_log_loss": log_loss(dataset["y_val"], val_probs, labels=labels),
        "recomputed_test_accuracy": accuracy_score(dataset["y_test"], test_pred),
        "recomputed_test_macro_f1": f1_score(dataset["y_test"], test_pred, average="macro"),
        "recomputed_test_log_loss": log_loss(dataset["y_test"], test_probs, labels=labels),
    }


def reconstruct_validation_metrics(run_dir: str | Path, batch_size: int = 1024, device_name: str = "auto") -> pd.DataFrame:
    """Add validation metrics by evaluating the already saved model artifacts.

    This performs no training. It also writes recomputed test metrics, enabling
    an integrity check against the original metrics.csv before final analysis.
    """
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    dataset = _load_cached_dataset(run_path)
    device = resolve_device(device_name)
    rows = []
    for _, row in metrics.iterrows():
        model_path = run_path / "models" / Path(str(row["model_path"])).name
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact {model_path}.")
        values = _evaluate_xgboost(model_path, dataset) if model_path.suffix == ".joblib" else _evaluate_mlp(
            model_path, dataset, device, batch_size
        )
        values["model_id"] = row["model_id"]
        rows.append(values)
    reconstructed = pd.DataFrame(rows)
    result = metrics.merge(reconstructed, on="model_id", validate="one_to_one")
    result.to_csv(metrics_path, index=False)
    return result
