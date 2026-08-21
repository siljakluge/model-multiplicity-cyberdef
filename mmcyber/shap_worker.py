from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def _load_torch_model(path: Path, device):
    import torch

    from mmcyber.model import MLPClassifier

    checkpoint = torch.load(path, map_location=device)
    model = MLPClassifier(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        activation=checkpoint.get("activation", "relu"),
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _torch_model_metadata(path: Path) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    hidden_dims = checkpoint["hidden_dims"]
    return {
        "architecture_id": checkpoint.get("architecture_id", ""),
        "hidden_dims_label": checkpoint.get("hidden_dims_label", "x".join(str(dim) for dim in hidden_dims)),
        "activation": checkpoint.get("activation", "relu"),
        "seed": checkpoint.get("seed"),
        "subset_fraction": checkpoint.get("subset_fraction"),
    }


def _load_joblib_model(path: Path):
    artifact = joblib.load(path)
    return artifact["model"]


def _joblib_model_metadata(path: Path) -> dict:
    artifact = joblib.load(path)
    return {
        "architecture_id": artifact.get("architecture_id", "xgboost"),
        "hidden_dims_label": artifact.get("hidden_dims_label", "none"),
        "activation": artifact.get("activation", "none"),
        "seed": artifact.get("train_seed"),
        "subset_fraction": artifact.get("subset_fraction"),
    }


def _normalize_shap_values(values: np.ndarray, n_classes: int) -> np.ndarray:
    if values.ndim == 2:
        return values[np.newaxis, ...]
    if values.ndim == 3 and values.shape[0] == 1:
        return values
    if values.ndim != 3:
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")
    if values.shape[0] == n_classes:
        return values
    if values.shape[-1] == n_classes:
        return np.moveaxis(values, -1, 0)
    if values.shape[1] == n_classes:
        return np.moveaxis(values, 1, 0)
    raise ValueError(f"Could not align SHAP output shape {values.shape} to {n_classes} classes")


def _xgboost_shap_values(model, explain_data: np.ndarray, n_classes: int) -> np.ndarray:
    import xgboost as xgb

    contributions = model.get_booster().predict(xgb.DMatrix(explain_data), pred_contribs=True)
    values = np.asarray(contributions)
    if values.ndim == 2:
        positive_values = values[:, :-1]
        if n_classes == 2:
            return np.stack([-positive_values, positive_values], axis=0)
        return positive_values[np.newaxis, ...]
    if values.ndim == 3:
        return np.moveaxis(values[:, :, :-1], 1, 0)
    raise ValueError(f"Unsupported XGBoost SHAP output shape: {values.shape} for {n_classes} classes")


def run_worker(payload_path: str | Path, result_path: str | Path) -> None:
    import shap
    import torch

    from mmcyber.model import resolve_device

    payload = joblib.load(payload_path)
    dataset = joblib.load(payload["dataset_path"])
    model_path = Path(payload["model_path"])
    background_idx = np.asarray(payload["background_idx"], dtype=np.int64)
    explain_idx = np.asarray(payload["explain_idx"], dtype=np.int64)
    feature_names = list(dataset["feature_names"])
    class_names = list(dataset["class_names"])
    x_train = dataset["x_train"]
    x_test = dataset["x_test"]

    if model_path.suffix == ".pt":
        device = resolve_device(payload.get("device", "auto"))
        model = _load_torch_model(model_path, device)
        metadata = _torch_model_metadata(model_path)
        background = torch.from_numpy(x_train[background_idx]).to(device)
        explain = torch.from_numpy(x_test[explain_idx]).to(device)
        explainer = shap.DeepExplainer(model, background)
        values = np.asarray(explainer.shap_values(explain))
    elif model_path.suffix == ".joblib":
        model = _load_joblib_model(model_path)
        metadata = _joblib_model_metadata(model_path)
        values = _xgboost_shap_values(model, x_test[explain_idx], len(class_names))
    else:
        raise ValueError(f"Unsupported model artifact {model_path}")

    class_first_values = _normalize_shap_values(np.asarray(values), len(class_names))
    summary_rows = []
    value_rows = []
    for class_idx, class_name in enumerate(class_names):
        mean_abs = np.abs(class_first_values[class_idx]).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:50]
        for rank, feature_idx in enumerate(top_idx, start=1):
            summary_rows.append(
                {
                    "model_id": model_path.stem,
                    **metadata,
                    "class_name": class_name,
                    "rank": rank,
                    "feature": feature_names[feature_idx],
                    "mean_abs_shap": float(mean_abs[feature_idx]),
                }
            )
        for sample_pos, sample_id in enumerate(explain_idx):
            for feature_idx, feature_name in enumerate(feature_names):
                value_rows.append(
                    {
                        "model_id": model_path.stem,
                        **metadata,
                        "sample_id": int(sample_id),
                        "class_name": class_name,
                        "feature": feature_name,
                        "shap_value": float(class_first_values[class_idx, sample_pos, feature_idx]),
                    }
                )

    joblib.dump(
        {
            "model_id": model_path.stem,
            "sample_indices": explain_idx,
            "shap_values": class_first_values,
            "summary_rows": summary_rows,
            "value_rows": value_rows,
        },
        result_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="mmcyber.shap_worker")
    parser.add_argument("--payload", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    run_worker(args.payload, args.result)


if __name__ == "__main__":
    main()
