from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from mmcyber.data import prepare_dataset
from mmcyber.model import MLPClassifier, resolve_device
from mmcyber.utils import load_config


def _load_torch_model(path: Path, device: torch.device) -> MLPClassifier:
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
    # SHAP returns different layouts across explainers and versions. Normalize
    # everything to [class, sample, feature] so the downstream CSV export stays
    # model-family agnostic.
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


def _iter_model_artifacts(models_dir: Path) -> list[Path]:
    return sorted(path for path in models_dir.glob("*") if path.suffix in {".pt", ".joblib"})


def _xgboost_shap_values(model, explain_data: np.ndarray, n_classes: int) -> np.ndarray:
    contributions = model.get_booster().predict(xgb.DMatrix(explain_data), pred_contribs=True)
    values = np.asarray(contributions)
    if values.ndim == 2:
        # Binary XGBoost returns one contribution vector for the positive class
        # plus a bias column. Mirror it to two class-specific tensors so the
        # downstream export matches the MLP layout [class, sample, feature].
        positive_values = values[:, :-1]
        if n_classes == 2:
            return np.stack([-positive_values, positive_values], axis=0)
        return positive_values[np.newaxis, ...]
    if values.ndim == 3:
        # Multiclass XGBoost returns [sample, class, feature + bias].
        return np.moveaxis(values[:, :, :-1], 1, 0)
    raise ValueError(f"Unsupported XGBoost SHAP output shape: {values.shape} for {n_classes} classes")


def _select_explain_indices(run_path: Path, n_test: int, max_explain: int, seed: int, only_conflicts: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if only_conflicts and (run_path / "sample_disagreement.csv").exists():
        sample_disagreement = pd.read_csv(run_path / "sample_disagreement.csv")
        if "is_conflict" in sample_disagreement.columns:
            conflict_ids = sample_disagreement.loc[sample_disagreement["is_conflict"].astype(bool), "sample_id"].to_numpy()
        elif "conflict_ratio" in sample_disagreement.columns:
            conflict_ids = sample_disagreement.loc[sample_disagreement["conflict_ratio"] > 0, "sample_id"].to_numpy()
        else:
            conflict_ids = np.array([], dtype=int)
        if len(conflict_ids):
            # Prefer ambiguous samples when requested; these are most informative
            # for comparing explanation behavior across near-equivalent models.
            return rng.choice(conflict_ids, size=min(max_explain, len(conflict_ids)), replace=False)

    return rng.choice(n_test, size=min(max_explain, n_test), replace=False)


def compute_shap(
    run_dir: str | Path,
    max_background: int = 128,
    max_explain: int = 256,
    only_conflicts: bool = False,
) -> None:
    import shap

    run_path = Path(run_dir)
    config = load_config(run_path / "config.resolved.json")
    data = prepare_dataset(config, run_path)
    device = resolve_device(config["training"].get("device", "auto"))

    rng = np.random.default_rng(42)
    # DeepExplainer needs a compact background set. Keeping the same random seed
    # across models makes SHAP values comparable within one run.
    background_idx = rng.choice(len(data.x_train), size=min(max_background, len(data.x_train)), replace=False)
    explain_idx = _select_explain_indices(run_path, len(data.x_test), max_explain, seed=42, only_conflicts=only_conflicts)
    background = torch.from_numpy(data.x_train[background_idx]).to(device)
    explain = torch.from_numpy(data.x_test[explain_idx]).to(device)

    shap_dir = run_path / "shap_values"
    shap_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    value_rows = []
    skipped_models = []

    for model_path in _iter_model_artifacts(run_path / "models"):
        if model_path.suffix == ".pt":
            model = _load_torch_model(model_path, device)
            metadata = _torch_model_metadata(model_path)
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(explain)
            values = np.asarray(shap_values)
        elif model_path.suffix == ".joblib":
            model = _load_joblib_model(model_path)
            metadata = _joblib_model_metadata(model_path)
            values = _xgboost_shap_values(model, data.x_test[explain_idx], len(data.class_names))
        else:
            skipped_models.append(model_path.name)
            continue

        class_first_values = _normalize_shap_values(np.asarray(values), len(data.class_names))

        np.savez_compressed(
            shap_dir / f"{model_path.stem}.npz",
            shap_values=class_first_values,
            sample_indices=explain_idx,
            feature_names=np.array(data.feature_names),
            class_names=np.array(data.class_names),
        )

        for class_idx, class_name in enumerate(data.class_names):
            mean_abs = np.abs(class_first_values[class_idx]).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:50]
            # Summary rows keep only the strongest features for compact plots;
            # value_rows below keeps the full per-sample tensor for variability
            # and correlation analysis.
            for rank, feature_idx in enumerate(top_idx, start=1):
                summary_rows.append(
                    {
                        "model_id": model_path.stem,
                        **metadata,
                        "class_name": class_name,
                        "rank": rank,
                        "feature": data.feature_names[feature_idx],
                        "mean_abs_shap": float(mean_abs[feature_idx]),
                    }
                )
            for sample_pos, sample_id in enumerate(explain_idx):
                for feature_idx, feature_name in enumerate(data.feature_names):
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

    skipped_models.extend(
        path.name for path in (run_path / "models").glob("*") if path.suffix not in {".pt", ".joblib"}
    )
    skipped_models = sorted(set(skipped_models))
    if skipped_models:
        pd.DataFrame({"skipped_model_artifact": skipped_models}).to_csv(run_path / "shap_skipped_models.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(run_path / "shap_summary.csv", index=False)
    pd.DataFrame(value_rows).to_csv(run_path / "shap_values_long.csv.gz", index=False)
