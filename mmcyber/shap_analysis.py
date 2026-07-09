from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mmcyber.data import prepare_dataset
from mmcyber.model import MLPClassifier, resolve_device
from mmcyber.utils import load_config


def _load_model(path: Path, device: torch.device) -> MLPClassifier:
    checkpoint = torch.load(path, map_location=device)
    model = MLPClassifier(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_shap(run_dir: str | Path, max_background: int = 128, max_explain: int = 256) -> None:
    import shap

    run_path = Path(run_dir)
    config = load_config(run_path / "config.resolved.json")
    data = prepare_dataset(config, run_path)
    device = resolve_device(config["training"].get("device", "auto"))

    rng = np.random.default_rng(42)
    background_idx = rng.choice(len(data.x_train), size=min(max_background, len(data.x_train)), replace=False)
    explain_idx = rng.choice(len(data.x_test), size=min(max_explain, len(data.x_test)), replace=False)
    background = torch.from_numpy(data.x_train[background_idx]).to(device)
    explain = torch.from_numpy(data.x_test[explain_idx]).to(device)

    shap_dir = run_path / "shap_values"
    shap_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for model_path in sorted((run_path / "models").glob("*.pt")):
        model = _load_model(model_path, device)
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(explain)
        values = np.asarray(shap_values)

        # shap returns either class-first or sample-first depending on version/model output.
        if values.ndim == 3 and values.shape[0] == len(data.class_names):
            class_first_values = values
        elif values.ndim == 3 and values.shape[-1] == len(data.class_names):
            class_first_values = np.moveaxis(values, -1, 0)
        else:
            class_first_values = values[np.newaxis, ...]

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
            for rank, feature_idx in enumerate(top_idx, start=1):
                summary_rows.append(
                    {
                        "model_id": model_path.stem,
                        "class_name": class_name,
                        "rank": rank,
                        "feature": data.feature_names[feature_idx],
                        "mean_abs_shap": float(mean_abs[feature_idx]),
                    }
                )

    pd.DataFrame(summary_rows).to_csv(run_path / "shap_summary.csv", index=False)
