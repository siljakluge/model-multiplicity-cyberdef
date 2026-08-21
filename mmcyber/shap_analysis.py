from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

from mmcyber.data import PreparedData, prepare_dataset
from mmcyber.utils import load_config


def _iter_model_artifacts(models_dir: Path) -> list[Path]:
    return sorted(path for path in models_dir.glob("*") if path.suffix in {".pt", ".joblib"})


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
            return rng.choice(conflict_ids, size=min(max_explain, len(conflict_ids)), replace=False)
    return rng.choice(n_test, size=min(max_explain, n_test), replace=False)


def _require_run_file(run_path: Path, name: str, hint: str) -> Path:
    path = run_path / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required run artifact {path}. {hint}")
    return path


def _shap_dataset_cache_path(run_dir: Path) -> Path:
    return run_dir / "shap_dataset.joblib"


def _ensure_shap_dataset_cache(run_dir: Path, data: PreparedData) -> Path:
    cache_path = _shap_dataset_cache_path(run_dir)
    if cache_path.exists():
        return cache_path
    joblib.dump(
        {
            "x_train": data.x_train,
            "x_test": data.x_test,
            "feature_names": data.feature_names,
            "class_names": data.class_names,
        },
        cache_path,
    )
    return cache_path


def _shap_worker_payload(
    *,
    run_dir: Path,
    model_path: Path,
    dataset_path: Path,
    background_idx: np.ndarray,
    explain_idx: np.ndarray,
    device_name: str,
) -> dict:
    return {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "background_idx": np.asarray(background_idx, dtype=np.int64),
        "explain_idx": np.asarray(explain_idx, dtype=np.int64),
        "device": device_name,
    }


def _compute_model_shap(payload: dict) -> dict:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with tempfile.TemporaryDirectory(prefix="mmcyber-shap-") as tmp_dir:
        payload_path = Path(tmp_dir) / "payload.joblib"
        result_path = Path(tmp_dir) / "result.joblib"
        joblib.dump(payload, payload_path)
        completed = subprocess.run(
            [sys.executable, "-m", "mmcyber.shap_worker", "--payload", str(payload_path), "--result", str(result_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or f"worker exited with code {completed.returncode}"
            raise RuntimeError(f"SHAP worker failed for {Path(payload['model_path']).name}: {details}")
        return joblib.load(result_path)


def compute_shap(
    run_dir: str | Path,
    max_background: int = 128,
    max_explain: int = 256,
    only_conflicts: bool = False,
) -> None:
    run_path = Path(run_dir)
    _require_run_file(run_path, "config.resolved.json", "Run train first so the resolved config is available.")
    models_dir = _require_run_file(run_path, "models", "Run train first so model artifacts exist.")

    config = load_config(run_path / "config.resolved.json")
    data = prepare_dataset(config, run_path)

    model_paths = _iter_model_artifacts(models_dir)
    if not model_paths:
        raise FileNotFoundError(f"No model artifacts found under {models_dir}. Run train first.")

    rng = np.random.default_rng(42)
    background_idx = rng.choice(len(data.x_train), size=min(max_background, len(data.x_train)), replace=False)
    explain_idx = _select_explain_indices(run_path, len(data.x_test), max_explain, seed=42, only_conflicts=only_conflicts)
    dataset_path = _ensure_shap_dataset_cache(run_path, data)

    shap_dir = run_path / "shap_values"
    shap_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    value_rows = []
    skipped_models = []
    device_name = str(config.get("training", {}).get("device", "auto"))

    for model_path in model_paths:
        if model_path.suffix not in {".pt", ".joblib"}:
            skipped_models.append(model_path.name)
            continue
        result = _compute_model_shap(
            _shap_worker_payload(
                run_dir=run_path,
                model_path=model_path,
                dataset_path=dataset_path,
                background_idx=background_idx,
                explain_idx=explain_idx,
                device_name=device_name,
            )
        )
        class_first_values = np.asarray(result["shap_values"])

        np.savez_compressed(
            shap_dir / f"{model_path.stem}.npz",
            shap_values=class_first_values,
            sample_indices=np.asarray(result["sample_indices"], dtype=np.int64),
            feature_names=np.array(data.feature_names),
            class_names=np.array(data.class_names),
        )
        summary_rows.extend(result["summary_rows"])
        value_rows.extend(result["value_rows"])

    skipped_models.extend(
        path.name for path in models_dir.glob("*") if path.suffix not in {".pt", ".joblib"}
    )
    skipped_models = sorted(set(skipped_models))
    if skipped_models:
        pd.DataFrame({"skipped_model_artifact": skipped_models}).to_csv(run_path / "shap_skipped_models.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(run_path / "shap_summary.csv", index=False)
    pd.DataFrame(value_rows).to_csv(run_path / "shap_values_long.csv.gz", index=False)
