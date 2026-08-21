from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mmcyber.data import PreparedData, prepare_dataset, stratified_subset_indices
from mmcyber.model import MLPClassifier, resolve_device
from mmcyber.utils import save_json, set_seed


def _hidden_dims_label(hidden_dims: list[int]) -> str:
    return "x".join(str(dim) for dim in hidden_dims)


def _activation_variants(config: dict) -> list[str]:
    return config["training"].get("activation_variants", [config["training"].get("activation", "relu")])


def _comparison_seeds(config: dict, origin_seed: int) -> list[int]:
    seeds = config["training"].get("comparison_seeds")
    if seeds:
        return [int(seed) for seed in seeds if int(seed) != origin_seed]
    return [int(seed) for seed in config["training"].get("seeds", []) if int(seed) != origin_seed]


def _bootstrap_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    n_samples = max(1, int(round(len(y) * fraction)))
    rng = np.random.default_rng(seed)
    return rng.choice(len(y), size=n_samples, replace=True)


def _training_subset(
    data: PreparedData,
    subset_fraction: float,
    sample_seed: int,
    subset_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if subset_strategy == "bootstrap":
        subset_idx = _bootstrap_indices(data.y_train, subset_fraction, sample_seed)
    else:
        subset_idx = stratified_subset_indices(data.y_train, subset_fraction, sample_seed)
    return data.x_train[subset_idx], data.y_train[subset_idx], subset_idx


def _resolved_origin_seed(config: dict, origin_seed: int | None) -> int:
    if origin_seed is not None:
        return int(origin_seed)
    return int(config["training"].get("origin_seed", config["training"]["seeds"][0]))


def _resolve_run_dir(config: dict, origin_seed: int) -> Path:
    run_dir = str(config["run_dir"]).format(origin_seed=origin_seed)
    return Path(run_dir)


def _resolve_experiment_name(config: dict, origin_seed: int) -> str:
    return str(config.get("experiment_name", "experiment")).format(origin_seed=origin_seed)


def _build_model_id(spec: dict) -> str:
    parts = [
        f"origin_seed{spec['origin_seed']}",
        spec["comparison_block"],
        spec["model_family"],
    ]
    if spec["comparison_block"] == "baseline":
        parts.append("reference")
    elif spec["comparison_block"] in {"seed", "xgboost"}:
        parts.append(f"train_seed{spec['train_seed']}")
    elif spec["comparison_block"] == "bootstrap":
        parts.append(f"resample_seed{spec['data_seed']}")
    elif spec["comparison_block"] == "architecture_activation":
        parts.append(spec["architecture_id"])
        parts.append(f"act_{spec['activation']}")
    return "_".join(parts)


def _default_architecture_variants(config: dict) -> list[list[int]]:
    variants = config["training"].get("architecture_variants")
    if variants:
        return [list(variant) for variant in variants]
    return [
        list(hidden_dims)
        for hidden_dims in config["training"].get("hidden_dims_variants", [config["training"]["hidden_dims"]])
    ]


def _mlp_spec(
    *,
    origin_seed: int,
    comparison_block: str,
    train_seed: int,
    data_seed: int,
    subset_strategy: str,
    subset_fraction: float,
    hidden_dims: list[int],
    activation: str,
) -> dict:
    spec = {
        "origin_seed": origin_seed,
        "comparison_block": comparison_block,
        "model_family": "mlp",
        "train_seed": train_seed,
        "data_seed": data_seed,
        "subset_strategy": subset_strategy,
        "subset_fraction": subset_fraction,
        "hidden_dims": list(hidden_dims),
        "activation": activation,
        "architecture_id": f"arch_{_hidden_dims_label(list(hidden_dims))}",
        "hidden_dims_label": _hidden_dims_label(list(hidden_dims)),
    }
    spec["model_id"] = _build_model_id(spec)
    return spec


def _xgboost_spec(origin_seed: int, train_seed: int, subset_fraction: float) -> dict:
    spec = {
        "origin_seed": origin_seed,
        "comparison_block": "xgboost",
        "model_family": "xgboost",
        "train_seed": train_seed,
        "data_seed": origin_seed,
        "subset_strategy": "stratified",
        "subset_fraction": subset_fraction,
        "hidden_dims": [],
        "activation": "none",
        "architecture_id": "xgboost",
        "hidden_dims_label": "none",
    }
    spec["model_id"] = _build_model_id(spec)
    return spec


def _factorized_specs(config: dict, origin_seed: int) -> list[dict]:
    training_config = config["training"]
    subset_fraction = float(training_config.get("factorized_subset_fraction", 1.0))
    base_hidden_dims = list(training_config["hidden_dims"])
    base_activation = training_config.get("activation", "relu")
    comparison_seeds = _comparison_seeds(config, origin_seed)
    specs = [
        _mlp_spec(
            origin_seed=origin_seed,
            comparison_block="baseline",
            train_seed=origin_seed,
            data_seed=origin_seed,
            subset_strategy="stratified",
            subset_fraction=subset_fraction,
            hidden_dims=base_hidden_dims,
            activation=base_activation,
        )
    ]
    specs.extend(
        _mlp_spec(
            origin_seed=origin_seed,
            comparison_block="seed",
            train_seed=seed,
            data_seed=origin_seed,
            subset_strategy="stratified",
            subset_fraction=subset_fraction,
            hidden_dims=base_hidden_dims,
            activation=base_activation,
        )
        for seed in comparison_seeds
    )
    specs.extend(
        _mlp_spec(
            origin_seed=origin_seed,
            comparison_block="bootstrap",
            train_seed=origin_seed,
            data_seed=seed,
            subset_strategy="bootstrap",
            subset_fraction=subset_fraction,
            hidden_dims=base_hidden_dims,
            activation=base_activation,
        )
        for seed in comparison_seeds
    )
    specs.extend(
        _mlp_spec(
            origin_seed=origin_seed,
            comparison_block="architecture_activation",
            train_seed=origin_seed,
            data_seed=origin_seed,
            subset_strategy="stratified",
            subset_fraction=subset_fraction,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        for hidden_dims in _default_architecture_variants(config)
        for activation in _activation_variants(config)
        if not (list(hidden_dims) == base_hidden_dims and activation == base_activation)
    )
    if training_config.get("xgboost", {}).get("enabled", False):
        xgboost_seeds = training_config["xgboost"].get("seeds", comparison_seeds)
        specs.extend(
            _xgboost_spec(origin_seed=origin_seed, train_seed=int(seed), subset_fraction=subset_fraction)
            for seed in xgboost_seeds
        )
    return specs


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for (batch_x,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size):
            logits = model(batch_x.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probabilities, axis=0)
    return probs.argmax(axis=1), probs


def _prediction_frame(spec: dict, data: PreparedData, pred: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": np.arange(len(data.y_test)),
            "y_true": data.y_test,
            "model_id": spec["model_id"],
            "seed": spec["train_seed"],
            "train_seed": spec["train_seed"],
            "data_seed": spec["data_seed"],
            "origin_seed": spec["origin_seed"],
            "comparison_block": spec["comparison_block"],
            "model_family": spec["model_family"],
            "subset_strategy": spec["subset_strategy"],
            "subset_fraction": spec["subset_fraction"],
            "architecture_id": spec["architecture_id"],
            "hidden_dims_label": spec["hidden_dims_label"],
            "activation": spec["activation"],
            "y_pred": pred,
        }
    )
    for class_idx, class_name in enumerate(data.class_names):
        frame[f"prob_{class_name}"] = probs[:, class_idx]
    return frame


def _write_training_artifacts(
    run_dir: Path,
    data: PreparedData,
    all_metrics: list[dict],
    prediction_frames: list[pd.DataFrame],
) -> None:
    pd.DataFrame(all_metrics).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(run_dir / "test_predictions.csv", index=False)
    pd.Series(data.class_names, name="class_name").to_csv(run_dir / "class_names.csv", index=False)


def _xgboost_dataset_cache_path(run_dir: Path) -> Path:
    return run_dir / "xgboost_dataset.joblib"


def _ensure_xgboost_dataset_cache(run_dir: Path, data: PreparedData) -> Path:
    cache_path = _xgboost_dataset_cache_path(run_dir)
    if cache_path.exists():
        return cache_path
    joblib.dump(
        {
            "x_train": data.x_train,
            "y_train": data.y_train,
            "x_test": data.x_test,
            "y_test": data.y_test,
            "class_names": data.class_names,
        },
        cache_path,
    )
    return cache_path


def _xgboost_payload(run_dir: Path, spec: dict, data: PreparedData, config: dict) -> dict:
    training_config = config["training"]
    xgb_config = training_config.get("xgboost", {})
    subset_idx = _training_subset(
        data,
        float(spec["subset_fraction"]),
        int(spec["data_seed"]),
        str(spec["subset_strategy"]),
    )[2]
    params = {
        "n_estimators": int(xgb_config.get("n_estimators", 200)),
        "max_depth": int(xgb_config.get("max_depth", 6)),
        "learning_rate": float(xgb_config.get("learning_rate", 0.1)),
        "subsample": float(xgb_config.get("subsample", 1.0)),
        "colsample_bytree": float(xgb_config.get("colsample_bytree", 1.0)),
        "reg_lambda": float(xgb_config.get("reg_lambda", 1.0)),
        "random_state": int(spec["train_seed"]),
        "n_jobs": int(xgb_config.get("n_jobs", 1)),
        "tree_method": xgb_config.get("tree_method", "hist"),
        "eval_metric": "mlogloss" if len(data.class_names) > 2 else "logloss",
    }
    if len(data.class_names) > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = len(data.class_names)
    else:
        params["objective"] = "binary:logistic"
    return {
        "dataset_path": str(_ensure_xgboost_dataset_cache(run_dir, data)),
        "subset_idx": subset_idx,
        "params": params,
        "spec": spec,
        "model_path": str(run_dir / "models" / f"{spec['model_id']}.joblib"),
    }


def train_one_model(
    data: PreparedData,
    config: dict,
    train_seed: int,
    data_seed: int,
    subset_fraction: float,
    subset_strategy: str,
    hidden_dims: list[int],
    activation: str,
    architecture_id: str,
    model_id: str,
    origin_seed: int,
    comparison_block: str,
    run_dir: Path,
) -> dict:
    training_config = config["training"]
    set_seed(train_seed)
    device = resolve_device(training_config.get("device", "auto"))
    x_train, y_train, subset_idx = _training_subset(data, subset_fraction, data_seed, subset_strategy)

    output_dim = len(data.class_names)
    model = MLPClassifier(
        input_dim=data.x_train.shape[1],
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=training_config["dropout"],
    ).to(device)

    train_loader = _loader(x_train, y_train, training_config["batch_size"], shuffle=True)
    val_loader = _loader(data.x_val, data.y_val, training_config["batch_size"], shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for _epoch in range(training_config["epochs"]):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                val_losses.append(criterion(model(batch_x), batch_y).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val_loss:
            # Keep the best validation checkpoint instead of the final epoch so
            # all models are compared at their strongest observed state.
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    pred, probs = _predict(model, data.x_test, device, training_config["batch_size"])
    model_path = run_dir / "models" / f"{model_id}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": data.x_train.shape[1],
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "activation": activation,
            "architecture_id": architecture_id,
            "hidden_dims_label": _hidden_dims_label(hidden_dims),
            "dropout": training_config["dropout"],
            "seed": train_seed,
            "train_seed": train_seed,
            "data_seed": data_seed,
            "origin_seed": origin_seed,
            "comparison_block": comparison_block,
            "model_family": "mlp",
            "subset_strategy": subset_strategy,
            "subset_fraction": subset_fraction,
            "class_names": data.class_names,
        },
        model_path,
    )

    return {
        "model_id": model_id,
        "seed": train_seed,
        "train_seed": train_seed,
        "data_seed": data_seed,
        "origin_seed": origin_seed,
        "comparison_block": comparison_block,
        "model_family": "mlp",
        "subset_strategy": subset_strategy,
        "subset_fraction": subset_fraction,
        "architecture_id": architecture_id,
        "hidden_dims_label": _hidden_dims_label(hidden_dims),
        "activation": activation,
        "subset_size": int(len(subset_idx)),
        "model_path": str(model_path),
        "accuracy": accuracy_score(data.y_test, pred),
        "macro_f1": f1_score(data.y_test, pred, average="macro"),
        "log_loss": log_loss(data.y_test, probs, labels=list(range(output_dim))),
        "pred": pred,
        "probs": probs,
    }


def train_one_xgboost(
    data: PreparedData,
    config: dict,
    *,
    model_id: str,
    origin_seed: int,
    train_seed: int,
    data_seed: int,
    subset_fraction: float,
    subset_strategy: str,
    comparison_block: str,
    run_dir: Path,
) -> dict:
    spec = {
        "model_id": model_id,
        "origin_seed": origin_seed,
        "comparison_block": comparison_block,
        "model_family": "xgboost",
        "train_seed": train_seed,
        "data_seed": data_seed,
        "subset_strategy": subset_strategy,
        "subset_fraction": subset_fraction,
        "hidden_dims": [],
        "activation": "none",
        "architecture_id": "xgboost",
        "hidden_dims_label": "none",
    }
    payload = _xgboost_payload(run_dir, spec, data, config)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with tempfile.TemporaryDirectory(prefix="mmcyber-xgb-") as tmp_dir:
        payload_path = Path(tmp_dir) / "payload.joblib"
        result_path = Path(tmp_dir) / "result.joblib"
        joblib.dump(payload, payload_path)
        completed = subprocess.run(
            [sys.executable, "-m", "mmcyber.xgboost_worker", "--payload", str(payload_path), "--result", str(result_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or f"worker exited with code {completed.returncode}"
            raise RuntimeError(f"XGBoost worker failed for {model_id}: {details}")
        return joblib.load(result_path)


def run_training(
    config: dict,
    seeds: list[int] | None = None,
    subset_fractions: list[float] | None = None,
    hidden_dims_variants: list[list[int]] | None = None,
    activation_variants: list[str] | None = None,
    origin_seed: int | None = None,
) -> None:
    origin_seed = _resolved_origin_seed(config, origin_seed)
    config = {**config, "experiment_name": _resolve_experiment_name(config, origin_seed)}
    config["run_dir"] = str(_resolve_run_dir(config, origin_seed))
    config["training"] = dict(config["training"])
    config["training"]["origin_seed"] = origin_seed

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, run_dir / "config.resolved.json")

    data = prepare_dataset(config, run_dir)

    all_metrics = []
    prediction_frames = []
    pd.Series(data.class_names, name="class_name").to_csv(run_dir / "class_names.csv", index=False)
    if config["training"].get("design") == "factorized":
        specs = _factorized_specs(config, origin_seed)
    else:
        seeds = seeds or config["training"]["seeds"]
        subset_fractions = subset_fractions or config["training"]["subset_fractions"]
        hidden_dims_variants = hidden_dims_variants or config["training"].get(
            "hidden_dims_variants",
            [config["training"]["hidden_dims"]],
        )
        activation_variants = activation_variants or _activation_variants(config)
        specs = []
        for seed in seeds:
            for fraction in subset_fractions:
                for hidden_dims in hidden_dims_variants:
                    for activation in activation_variants:
                        specs.append(
                            _mlp_spec(
                                origin_seed=origin_seed,
                                comparison_block="grid",
                                train_seed=seed,
                                data_seed=seed,
                                subset_strategy="stratified",
                                subset_fraction=fraction,
                                hidden_dims=list(hidden_dims),
                                activation=activation,
                            )
                        )

    pd.DataFrame(specs).to_csv(run_dir / "training_plan.csv", index=False)
    for spec in tqdm(specs, desc="models"):
        if spec["model_family"] == "xgboost":
            result = train_one_xgboost(
                data,
                config,
                model_id=spec["model_id"],
                origin_seed=spec["origin_seed"],
                train_seed=spec["train_seed"],
                data_seed=spec["data_seed"],
                subset_fraction=spec["subset_fraction"],
                subset_strategy=spec["subset_strategy"],
                comparison_block=spec["comparison_block"],
                run_dir=run_dir,
            )
        else:
            result = train_one_model(
                data,
                config,
                spec["train_seed"],
                spec["data_seed"],
                spec["subset_fraction"],
                spec["subset_strategy"],
                spec["hidden_dims"],
                spec["activation"],
                spec["architecture_id"],
                spec["model_id"],
                spec["origin_seed"],
                spec["comparison_block"],
                run_dir,
            )
        probs = result.pop("probs")
        pred = result.pop("pred")
        all_metrics.append(result)
        prediction_frames.append(_prediction_frame(spec, data, pred, probs))
        _write_training_artifacts(run_dir, data, all_metrics, prediction_frames)
