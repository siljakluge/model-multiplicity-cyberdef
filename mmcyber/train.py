from __future__ import annotations

from pathlib import Path

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


def train_one_model(
    data: PreparedData,
    config: dict,
    seed: int,
    subset_fraction: float,
    model_id: str,
    run_dir: Path,
) -> dict:
    training_config = config["training"]
    set_seed(seed)
    device = resolve_device(training_config.get("device", "auto"))
    subset_idx = stratified_subset_indices(data.y_train, subset_fraction, seed)
    x_train = data.x_train[subset_idx]
    y_train = data.y_train[subset_idx]

    output_dim = len(data.class_names)
    model = MLPClassifier(
        input_dim=data.x_train.shape[1],
        output_dim=output_dim,
        hidden_dims=training_config["hidden_dims"],
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
            "hidden_dims": training_config["hidden_dims"],
            "dropout": training_config["dropout"],
            "seed": seed,
            "subset_fraction": subset_fraction,
            "class_names": data.class_names,
        },
        model_path,
    )

    return {
        "model_id": model_id,
        "seed": seed,
        "subset_fraction": subset_fraction,
        "subset_size": int(len(subset_idx)),
        "model_path": str(model_path),
        "accuracy": accuracy_score(data.y_test, pred),
        "macro_f1": f1_score(data.y_test, pred, average="macro"),
        "log_loss": log_loss(data.y_test, probs, labels=list(range(output_dim))),
        "pred": pred,
        "probs": probs,
    }


def run_training(config: dict, seeds: list[int] | None = None, subset_fractions: list[float] | None = None) -> None:
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, run_dir / "config.resolved.json")

    data = prepare_dataset(config, run_dir)
    seeds = seeds or config["training"]["seeds"]
    subset_fractions = subset_fractions or config["training"]["subset_fractions"]

    all_metrics = []
    prediction_frames = []
    for seed in tqdm(seeds, desc="seeds"):
        for fraction in subset_fractions:
            model_id = f"seed{seed}_frac{str(fraction).replace('.', 'p')}"
            result = train_one_model(data, config, seed, fraction, model_id, run_dir)
            probs = result.pop("probs")
            pred = result.pop("pred")
            all_metrics.append(result)

            frame = pd.DataFrame(
                {
                    "sample_id": np.arange(len(data.y_test)),
                    "y_true": data.y_test,
                    "model_id": model_id,
                    "y_pred": pred,
                }
            )
            for class_idx, class_name in enumerate(data.class_names):
                frame[f"prob_{class_name}"] = probs[:, class_idx]
            prediction_frames.append(frame)

    pd.DataFrame(all_metrics).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(run_dir / "test_predictions.csv", index=False)
    pd.Series(data.class_names, name="class_name").to_csv(run_dir / "class_names.csv", index=False)
