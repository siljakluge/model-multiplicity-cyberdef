from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def run_worker(payload_path: str | Path, result_path: str | Path) -> None:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError("xgboost is not installed. Install project dependencies to use the XGBoost comparison block.") from exc

    payload = joblib.load(payload_path)
    dataset = joblib.load(payload["dataset_path"])
    subset_idx = np.asarray(payload["subset_idx"], dtype=np.int64)
    spec = payload["spec"]

    x_train = dataset["x_train"][subset_idx]
    y_train = dataset["y_train"][subset_idx]
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]
    class_names = dataset["class_names"]

    model = XGBClassifier(**payload["params"])
    model.fit(x_train, y_train, verbose=False)
    pred = model.predict(x_test)
    probs = model.predict_proba(x_test)

    model_path = Path(payload["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "input_dim": int(x_train.shape[1]),
            "output_dim": int(len(class_names)),
            "train_seed": spec["train_seed"],
            "data_seed": spec["data_seed"],
            "origin_seed": spec["origin_seed"],
            "comparison_block": spec["comparison_block"],
            "model_family": "xgboost",
            "subset_strategy": spec["subset_strategy"],
            "subset_fraction": spec["subset_fraction"],
            "architecture_id": "xgboost",
            "hidden_dims_label": "none",
            "activation": "none",
            "class_names": class_names,
        },
        model_path,
    )

    result = {
        "model_id": spec["model_id"],
        "seed": spec["train_seed"],
        "train_seed": spec["train_seed"],
        "data_seed": spec["data_seed"],
        "origin_seed": spec["origin_seed"],
        "comparison_block": spec["comparison_block"],
        "model_family": "xgboost",
        "subset_strategy": spec["subset_strategy"],
        "subset_fraction": spec["subset_fraction"],
        "architecture_id": "xgboost",
        "hidden_dims_label": "none",
        "activation": "none",
        "subset_size": int(len(subset_idx)),
        "model_path": str(model_path),
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "log_loss": log_loss(y_test, probs, labels=list(range(len(class_names)))),
        "pred": pred,
        "probs": probs,
    }
    joblib.dump(result, result_path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="mmcyber.xgboost_worker")
    parser.add_argument("--payload", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    run_worker(args.payload, args.result)


if __name__ == "__main__":
    main()
