"""Non-training server preflight checks for the full five-dataset run."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import tempfile

from mmcyber.data import PreparedData, prepare_dataset
from mmcyber.train import _factorized_specs
from mmcyber.utils import load_config


FULL_CONFIGS = (
    Path("configs/default.yaml"),
    Path("configs/folktables_income_full.yaml"),
    Path("configs/folktables_employment_full.yaml"),
    Path("configs/adult_full.yaml"),
    Path("configs/bank_marketing_full.yaml"),
)
EXPECTED_DATASETS = ("nsl_kdd", "ACSIncome", "ACSEmployment", "adult", "bank_marketing")


def _validate_prepared_data(data: PreparedData, label: str) -> None:
    if not data.feature_names:
        raise ValueError(f"{label}: preprocessing produced no features.")
    if len(data.class_names) != 2:
        raise ValueError(f"{label}: expected binary class names, got {data.class_names!r}.")
    for split_name, x, y in (
        ("train", data.x_train, data.y_train),
        ("validation", data.x_val, data.y_val),
        ("test", data.x_test, data.y_test),
    ):
        if len(x) == 0 or len(x) != len(y):
            raise ValueError(f"{label}: invalid {split_name} split shape {x.shape}/{y.shape}.")
        if len(set(y.tolist())) != 2:
            raise ValueError(f"{label}: {split_name} split does not contain both classes.")


def preflight_full_configs(config_paths: Iterable[str | Path] = FULL_CONFIGS) -> list[str]:
    """Validate plans and prepare every full dataset without training models.

    Missing raw data are downloaded by the normal loaders, so this also verifies
    the server's network/proxy setup. Preprocessors are written only to a
    temporary directory and no files under ``runs/`` are changed.
    """
    paths = [Path(path) for path in config_paths]
    if not paths:
        raise ValueError("At least one full-run config is required.")

    labels: list[str] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Full-run config not found: {path}")
        config = load_config(path)
        dataset = config.get("dataset", {})
        training = config.get("training", {})
        label = str(dataset.get("problem") or dataset.get("name") or path.stem)
        origins = training.get("origin_seeds")
        if origins != list(range(1, 11)):
            raise ValueError(f"{path}: expected origin_seeds [1, ..., 10], got {origins!r}.")
        if "{origin_seed}" not in str(config.get("run_dir", "")):
            raise ValueError(f"{path}: run_dir must include '{{origin_seed}}'.")
        for origin_seed in origins:
            specs = _factorized_specs(config, origin_seed)
            if len(specs) != 63 or len({spec["model_id"] for spec in specs}) != 63:
                raise ValueError(f"{path}: origin {origin_seed} does not define 63 unique MLP models.")

        with tempfile.TemporaryDirectory(prefix="mmcyber-preflight-") as temporary:
            data = prepare_dataset(config, Path(temporary))
        _validate_prepared_data(data, label)
        labels.append(label)
        print(
            f"OK {label}: data prepared ({len(data.x_train)}/{len(data.x_val)}/{len(data.x_test)} "
            f"train/val/test), 10 origins x 63 MLPs"
        )

    if tuple(labels) != EXPECTED_DATASETS:
        raise ValueError(f"Expected full dataset order {EXPECTED_DATASETS!r}, got {tuple(labels)!r}.")
    print("Preflight passed: all five datasets are cached or downloadable and all 3,150 planned models are valid.")
    return labels
