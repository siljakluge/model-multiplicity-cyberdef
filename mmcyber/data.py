from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


NSL_KDD_TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
NSL_KDD_TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]


@dataclass
class PreparedData:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    preprocessor: ColumnTransformer


def download_nsl_kdd(raw_dir: Path) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_path = raw_dir / "KDDTrain+.txt"
    test_path = raw_dir / "KDDTest+.txt"
    if not train_path.exists():
        urlretrieve(NSL_KDD_TRAIN_URL, train_path)
    if not test_path.exists():
        urlretrieve(NSL_KDD_TEST_URL, test_path)
    return train_path, test_path


def _read_nsl_kdd(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, names=NSL_KDD_COLUMNS)


def _target(df: pd.DataFrame, task: str, encoder: LabelEncoder | None = None) -> tuple[np.ndarray, list[str], LabelEncoder | None]:
    if task == "binary":
        y = (df["label"] != "normal").astype(int).to_numpy()
        return y, ["normal", "attack"], None
    if task == "multiclass":
        if encoder is None:
            encoder = LabelEncoder()
            y = encoder.fit_transform(df["label"])
        else:
            y = encoder.transform(df["label"])
        return y, encoder.classes_.tolist(), encoder
    raise ValueError(f"Unsupported task: {task}")


def prepare_dataset(config: dict, run_dir: Path) -> PreparedData:
    dataset_config = config["dataset"]
    if dataset_config["name"] != "nsl_kdd":
        raise ValueError("Only dataset.name=nsl_kdd is implemented in this scaffold.")

    train_path, test_path = download_nsl_kdd(Path(dataset_config["raw_dir"]))
    train_df = _read_nsl_kdd(train_path)
    test_df = _read_nsl_kdd(test_path)

    task = dataset_config.get("task", "binary")
    if task == "multiclass":
        # Fit on train+test labels so the encoder can represent every NSL-KDD
        # attack class that may appear in the held-out split.
        encoder = LabelEncoder().fit(pd.concat([train_df["label"], test_df["label"]], ignore_index=True))
        y_full, class_names, _ = _target(train_df, task, encoder)
        y_test, _, _ = _target(test_df, task, encoder)
    else:
        y_full, class_names, _ = _target(train_df, task)
        y_test, _, _ = _target(test_df, task)

    drop_cols = ["label", "difficulty"]
    x_full_df = train_df.drop(columns=drop_cols)
    x_test_df = test_df.drop(columns=drop_cols)

    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [c for c in x_full_df.columns if c not in categorical_cols]
    # Preserve a fixed feature space across all models: numerical columns are
    # scaled, categorical protocol/service/flag values are one-hot encoded, and
    # unseen test categories are ignored instead of failing.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    x_train_df, x_val_df, y_train, y_val = train_test_split(
        x_full_df,
        y_full,
        test_size=dataset_config.get("test_size", 0.2),
        random_state=42,
        stratify=y_full,
    )

    x_train = preprocessor.fit_transform(x_train_df).astype(np.float32)
    x_val = preprocessor.transform(x_val_df).astype(np.float32)
    x_test = preprocessor.transform(x_test_df).astype(np.float32)

    feature_names = preprocessor.get_feature_names_out().tolist()
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
    pd.Series(feature_names, name="feature").to_csv(run_dir / "feature_names.csv", index=False)

    return PreparedData(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=feature_names,
        class_names=class_names,
        preprocessor=preprocessor,
    )


def stratified_subset_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    indices: list[np.ndarray] = []
    # Keep class proportions roughly stable when training on partial data, so
    # performance differences are less likely to come from dropped minority
    # classes and more from model multiplicity.
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        size = max(1, int(round(len(label_indices) * fraction)))
        indices.append(rng.choice(label_indices, size=size, replace=False))
    result = np.concatenate(indices)
    rng.shuffle(result)
    return result
