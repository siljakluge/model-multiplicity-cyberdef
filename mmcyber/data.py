from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
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
    preprocessor: Any


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


def _target(
    df: pd.DataFrame,
    task: str,
    encoder: LabelEncoder | None = None,
) -> tuple[np.ndarray, list[str], LabelEncoder | None]:
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


def _save_preprocessor_artifacts(run_dir: Path, preprocessor: Any, feature_names: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
    pd.Series(feature_names, name="feature").to_csv(run_dir / "feature_names.csv", index=False)


def _fit_and_transform(
    *,
    x_train_df: pd.DataFrame,
    x_val_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], ColumnTransformer]:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    x_train = preprocessor.fit_transform(x_train_df).astype(np.float32)
    x_val = preprocessor.transform(x_val_df).astype(np.float32)
    x_test = preprocessor.transform(x_test_df).astype(np.float32)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return x_train, x_val, x_test, feature_names, preprocessor


def _prepare_nsl_kdd(dataset_config: dict, run_dir: Path) -> PreparedData:
    train_path, test_path = download_nsl_kdd(Path(dataset_config["raw_dir"]))
    train_df = _read_nsl_kdd(train_path)
    test_df = _read_nsl_kdd(test_path)

    task = dataset_config.get("task", "binary")
    if task == "multiclass":
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
    numeric_cols = [column for column in x_full_df.columns if column not in categorical_cols]

    x_train_df, x_val_df, y_train, y_val = train_test_split(
        x_full_df,
        y_full,
        test_size=dataset_config.get("test_size", 0.2),
        random_state=42,
        stratify=y_full,
    )

    x_train, x_val, x_test, feature_names, preprocessor = _fit_and_transform(
        x_train_df=x_train_df,
        x_val_df=x_val_df,
        x_test_df=x_test_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    _save_preprocessor_artifacts(run_dir, preprocessor, feature_names)
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


def _load_folktables_problem(problem_name: str):
    try:
        from folktables import ACSIncome, ACSEmployment, ACSMobility, ACSPublicCoverage, ACSTravelTime
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "folktables is not installed. Install project dependencies to use dataset.name=folktables."
        ) from exc

    problems = {
        "ACSIncome": ACSIncome,
        "ACSEmployment": ACSEmployment,
        "ACSMobility": ACSMobility,
        "ACSPublicCoverage": ACSPublicCoverage,
        "ACSTravelTime": ACSTravelTime,
    }
    if problem_name not in problems:
        available = ", ".join(sorted(problems))
        raise ValueError(f"Unsupported folktables problem {problem_name!r}. Available: {available}")
    return problems[problem_name]


def _normalize_states(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    states = [str(item) for item in value]
    return states or None


def _folktables_xy(problem, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features, target, _group = problem.df_to_pandas(frame)
    features = features.reset_index(drop=True)
    target_series = pd.Series(target.squeeze()).reset_index(drop=True)
    return features, target_series.astype(np.int64).to_numpy()


def _prepare_folktables(dataset_config: dict, run_dir: Path) -> PreparedData:
    try:
        from folktables import ACSDataSource
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "folktables is not installed. Install project dependencies to use dataset.name=folktables."
        ) from exc

    problem = _load_folktables_problem(str(dataset_config.get("problem", "ACSIncome")))
    data_source = ACSDataSource(
        survey_year=str(dataset_config.get("survey_year", "2018")),
        horizon=str(dataset_config.get("horizon", "1-Year")),
        survey=str(dataset_config.get("survey", "person")),
        root_dir=str(Path(dataset_config["raw_dir"])),
    )
    train_states = _normalize_states(dataset_config.get("train_states", dataset_config.get("states")))
    test_states = _normalize_states(dataset_config.get("test_states"))
    density = float(dataset_config.get("density", 1.0))
    random_seed = int(dataset_config.get("download_random_seed", 0))
    download = bool(dataset_config.get("download", True))

    train_frame = data_source.get_data(
        states=train_states,
        density=density,
        random_seed=random_seed,
        download=download,
    )
    x_full_df, y_full = _folktables_xy(problem, train_frame)

    if test_states:
        test_frame = data_source.get_data(
            states=test_states,
            density=density,
            random_seed=random_seed,
            download=download,
        )
        x_test_df, y_test = _folktables_xy(problem, test_frame)
    else:
        x_trainval_df, x_test_df, y_trainval, y_test = train_test_split(
            x_full_df,
            y_full,
            test_size=dataset_config.get("test_size", 0.2),
            random_state=42,
            stratify=y_full,
        )
        x_full_df = x_trainval_df.reset_index(drop=True)
        y_full = np.asarray(y_trainval, dtype=np.int64)

    x_train_df, x_val_df, y_train, y_val = train_test_split(
        x_full_df,
        y_full,
        test_size=dataset_config.get("validation_size", dataset_config.get("test_size", 0.2)),
        random_state=42,
        stratify=y_full,
    )

    numeric_cols = x_train_df.columns.tolist()
    categorical_cols: list[str] = []
    x_train, x_val, x_test, feature_names, preprocessor = _fit_and_transform(
        x_train_df=x_train_df,
        x_val_df=x_val_df,
        x_test_df=x_test_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    _save_preprocessor_artifacts(run_dir, preprocessor, feature_names)
    return PreparedData(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=np.asarray(y_train, dtype=np.int64),
        y_val=np.asarray(y_val, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        feature_names=feature_names,
        class_names=["negative", "positive"],
        preprocessor=preprocessor,
    )


def prepare_dataset(config: dict, run_dir: Path) -> PreparedData:
    dataset_config = config["dataset"]
    dataset_name = str(dataset_config["name"])
    if dataset_name == "nsl_kdd":
        return _prepare_nsl_kdd(dataset_config, run_dir)
    if dataset_name == "folktables":
        return _prepare_folktables(dataset_config, run_dir)
    raise ValueError(f"Unsupported dataset.name={dataset_name!r}")


def stratified_subset_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    indices: list[np.ndarray] = []
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        size = max(1, int(round(len(label_indices) * fraction)))
        indices.append(rng.choice(label_indices, size=size, replace=False))
    result = np.concatenate(indices)
    rng.shuffle(result)
    return result
