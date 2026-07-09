from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    return plt, sns


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _model_order(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["model_id"].unique())


def plot_metrics(run_dir: str | Path, out_dir: str | Path | None = None) -> None:
    plt, sns = _setup_matplotlib()
    run_path = Path(run_dir)
    out_path = Path(out_dir) if out_dir else run_path / "plots"
    metrics = pd.read_csv(run_path / "metrics.csv")

    long = metrics.melt(
        id_vars=["model_id", "seed", "subset_fraction"],
        value_vars=["accuracy", "macro_f1", "log_loss"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=long, x="model_id", y="value", hue="metric", ax=ax)
    ax.set_title("Model-Level Performance")
    ax.set_xlabel("model")
    ax.set_ylabel("metric value")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, out_path / "metrics_by_model.png")
    plt.close(fig)

    if metrics["subset_fraction"].nunique() > 1 or metrics["seed"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.lineplot(
            data=metrics,
            x="subset_fraction",
            y="macro_f1",
            hue="seed",
            marker="o",
            ax=ax,
        )
        ax.set_title("Macro-F1 Across Training Subsets")
        ax.set_xlabel("training subset fraction")
        ax.set_ylabel("macro-F1")
        _save(fig, out_path / "macro_f1_by_subset.png")
        plt.close(fig)


def plot_decision_disagreement(run_dir: str | Path, out_dir: str | Path | None = None) -> None:
    plt, sns = _setup_matplotlib()
    run_path = Path(run_dir)
    out_path = Path(out_dir) if out_dir else run_path / "plots"
    predictions = pd.read_csv(run_path / "test_predictions.csv")
    pairwise = pd.read_csv(run_path / "disagreement_summary.csv")
    sample = pd.read_csv(run_path / "sample_disagreement.csv")

    pivot = predictions.pivot(index="sample_id", columns="model_id", values="y_pred")
    models = list(pivot.columns)
    matrix = pd.DataFrame(np.eye(len(models)), index=models, columns=models)
    for row in pairwise.itertuples(index=False):
        matrix.loc[row.model_a, row.model_b] = row.disagreement_rate
        matrix.loc[row.model_b, row.model_a] = row.disagreement_rate

    fig, ax = plt.subplots(figsize=(max(5, len(models) * 0.7), max(4, len(models) * 0.6)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="mako_r", vmin=0, vmax=max(0.1, matrix.to_numpy().max()), ax=ax)
    ax.set_title("Pairwise Prediction Disagreement")
    _save(fig, out_path / "decision_disagreement_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(data=sample, x="vote_entropy", bins=30, ax=ax)
    ax.set_title("Per-Sample Vote Entropy")
    ax.set_xlabel("vote entropy")
    ax.set_ylabel("sample count")
    _save(fig, out_path / "sample_vote_entropy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(data=sample, x="majority_fraction", bins=30, ax=ax)
    ax.set_title("Majority Vote Strength")
    ax.set_xlabel("majority fraction")
    ax.set_ylabel("sample count")
    _save(fig, out_path / "majority_fraction.png")
    plt.close(fig)

    if "conflict_ratio" in sample.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(data=sample, x="conflict_ratio", bins=30, ax=ax)
        ax.set_title("Per-Sample Conflict Ratio")
        ax.set_xlabel("conflict ratio")
        ax.set_ylabel("sample count")
        _save(fig, out_path / "conflict_ratio.png")
        plt.close(fig)

    summary_path = run_path / "multiplicity_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        value_vars = [
            "ambiguity",
            "mean_conflict_ratio",
            "max_conflict_ratio",
            "mean_pairwise_disagreement",
            "max_pairwise_disagreement",
        ]
        plot_frame = summary.melt(value_vars=value_vars, var_name="metric", value_name="value")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=plot_frame, x="metric", y="value", ax=ax)
        ax.set_title("Multiplicity Summary")
        ax.set_xlabel("metric")
        ax.set_ylabel("value")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, out_path / "multiplicity_summary.png")
        plt.close(fig)


def plot_shap(run_dir: str | Path, out_dir: str | Path | None = None, top_n: int = 20) -> None:
    plt, sns = _setup_matplotlib()
    run_path = Path(run_dir)
    out_path = Path(out_dir) if out_dir else run_path / "plots"
    shap_summary = pd.read_csv(run_path / "shap_summary.csv")

    for class_name, class_frame in shap_summary.groupby("class_name"):
        top_features = (
            class_frame.groupby("feature")["mean_abs_shap"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        plot_frame = class_frame[class_frame["feature"].isin(top_features)].copy()
        order = top_features[::-1]
        fig, ax = plt.subplots(figsize=(8.5, max(5, top_n * 0.28)))
        sns.barplot(data=plot_frame, y="feature", x="mean_abs_shap", hue="model_id", order=order, ax=ax)
        ax.set_title(f"Top SHAP Features: {class_name}")
        ax.set_xlabel("mean absolute SHAP")
        ax.set_ylabel("feature")
        _save(fig, out_path / f"shap_top_features_{class_name}.png")
        plt.close(fig)


def plot_explanation_disagreement(run_dir: str | Path, out_dir: str | Path | None = None) -> None:
    plt, sns = _setup_matplotlib()
    run_path = Path(run_dir)
    out_path = Path(out_dir) if out_dir else run_path / "plots"
    explanation = pd.read_csv(run_path / "explanation_disagreement.csv")

    metrics = [
        ("mean_abs_shap_cosine_distance", "SHAP Cosine Distance"),
        ("top_k_jaccard", "Top-k SHAP Feature Jaccard"),
    ]
    for column, title in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=explanation, x="class_name", y=column, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("class")
        ax.set_ylabel(column.replace("_", " "))
        _save(fig, out_path / f"explanation_{column}.png")
        plt.close(fig)


def plot_shap_variability(run_dir: str | Path, out_dir: str | Path | None = None, top_n: int = 20) -> None:
    plt, sns = _setup_matplotlib()
    run_path = Path(run_dir)
    out_path = Path(out_dir) if out_dir else run_path / "plots"
    variability_path = run_path / "shap_variability.csv"
    correlations_path = run_path / "shap_variability_correlations.csv"
    if not variability_path.exists() or not correlations_path.exists():
        return

    variability = pd.read_csv(variability_path)
    correlations = pd.read_csv(correlations_path)
    for class_name, class_frame in variability.groupby("class_name"):
        top_features = (
            class_frame.groupby("feature")["shap_value_range"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        plot_frame = class_frame[class_frame["feature"].isin(top_features)]
        fig, ax = plt.subplots(figsize=(8.5, max(5, top_n * 0.28)))
        sns.barplot(
            data=plot_frame,
            y="feature",
            x="shap_value_range",
            order=top_features[::-1],
            ax=ax,
        )
        ax.set_title(f"SHAP Value Range Across Models: {class_name}")
        ax.set_xlabel("mean per-sample SHAP value range")
        ax.set_ylabel("feature")
        _save(fig, out_path / f"shap_value_range_{class_name}.png")
        plt.close(fig)

    metrics = [
        "spearman_conflict_vs_shap_range",
        "spearman_conflict_vs_shap_variance",
        "spearman_conflict_vs_sign_instability",
    ]
    for metric in metrics:
        ranked = correlations.reindex(correlations[metric].abs().sort_values(ascending=False).index)
        features = ranked["feature"].drop_duplicates().head(top_n)
        plot_frame = correlations[correlations["feature"].isin(features)]
        matrix = plot_frame.pivot(index="feature", columns="class_name", values=metric)
        fig, ax = plt.subplots(figsize=(8, max(4, len(matrix) * 0.3)))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("class")
        ax.set_ylabel("feature")
        _save(fig, out_path / f"{metric}_heatmap.png")
        plt.close(fig)


def plot_all(run_dir: str | Path, out_dir: str | Path | None = None, top_n: int = 20) -> None:
    plot_metrics(run_dir, out_dir)
    plot_decision_disagreement(run_dir, out_dir)
    plot_shap(run_dir, out_dir, top_n=top_n)
    plot_explanation_disagreement(run_dir, out_dir)
    plot_shap_variability(run_dir, out_dir, top_n=top_n)
