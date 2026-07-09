from __future__ import annotations

from pathlib import Path
import re

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


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "value"


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(x.rank().corr(y.rank()))


def _feature_order_by_mean_abs_shap(shap_values: pd.DataFrame, class_name: str, top_n: int | None = None) -> list[str]:
    class_values = shap_values[shap_values["class_name"] == class_name]
    order = (
        class_values.assign(abs_shap=class_values["shap_value"].abs())
        .groupby("feature")["abs_shap"]
        .mean()
        .sort_values(ascending=False)
    )
    if top_n is not None:
        order = order.head(top_n)
    return order.index.tolist()


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
    import matplotlib.colors as mcolors

    run_path = Path(run_dir)
    out_path = Path(out_dir) if out_dir else run_path / "plots"
    variability_path = run_path / "shap_variability.csv"
    correlations_path = run_path / "shap_variability_correlations.csv"
    shap_values_path = run_path / "shap_values_long.csv.gz"
    if not variability_path.exists() or not correlations_path.exists():
        return

    variability = pd.read_csv(variability_path)
    correlations = pd.read_csv(correlations_path)
    shap_values = pd.read_csv(shap_values_path) if shap_values_path.exists() else None

    if shap_values is not None:
        plot_ba_feature_rankings(shap_values, out_path, plt, sns, top_n=top_n)
        plot_ba_mean_abs_shap(shap_values, out_path, plt, sns, top_n=top_n)
        plot_ba_sign_instability(variability, shap_values, out_path, plt, top_n=top_n)
        plot_ba_range_variance_scatter(variability, shap_values, out_path, plt, mcolors, top_n=top_n)
        plot_ba_correlation_matrices(variability, shap_values, out_path, plt, sns, top_n=top_n)

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


def plot_ba_mean_abs_shap(
    shap_values: pd.DataFrame,
    out_path: Path,
    plt,
    sns,
    top_n: int = 20,
) -> None:
    for class_name, class_values in shap_values.groupby("class_name"):
        plot_frame = (
            class_values.assign(abs_shap=class_values["shap_value"].abs())
            .groupby("feature", as_index=False)["abs_shap"]
            .mean()
            .rename(columns={"abs_shap": "mean_abs_shap"})
            .sort_values("mean_abs_shap", ascending=False)
            .head(top_n)
        )
        fig, ax = plt.subplots(figsize=(max(7, len(plot_frame) * 0.42), 4.8))
        sns.barplot(data=plot_frame, x="feature", y="mean_abs_shap", color="#4C78A8", ax=ax)
        ax.set_title(f"{class_name}: Mean Absolute SHAP Values")
        ax.set_xlabel("feature")
        ax.set_ylabel("mean absolute SHAP value")
        ax.tick_params(axis="x", rotation=90)
        _save(fig, out_path / f"ba_mean_abs_shap_{_safe_name(class_name)}.png")
        plt.close(fig)


def plot_ba_feature_rankings(
    shap_values: pd.DataFrame,
    out_path: Path,
    plt,
    sns,
    top_n: int = 20,
) -> None:
    for class_name, class_values in shap_values.groupby("class_name"):
        model_feature = (
            class_values.assign(abs_shap=class_values["shap_value"].abs())
            .groupby(["model_id", "feature"])["abs_shap"]
            .mean()
            .unstack("feature", fill_value=0.0)
        )
        if model_feature.empty:
            continue
        feature_order = model_feature.mean(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
        ranks = model_feature[feature_order].rank(axis=1, method="first", ascending=False).astype(int)
        plot_frame = ranks.reset_index().melt(id_vars="model_id", var_name="feature", value_name="rank")
        order = plot_frame.groupby("feature")["rank"].mean().sort_values().index

        fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.48), 5.2))
        sns.boxplot(data=plot_frame, x="feature", y="rank", order=order, width=0.7, showfliers=False, ax=ax)
        sns.stripplot(
            data=plot_frame,
            x="feature",
            y="rank",
            order=order,
            color="black",
            size=3,
            alpha=0.55,
            ax=ax,
        )
        ax.set_yticks(np.arange(1, len(order) + 1, 1))
        ax.invert_yaxis()
        ax.set_title(f"{class_name}: Distribution of Feature Rankings")
        ax.set_xlabel("feature")
        ax.set_ylabel("rank (1 = most important)")
        ax.tick_params(axis="x", rotation=90)
        _save(fig, out_path / f"ba_feature_ranking_{_safe_name(class_name)}.png")
        plt.close(fig)


def plot_ba_sign_instability(
    variability: pd.DataFrame,
    shap_values: pd.DataFrame,
    out_path: Path,
    plt,
    top_n: int = 20,
) -> None:
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list("sign_instability", ["#FDE725", "#35B779", "#31688E"])
    for class_name, class_values in variability.groupby("class_name"):
        feature_order = _feature_order_by_mean_abs_shap(shap_values, class_name, top_n=top_n)
        heatmap_frame = class_values[class_values["feature"].isin(feature_order)].pivot_table(
            index="sample_id",
            columns="feature",
            values="sign_instability",
            aggfunc="mean",
        )
        heatmap_frame = heatmap_frame.reindex(columns=feature_order).sort_index()
        if heatmap_frame.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(7, 1.0 + 0.45 * len(feature_order)), 8))
        image = ax.imshow(heatmap_frame.to_numpy(), aspect="auto", vmin=0, vmax=0.5, cmap=cmap)
        ax.set_title(f"{class_name}: Sign Instability")
        ax.set_xlabel("features")
        ax.set_ylabel("conflict points")
        ax.set_xticks(np.arange(len(feature_order)))
        ax.set_xticklabels(feature_order, rotation=90)
        step = max(1, len(heatmap_frame) // 25)
        y_positions = np.arange(0, len(heatmap_frame), step)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(i) for i in y_positions])
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("sign instability")
        colorbar.set_ticks([0, 0.25, 0.5])
        colorbar.set_ticklabels(["stable", "medium", "unstable"])
        _save(fig, out_path / f"ba_heatmap_sign_instability_{_safe_name(class_name)}.png")
        plt.close(fig)


def plot_ba_range_variance_scatter(
    variability: pd.DataFrame,
    shap_values: pd.DataFrame,
    out_path: Path,
    plt,
    mcolors,
    top_n: int = 20,
) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list("sign_instability", ["#FDE725", "#35B779", "#31688E"])
    plot_specs = [
        ("shap_value_range", "SHAP explanation range", "ba_range"),
        ("shap_value_variance", "SHAP explanation variance", "ba_variance"),
    ]
    for class_name, class_values in variability.groupby("class_name"):
        feature_order = _feature_order_by_mean_abs_shap(shap_values, class_name, top_n=top_n)
        class_out = out_path / f"ba_scatter_{_safe_name(class_name)}"
        max_values = {
            column: class_values.loc[class_values["feature"].isin(feature_order), column].max()
            for column, _, _ in plot_specs
        }
        for feature_idx, feature in enumerate(feature_order):
            feature_frame = class_values[class_values["feature"] == feature].copy()
            if feature_frame.empty or "conflict_ratio" not in feature_frame:
                continue
            for column, ylabel, prefix in plot_specs:
                corr = _spearman(feature_frame["conflict_ratio"], feature_frame[column])
                fig, ax = plt.subplots(figsize=(6, 4.5))
                scatter = ax.scatter(
                    feature_frame["conflict_ratio"],
                    feature_frame[column],
                    c=feature_frame["sign_instability"],
                    cmap=cmap,
                    norm=mcolors.Normalize(vmin=0, vmax=0.5),
                    alpha=0.65,
                    edgecolor="black",
                    linewidth=0.25,
                )
                ax.set_xlabel("conflict ratio")
                ax.set_ylabel(ylabel)
                ax.set_xlim(0, max(0.5, float(feature_frame["conflict_ratio"].max())))
                ymax = max_values[column]
                if pd.notna(ymax) and ymax > 0:
                    ax.set_ylim(0, float(ymax) * 1.05)
                title = f"{feature}: {ylabel}"
                if np.isfinite(corr):
                    title += f" (Spearman r = {corr:.3f})"
                ax.set_title(title)
                colorbar = fig.colorbar(scatter, ax=ax)
                colorbar.set_label("sign instability")
                colorbar.set_ticks([0, 0.25, 0.5])
                _save(fig, class_out / f"{prefix}_{feature_idx:02d}_{_safe_name(feature)}.png")
                plt.close(fig)


def plot_ba_correlation_matrices(
    variability: pd.DataFrame,
    shap_values: pd.DataFrame,
    out_path: Path,
    plt,
    sns,
    top_n: int = 20,
) -> None:
    metrics = [
        ("r_conflict_sign", "r(conflict,sign)"),
        ("r_var_sign", "r(var,sign)"),
        ("r_range_sign", "r(range,sign)"),
        ("r_conflict_var", "r(conflict,var)"),
        ("r_conflict_range", "r(conflict,range)"),
        ("r_var_range", "r(var,range)"),
    ]
    for class_name, class_values in variability.groupby("class_name"):
        feature_order = _feature_order_by_mean_abs_shap(shap_values, class_name, top_n=top_n)
        rows = []
        for feature in feature_order:
            feature_frame = class_values[class_values["feature"] == feature]
            rows.append(
                {
                    "feature": feature,
                    "r_conflict_sign": _spearman(feature_frame["conflict_ratio"], feature_frame["sign_instability"]),
                    "r_var_sign": _spearman(feature_frame["shap_value_variance"], feature_frame["sign_instability"]),
                    "r_range_sign": _spearman(feature_frame["shap_value_range"], feature_frame["sign_instability"]),
                    "r_conflict_var": _spearman(feature_frame["conflict_ratio"], feature_frame["shap_value_variance"]),
                    "r_conflict_range": _spearman(feature_frame["conflict_ratio"], feature_frame["shap_value_range"]),
                    "r_var_range": _spearman(feature_frame["shap_value_variance"], feature_frame["shap_value_range"]),
                }
            )
        matrix = pd.DataFrame(rows).set_index("feature")
        if matrix.empty:
            continue
        matrix = matrix[[column for column, _label in metrics]]
        matrix.columns = [label for _column, label in metrics]
        fig, ax = plt.subplots(figsize=(12, max(4.8, 0.35 * len(matrix))))
        sns.heatmap(
            matrix.abs(),
            vmin=0,
            vmax=1,
            cmap="YlGn",
            annot=matrix,
            fmt=".3f",
            ax=ax,
        )
        colorbar = ax.collections[0].colorbar
        colorbar.set_label("absolute correlation strength")
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_title(f"{class_name}: Correlation Matrix", pad=40)
        ax.set_xlabel("")
        ax.set_ylabel("feature")
        _save(fig, out_path / f"ba_correlations_{_safe_name(class_name)}.png")
        plt.close(fig)


def plot_all(run_dir: str | Path, out_dir: str | Path | None = None, top_n: int = 20) -> None:
    plot_metrics(run_dir, out_dir)
    plot_decision_disagreement(run_dir, out_dir)
    plot_shap(run_dir, out_dir, top_n=top_n)
    plot_explanation_disagreement(run_dir, out_dir)
    plot_shap_variability(run_dir, out_dir, top_n=top_n)
