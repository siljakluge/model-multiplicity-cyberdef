"""Author: Coda. Generate figures for the exploratory Drive-run audit."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = pd.read_csv(ROOT / "preliminary_origin_aggregates.csv")
FIGURES = ROOT / "figures_preliminary"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 180, "savefig.bbox": "tight"})
COLORS = ["#457b9d", "#e76f51", "#2a9d8f"]


def _errorbars(ax, column: str, label: str, percent: bool = True) -> None:
    values = DATA[f"{column}_mean"].to_numpy()
    low = DATA[f"{column}_ci_low"].to_numpy()
    high = DATA[f"{column}_ci_high"].to_numpy()
    if percent:
        values, low, high = values * 100, low * 100, high * 100
    ax.bar(DATA["dataset"], values, color=COLORS, width=0.62)
    ax.errorbar(DATA["dataset"], values, yerr=np.vstack([values - low, high - values]), fmt="none", color="#202020", capsize=4)
    ax.set_ylabel(label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)


fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
_errorbars(axes[0], "decision_disagreement", "Mean pairwise disagreement (%)")
_errorbars(axes[1], "ambiguity", "Ambiguous test instances (%)")
fig.suptitle("Operational multiplicity across independent origins")
fig.savefig(FIGURES / "operational_multiplicity.png")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
_errorbars(axes[0], "shap_cosine_distance", "SHAP cosine distance", percent=False)
_errorbars(axes[1], "top10_jaccard", "Top-10 SHAP overlap", percent=False)
_errorbars(axes[2], "sign_instability", "Attribution sign instability (%)")
fig.suptitle("Explanatory multiplicity across independent origins")
fig.savefig(FIGURES / "explanatory_multiplicity.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.1, 3.6))
values = DATA["rashomon_models_mean"].to_numpy()
low = DATA["rashomon_models_ci_low"].to_numpy()
high = DATA["rashomon_models_ci_high"].to_numpy()
ax.bar(DATA["dataset"], values, color=COLORS, width=0.62)
ax.errorbar(DATA["dataset"], values, yerr=np.vstack([values - low, high - values]), fmt="none", color="#202020", capsize=4)
ax.set_ylabel("Validation-independent candidate models\n(exploratory test-selected set)")
ax.set_title("Stored exploratory Rashomon-set size")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.22)
fig.savefig(FIGURES / "exploratory_rashomon_size.png")
plt.close(fig)
