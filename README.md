# Model Multiplicity for Cyber Defense

This is a test for studying model multiplicity in intrusion detection. It trains many neural classifiers on the same task, varies random seeds and training subsets, then measures:

- prediction disagreement, ambiguity, and pointwise conflict ratios between similarly good models
- accuracy/F1 variability
- SHAP explanation variability: value range, variance, sign instability, and correlation with predictive conflict

The default dataset is **NSL-KDD** -> small, public, and fast enough for first experiments. For a stronger study, plan is to swap in UNSW-NB15 or CIC-IDS2017/2018 later.

## Setup

Clone the repository somewhere local on your machine first:

```bash
git clone git@github.com:siljakluge/model-multiplicity-cyberdef.git
cd model-multiplicity-cyberdef
```

If SSH is not set up on that machine, use HTTPS instead:

```bash
git clone https://github.com/siljakluge/model-multiplicity-cyberdef.git
cd model-multiplicity-cyberdef
```

Check your Python version before creating the virtual environment:

```bash
python3 --version
```

The project needs Python 3.10 or newer. If this prints something like `Python 3.8.3`, do not use that interpreter. On macOS, install a newer Python with Homebrew:

```bash
brew install python@3.11
```

Then create a virtual environment with that exact interpreter and install the requirements:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

If you see a warning like `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`, your environment picked a too-new NumPy version for the current PyTorch wheel. Fix it inside the activated venv with:

```bash
python -m pip install "numpy<2" --force-reinstall
python -m pip install -r requirements.txt
python -m mmcyber.cli --help
```

After activation, `python --version` should print Python 3.10 or newer. If it still prints `3.8.3`, delete the venv and recreate it with `python3.11`:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

The commands below use `python -m mmcyber.cli` instead of the shorter `mmcyber` executable. That is a bit more verbose, but it works reliably as long as the virtual environment is active.

Check the installation:

```bash
python -m mmcyber.cli --help
```

## Run a Small Experiment

```bash
python -m mmcyber.cli train --config configs/default.yaml
python -m mmcyber.cli disagree --run-dir runs/nslkdd_multiplicity --rashomon-tolerance 0.015
python -m mmcyber.cli shap --run-dir runs/nslkdd_multiplicity --max-background 128 --max-explain 256 --only-conflicts
python -m mmcyber.cli explain-disagree --run-dir runs/nslkdd_multiplicity --top-k 20
python -m mmcyber.cli plot --run-dir runs/nslkdd_multiplicity --top-n 20
```

For a quicker smoke test:

```bash
python -m mmcyber.cli train --config configs/smoke.yaml
python -m mmcyber.cli disagree --run-dir runs/nslkdd_smoke --rashomon-tolerance 1.0
python -m mmcyber.cli shap --run-dir runs/nslkdd_smoke --max-background 8 --max-explain 8 --only-conflicts
python -m mmcyber.cli explain-disagree --run-dir runs/nslkdd_smoke --top-k 5
python -m mmcyber.cli plot --run-dir runs/nslkdd_smoke --top-n 5
```

The NSL-KDD files are downloaded automatically into `data/raw/nsl-kdd/` on the first run. `data/` and `runs/` are intentionally not committed.

## Outputs

`runs/nslkdd_multiplicity/`

- `models/*.pt`: trained PyTorch models
- `preprocessor.joblib`: fitted preprocessing pipeline
- `test_predictions.csv`: per-model predictions and probabilities
- `metrics.csv`: model-level metrics
- `multiplicity_summary.csv`: Rashomon-set size, ambiguity, mean/max conflict ratio, mean/max pairwise disagreement
- `disagreement_summary.csv`: pairwise disagreement between models in the Rashomon set
- `sample_disagreement.csv`: per-sample entropy, majority vote, conflict ratio, and conflict indicator
- `shap_values/*.npz`: SHAP values per model
- `shap_values_long.csv.gz`: SHAP values in model/sample/class/feature long format
- `shap_summary.csv`: mean absolute SHAP values per feature and model
- `explanation_disagreement.csv`: pairwise SHAP-vector cosine distance and top-k feature overlap
- `shap_variability.csv`: BA-style per-sample/per-feature SHAP range, variance, and sign instability across models
- `shap_variability_correlations.csv`: Spearman correlations between conflict ratio and SHAP instability metrics
- `plots/*.png` and `plots/*.pdf`: publication-friendly quick-look plots

## Plots

`mmcyber plot` creates:

- `metrics_by_model`: accuracy, macro-F1, and log-loss per model
- `macro_f1_by_subset`: subset-size trend when multiple subset fractions exist
- `decision_disagreement_heatmap`: pairwise prediction disagreement
- `multiplicity_summary`: ambiguity/conflict/disagreement overview
- `conflict_ratio`: pointwise predictive multiplicity distribution
- `sample_vote_entropy`: per-sample ensemble uncertainty
- `majority_fraction`: strength of the majority vote
- `shap_top_features_<class>`: top SHAP features per class and model
- `explanation_mean_abs_shap_cosine_distance`: explanation distance by class
- `explanation_top_k_jaccard`: top-k explanation overlap by class
- `shap_value_range_<class>`: mean per-sample SHAP range across models
- `spearman_conflict_vs_*_heatmap`: feature-level correlations between conflict ratio and SHAP instability
- `ba_mean_abs_shap_<class>`: BA-style mean absolute SHAP barplot
- `ba_feature_ranking_<class>`: BA-style feature-rank boxplot across Rashomon models
- `ba_heatmap_sign_instability_<class>`: BA-style sign-instability heatmap over conflict points and features
- `ba_correlations_<class>`: BA-style correlation matrix for conflict ratio, sign instability, SHAP range, and SHAP variance
- `ba_scatter_<class>/ba_range_*` and `ba_scatter_<class>/ba_variance_*`: per-feature conflict-ratio scatterplots colored by sign instability

## Research Notes

Useful knobs:

- `seeds`: changes initialization, data loader order, and subset sampling
- `subset_fractions`: trains models on random fractions of the training set
- `subset_strategy`: currently `stratified`, preserving class balance as far as possible
- `task`: `binary` maps attacks vs normal; `multiclass` keeps attack categories
- `--rashomon-tolerance`: defines which models count as similarly good, using `best(metric) - tolerance`
- `--only-conflicts`: computes SHAP on datapoints with `conflict_ratio > 0`, matching the Bachelorarbeit-style analysis

Started with binary classification. Multiclass works but needs more care when comparing explanations across rare classes.
