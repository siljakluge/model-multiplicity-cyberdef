# Model Multiplicity for Cyber Defense

## Server full-run quick start

On a Linux server, clone the repository, create the environment, then use the
single full-run script below. It first runs a non-training preflight which loads
and validates all five datasets (downloading missing official files through the
configured proxy if necessary), checks every 10-origin/63-model plan, and leaves
no output in `runs/`. Only if that succeeds does it launch the five full runs:
10 origins × 63 MLPs × 5 datasets = **3,150 training runs**. CUDA is selected
automatically when available; rerunning the script resumes identical jobs.

```bash
git clone https://github.com/siljakluge/model-multiplicity-cyberdef.git
cd model-multiplicity-cyberdef
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

# Checks NSL-KDD, ACSIncome, ACSEmployment, Adult, and Bank Marketing first;
# then starts all five complete 10-origin runs.
bash scripts/run_all_full_datasets.sh
```

To run only the non-training check, use `python -m mmcyber.cli preflight`.
If the server needs an HTTP(S) proxy, export `HTTP_PROXY`, `HTTPS_PROXY`, and
(where needed) `NO_PROXY` before the preflight. The full five-dataset checklist,
aggregation, and backup commands are in [SERVER_RUN.md](SERVER_RUN.md).

This is a test for studying model multiplicity in intrusion detection. It trains many neural classifiers on the same task, varies random seeds, training data, joint network-size/activation choices, and training-subset fractions, then measures:

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
python -m mmcyber.cli disagree --run-dir runs/nslkdd_multiplicity_seed1 --rashomon-tolerance 0.015
python -m mmcyber.cli shap --run-dir runs/nslkdd_multiplicity_seed1 --max-background 128 --max-explain 256 --only-conflicts
python -m mmcyber.cli explain-disagree --run-dir runs/nslkdd_multiplicity_seed1 --top-k 20
python -m mmcyber.cli plot --run-dir runs/nslkdd_multiplicity_seed1 --top-n 20
```

The default config runs the ten configured `origin_seeds` (1--10). Each origin
gets disjoint reproducible streams for dataset splitting, model training, and
bootstrap resampling; its resolved configuration and `training_plan.csv` record
the resulting seeds. To run only one origin:

```bash
python -m mmcyber.cli train --config configs/default.yaml --origin-seed 7
```

The plot step also exports a stable paper subset into
`paper/Overleaf-Stand/figures/` automatically when that directory exists. To
target a different LaTeX or Overleaf figure folder, pass it explicitly:

```bash
python -m mmcyber.cli plot --run-dir runs/nslkdd_multiplicity_seed1 --top-n 20 --paper-dir /path/to/figures
```

For a quicker smoke test:

```bash
python -m mmcyber.cli train --config configs/smoke.yaml
python -m mmcyber.cli disagree --run-dir runs/nslkdd_smoke_seed1 --rashomon-tolerance 1.0
python -m mmcyber.cli shap --run-dir runs/nslkdd_smoke_seed1 --max-background 8 --max-explain 8 --only-conflicts
python -m mmcyber.cli explain-disagree --run-dir runs/nslkdd_smoke_seed1 --top-k 5
python -m mmcyber.cli plot --run-dir runs/nslkdd_smoke_seed1 --top-n 5
```

To override architecture variants from the command line, pass comma-separated hidden-layer lists:

```bash
python -m mmcyber.cli train --config configs/default.yaml --hidden-dims-variants 64,32 128,64 256,128
```

To override activation variants from the command line:

```bash
python -m mmcyber.cli train --config configs/default.yaml --activation-variants relu tanh leaky_relu
```

The NSL-KDD files are downloaded automatically into `data/raw/nsl-kdd/` on the first run. `data/` and `runs/` are intentionally not committed.

## Outputs

`runs/nslkdd_multiplicity_seed1/`

- `models/*.pt`: trained PyTorch models
- `preprocessor.joblib`: fitted preprocessing pipeline
- `training_plan.csv`: explicit per-model experiment plan anchored to the origin seed
- `test_predictions.csv`: per-model predictions and probabilities
- `metrics.csv`: model-level metrics
- `disagreement_by_factor.csv`: pairwise disagreement aggregated by factor blocks such as `seed`, `bootstrap`, `architecture_activation`, or `combined`
- `rashomon_by_source.csv`: counts and fractions of Rashomon-set models per multiplicity source
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
- `paper/Overleaf-Stand/figures/results_*.pdf`: auto-exported paper figures with stable filenames

## Plots

`mmcyber plot` creates:

- `metrics_by_model`: accuracy, macro-F1, and log-loss per model
- `macro_f1_by_subset`: subset-size trend when multiple subset fractions exist
- `macro_f1_by_architecture`: average macro-F1 trend across network-size variants
- `macro_f1_by_activation`: average macro-F1 trend across activation variants
- `decision_disagreement_heatmap`: pairwise prediction disagreement
- `disagreement_by_factor`: direct comparison of how much `seed`, `bootstrap resampling`, joint `network size + activation`,  contribute to disagreement
- `rashomon_by_source`: how many models per multiplicity source actually ended up in the Rashomon set
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

The paper export keeps a curated subset under fixed names such as
`results_disagreement_by_factor.pdf`, `results_macro_f1_by_activation.pdf`, and
`results_feature_ranking_attack.pdf`, so the figures can be dropped into the
paper without manual renaming.

## Research Notes

Useful knobs:

- `seeds`: changes initialization, data loader order, and subset sampling
- `origin_seed`: stable anchor for naming a single run; `origin_seeds` selects the full repeated experiment
- `seed_schedule: origin_disjoint`: derives non-overlapping split, training, and bootstrap seed streams for each origin
- `comparison_seeds`: local comparison indices; with `origin_disjoint` they are mapped to origin-specific seeds
- `subset_fractions`: trains models on random fractions of the training set
- `factorized_subset_fraction`: training fraction used by the factorized design
- `hidden_dims_variants`: trains multiple network-size variants within the same experiment
- `architecture_variants`: defines the network sizes crossed with all activation variants in the joint architecture-activation block
- `activation_variants`: defines the activation functions crossed with all architecture variants in the joint architecture-activation block
- `subset_strategy`: `stratified` for ordinary subsets; bootstrap resampling is used automatically in the bootstrap comparison block
- `task`: `binary` maps attacks vs normal; `multiclass` keeps attack categories
- `--rashomon-tolerance`: defines which models count as similarly good, using `best(metric) - tolerance`
- `--only-conflicts`: computes SHAP on datapoints with `conflict_ratio > 0`, matching the Bachelorarbeit-style analysis

Started with binary classification. Multiclass works but needs more care when comparing explanations across rare classes.

## Reliable MLP-only full runs

XGBoost is disabled and is not installed as a project dependency. The
`factorized` experiment now includes a controlled training-subset block
and gives every origin the same number of comparison seeds. The default
configuration specifies **63 MLP models per origin**, or **630 across ten origins**.

Use a fresh output directory (edit `run_dir` in your config) when changing
training settings. The runner refuses to mix incompatible configurations.
After interruptions, repeating the identical command reuses completed
per-model training and SHAP artifacts. Model result caches live in
`model_results/*.joblib`, SHAP worker caches in `shap_results/*.joblib`.

```bash
python -m mmcyber.cli run-all --config configs/default.yaml \
  --origin-seeds 1 --rashomon-tolerance 0.015 \
  --max-background 32 --max-explain 32 \
  --aggregate-out-dir runs/aggregate_mlp
```

Test one origin first. Then remove `--origin-seeds 1` and adjust the SHAP
budget to run all origins. `--only-conflicts` requires at least one conflict;
it does not silently fall back to non-conflict samples. For an isolated
smoke run, use `configs/smoke.yaml` and the small SHAP budgets from above.

Run offline unit tests using `python -m unittest discover -s tests -v`.

## Supported datasets and data cache

The supported dataset identifiers are `nsl_kdd`, `folktables` (including
`ACSIncome` and `ACSEmployment`), `adult`, and `bank_marketing`. Adult uses
the official UCI [adult.data and adult.test](https://archive.ics.uci.edu/dataset/2/adult)
files and preserves their official train/test split; validation is a stratified
split of training data. Bank Marketing uses the official UCI
[bank.zip](https://archive.ics.uci.edu/dataset/222/bank+marketing) archive and
extracts only `bank-full.csv` (45,211 rows), never the 10% sample or
`bank-additional-full.csv`. It makes a reproducible 64/16/20 train/validation/test split.

First use downloads and caches originals under `data/raw/adult/` and
`data/raw/bank-marketing/`; reruns work offline when those files exist. Python's
standard `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` variables are honoured.
If a download fails, set the proxy correctly or copy the untouched official file
into that cache path; TLS validation is never disabled.

Missing numeric values are median-imputed and scaled; missing categorical values
are most-frequent-imputed and one-hot encoded. Transformers are fitted only on
the training partition. `unknown` is a legitimate Bank Marketing category and
is retained rather than silently replaced. Bank's `duration` is retained only
for the stated retrospective multiplicity analysis: it is known after a call
and therefore would be leakage for a prospective pre-call deployment model.

## Adult and Bank Marketing commands

Run a one-origin end-to-end smoke test (including SHAP and aggregation):

```bash
python -m mmcyber.cli run-all --config configs/adult_smoke.yaml --rashomon-tolerance 1.0 --max-background 8 --max-explain 8 --aggregate-out-dir runs/aggregate_adult_smoke
python -m mmcyber.cli run-all --config configs/bank_marketing_smoke.yaml --rashomon-tolerance 1.0 --max-background 8 --max-explain 8 --aggregate-out-dir runs/aggregate_bank_marketing_smoke
```

For the full ten-origin experiments, use `configs/adult_full.yaml` and
`configs/bank_marketing_full.yaml`. Both retain the factorized MLP-only design
(63 models/origin), origin-disjoint seed streams, 20 epochs, patience 5, and
validation-accuracy Rashomon selection (`0.015`). Each origin writes into its
own `runs/<dataset>_multiplicity_seed<origin>/` directory. Repeat an identical
command after interruption to resume validated model and SHAP caches; use a new
`run_dir` whenever the configuration or data changes.
