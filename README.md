# Model Multiplicity for Cyber Defense

This is a small research scaffold for studying model multiplicity in intrusion detection. It trains many neural classifiers on the same task, varies random seeds and training subsets, then measures:

- prediction disagreement between models
- accuracy/F1 variability
- SHAP explanation variability

The default dataset is **NSL-KDD** -> small, public, and fast enough for first experiments. For a stronger study, plan is to swap in UNSW-NB15 or CIC-IDS2017/2018 later.

## Setup

```bash
cd /data/.openclaw/workspace/model-multiplicity-cyber
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run a Small Experiment

```bash
mmcyber train --config configs/default.yaml
mmcyber disagree --run-dir runs/nslkdd_multiplicity
mmcyber shap --run-dir runs/nslkdd_multiplicity --max-background 128 --max-explain 256
mmcyber explain-disagree --run-dir runs/nslkdd_multiplicity --top-k 20
mmcyber plot --run-dir runs/nslkdd_multiplicity --top-n 20
```

For a quicker smoke test:

```bash
mmcyber train --config configs/default.yaml --seeds 1 2 --subset-fractions 0.4
```

## Outputs

`runs/nslkdd_multiplicity/`

- `models/*.pt`: trained PyTorch models
- `preprocessor.joblib`: fitted preprocessing pipeline
- `test_predictions.csv`: per-model predictions and probabilities
- `metrics.csv`: model-level metrics
- `disagreement_summary.csv`: pairwise disagreement matrix in long form
- `sample_disagreement.csv`: per-sample entropy/vote statistics
- `shap_values/*.npz`: SHAP values per model
- `shap_summary.csv`: mean absolute SHAP values per feature and model
- `explanation_disagreement.csv`: pairwise SHAP-vector cosine distance and top-k feature overlap
- `plots/*.png` and `plots/*.pdf`: publication-friendly quick-look plots

## Plots

`mmcyber plot` creates:

- `metrics_by_model`: accuracy, macro-F1, and log-loss per model
- `macro_f1_by_subset`: subset-size trend when multiple subset fractions exist
- `decision_disagreement_heatmap`: pairwise prediction disagreement
- `sample_vote_entropy`: per-sample ensemble uncertainty
- `majority_fraction`: strength of the majority vote
- `shap_top_features_<class>`: top SHAP features per class and model
- `explanation_mean_abs_shap_cosine_distance`: explanation distance by class
- `explanation_top_k_jaccard`: top-k explanation overlap by class

## Research Notes

Useful knobs:

- `seeds`: changes initialization, data loader order, and subset sampling
- `subset_fractions`: trains models on random fractions of the training set
- `subset_strategy`: currently `stratified`, preserving class balance as far as possible
- `task`: `binary` maps attacks vs normal; `multiclass` keeps attack categories

Started with binary classification. Multiclass works but needs more care when comparing explanations across rare classes.
