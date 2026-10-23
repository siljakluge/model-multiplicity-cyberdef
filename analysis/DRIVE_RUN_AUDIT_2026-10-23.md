<!-- Author: Coda -->
# Drive run audit — 2026-10-23

## Scope and status

The shared Drive sweep contains repeated origin runs for Adult, Bank Marketing,
NSL-KDD, and two Folktables tasks (ACSIncome/CA and ACSEmployment/CA). The
manifest records ten origins per task and the full train, disagreement, SHAP,
explanation-disagreement, and plot stages for the NSL-KDD runs.

This document reports **exploratory descriptive results only**. The existing
artifacts define Rashomon membership using `accuracy` in
`multiplicity_summary.csv`; the implementation must be audited/corrected so
that this is validation, not final-test, accuracy before any final paper claim.

## Origin-level aggregation

Descriptive unit of aggregation: origin run. Values below are means over ten
origins; model pairs and test points were not used as independent replications.
The reported percentile 95% bootstrap intervals over origin means are
exploratory only: the factorized runs partially overlap because they reuse
comparison seeds and a fixed train/validation split, so the origins are not
independent replication units. They must not be used for conventional
replication-based inference.

| Dataset | Rashomon models | Ambiguity | Mean decision disagreement | Mean SHAP cosine distance | Top-10 SHAP Jaccard | Sign instability |
|---|---:|---:|---:|---:|---:|---:|
| Adult | 58.3 [57.5, 59.1] | 15.35% [15.13, 15.55] | 3.42% [3.39, 3.45] | 0.0720 [0.0676, 0.0762] | 0.686 [0.673, 0.698] | 9.47% [9.29, 9.65] |
| Bank Marketing | 59.1 [58.7, 59.6] | 14.31% [14.14, 14.48] | 3.49% [3.47, 3.52] | 0.0267 [0.0259, 0.0274] | 0.598 [0.592, 0.605] | 16.45% [16.27, 16.66] |
| NSL-KDD | 10.4 [8.7, 12.3] | 7.24% [6.70, 7.81] | 2.60% [2.52, 2.70] | 0.0462 [0.0447, 0.0479] | 0.709 [0.700, 0.718] | 7.45% [7.26, 7.68] |

The point of the result is not a ranking of datasets: all three show that
near-optimal models are non-equivalent in both predictions and explanations.
For example, Bank Marketing has the lowest mean SHAP-cosine distance but the
lowest top-10 feature overlap and highest sign instability. It is therefore
not defensible to reduce explanation stability to a single scalar.

## Evidence to use in a paper

1. **Rashomon-set size and performance:** distribution of selected model count
   and performance gap to the origin's best validation model.
2. **Operational multiplicity:** per-origin mean/max pairwise disagreement,
   ambiguity and conflict ratio. Use the sample-level disagreement file only
   to characterize where disagreement occurs, never as independent evidence.
3. **Explanatory multiplicity:** SHAP cosine distance, top-k Jaccard and sign
   instability, computed exclusively among selected Rashomon models.
4. **Robustness:** threshold-sensitivity curves (available per origin) and a
   replication table across datasets.

## Required remediation before final figures

- Implemented locally on 2026-10-23: training now records validation metrics;
  `mmcyber disagree` defaults to `val_accuracy` and writes an auditable
  `rashomon_models.csv`; SHAP and explanation-disagreement require and filter
  to that exact membership file. Reserve the held-out test split for reported
  performance, disagreement and SHAP estimates.
- Reconstruct validation metrics from the saved model checkpoints and each
  run's `mlp_dataset.joblib`, which stores the exact validation split; the
  implemented `mmcyber reevaluate` command does this without retraining and
  also records recomputed test metrics as an artifact-integrity check. A real
  NSL-KDD checkpoint was successfully reconstructed on 2026-10-23.
- Re-export the two Folktables tasks in a directly readable layout and add
  them to the same origin-level table. Their folders are present in Drive but
  the compact CSV artifacts were not retrievable through the anonymous shared
  endpoint during this audit.
- Version a single analysis script that downloads/reads artifacts and writes
  this table and its figures, so paper numbers cannot drift from the run.
  The local command is now `mmcyber aggregate --run-dirs ... --out-dir ...`;
  it creates origin-level and dataset-level CSVs with bootstrap CIs and rejects
  non-validation-selected runs.

## Recommended paper story

**Similarly accurate intrusion-detection models are not operationally or
explanatorily interchangeable.** Across repeated origins, model choice changes
predictions for roughly 2.6--3.5% of test instances and changes the features
used to justify those decisions. The cyber-specific contribution should focus
on the security-relevant subset of conflicting decisions (attack vs normal,
false positives/negatives), not merely aggregate disagreement.
