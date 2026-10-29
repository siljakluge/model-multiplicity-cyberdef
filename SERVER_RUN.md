<!-- Author: Coda -->
# Server full-run checklist

Use Python 3.10+ and an isolated environment:

```bash
git clone https://github.com/siljakluge/model-multiplicity-cyberdef.git
cd model-multiplicity-cyberdef
git checkout main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

If external access requires a proxy, configure it before the first run; cached
original datasets make later runs offline-capable:

```bash
export HTTPS_PROXY=http://proxy.company.example:8080
export HTTP_PROXY="$HTTPS_PROXY"
export NO_PROXY=localhost,127.0.0.1
```

First validate server networking and CUDA with a smoke run. Do not launch the
full job until it succeeds:

```bash
python -m mmcyber.cli run-all --config configs/adult_smoke.yaml --rashomon-tolerance 1.0 --max-background 8 --max-explain 8 --aggregate-out-dir runs/aggregate_adult_smoke
python -m mmcyber.cli run-all --config configs/bank_marketing_smoke.yaml --rashomon-tolerance 1.0 --max-background 8 --max-explain 8 --aggregate-out-dir runs/aggregate_bank_marketing_smoke
```

## Full experiments

Each command runs origins 1--10, writes one directory per origin, and resumes
safely when rerun unchanged. Use `tmux`, `screen`, or the company scheduler.

```bash
python -m mmcyber.cli run-all --config configs/default.yaml --rashomon-tolerance 0.015 --max-background 128 --max-explain 256 --aggregate-out-dir runs/aggregate_nslkdd
python -m mmcyber.cli run-all --config configs/folktables_income_full.yaml --rashomon-tolerance 0.015 --max-background 128 --max-explain 256 --aggregate-out-dir runs/aggregate_acsincome_ca2018
python -m mmcyber.cli run-all --config configs/folktables_employment_full.yaml --rashomon-tolerance 0.015 --max-background 128 --max-explain 256 --aggregate-out-dir runs/aggregate_acsemployment_ca2018
python -m mmcyber.cli run-all --config configs/adult_full.yaml --rashomon-tolerance 0.015 --max-background 128 --max-explain 256 --aggregate-out-dir runs/aggregate_adult
python -m mmcyber.cli run-all --config configs/bank_marketing_full.yaml --rashomon-tolerance 0.015 --max-background 128 --max-explain 256 --aggregate-out-dir runs/aggregate_bank_marketing
```

The unchanged factorized plan is 63 MLPs/origin: 630 models/dataset and 3,150
training runs across these five datasets. `device: auto` selects CUDA when
available. Rashomon membership always uses validation accuracy; test accuracy
is only evaluated after selection.

## Cross-dataset backup and aggregation

Keep the per-dataset aggregate outputs and every origin directory. They contain
resolved configs, fingerprints, model caches, predictions, metrics, SHAP caches
and plots. To create a combined descriptive aggregation after all five jobs:

```bash
python -m mmcyber.cli aggregate --run-dirs \
  runs/nslkdd_multiplicity_seed{1,2,3,4,5,6,7,8,9,10} \
  runs/acsincome_ca2018_multiplicity_seed{1,2,3,4,5,6,7,8,9,10} \
  runs/acsemployment_ca2018_multiplicity_seed{1,2,3,4,5,6,7,8,9,10} \
  runs/adult_multiplicity_seed{1,2,3,4,5,6,7,8,9,10} \
  runs/bank_marketing_multiplicity_seed{1,2,3,4,5,6,7,8,9,10} \
  --out-dir runs/aggregate_all_datasets
tar -czf model-multiplicity-results-$(date +%F).tar.gz runs/aggregate_* runs/*_multiplicity_seed*
```

Do not combine datasets for inferential comparisons without explicitly checking
their task definitions and scales; this aggregate is a traceable descriptive
export. Bank Marketing retains `duration` only for retrospective analysis, not
for a prospective pre-call system.
