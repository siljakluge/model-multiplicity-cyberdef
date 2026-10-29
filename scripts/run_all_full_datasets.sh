#!/usr/bin/env bash
# Author: Coda
# Run from the repository root with the project virtual environment activated.
set -euo pipefail

python -m mmcyber.cli preflight

python -m mmcyber.cli run-all --config configs/default.yaml --rashomon-tolerance 0.015 --max-background 128 --max-explain 256 --aggregate-out-dir runs/aggregate_nslkdd
python -m mmcyber.cli run-all --config configs/folktables_income_full.yaml --rashomon-tolerance 0.015 --max-background 128 --max-explain 256 --aggregate-out-dir runs/aggregate_acsincome_ca2018
python -m mmcyber.cli run-all --config configs/folktables_employment_full.yaml --rashomon-tolerance 0.015 --max-background 128 --max-explain 256 --aggregate-out-dir runs/aggregate_acsemployment_ca2018
python -m mmcyber.cli run-all --config configs/adult_full.yaml --rashomon-tolerance 0.015 --max-background 128 --max-explain 256 --aggregate-out-dir runs/aggregate_adult
python -m mmcyber.cli run-all --config configs/bank_marketing_full.yaml --rashomon-tolerance 0.015 --max-background 128 --max-explain 256 --aggregate-out-dir runs/aggregate_bank_marketing
