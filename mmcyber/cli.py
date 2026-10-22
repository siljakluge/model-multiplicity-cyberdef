from __future__ import annotations

import argparse

from mmcyber.disagreement import compute_disagreement
from mmcyber.explanation_disagreement import compute_explanation_disagreement
from mmcyber.aggregate import aggregate_runs
from mmcyber.reevaluate import reconstruct_validation_metrics
from mmcyber.plotting import plot_all
from mmcyber.shap_analysis import compute_shap
from mmcyber.train import run_training
from mmcyber.utils import load_config


def _parse_hidden_dims_variants(values: list[str] | None) -> list[list[int]] | None:
    if not values:
        return None
    variants = []
    for value in values:
        dims = [int(part) for part in value.split(",") if part.strip()]
        if not dims:
            raise ValueError(f"Invalid hidden-dims variant {value!r}")
        variants.append(dims)
    return variants


def _parse_activation_variants(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    return [value.strip() for value in values if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="mmcyber")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--seeds", nargs="*", type=int)
    train_parser.add_argument("--subset-fractions", nargs="*", type=float)
    train_parser.add_argument("--hidden-dims-variants", nargs="*")
    train_parser.add_argument("--activation-variants", nargs="*")
    train_parser.add_argument("--origin-seed", type=int)

    disagree_parser = subparsers.add_parser("disagree")
    disagree_parser.add_argument("--run-dir", required=True)
    disagree_parser.add_argument("--rashomon-tolerance", type=float, default=0.015)
    disagree_parser.add_argument("--rashomon-metric", default="val_accuracy")

    shap_parser = subparsers.add_parser("shap")
    shap_parser.add_argument("--run-dir", required=True)
    shap_parser.add_argument("--max-background", type=int, default=128)
    shap_parser.add_argument("--max-explain", type=int, default=256)
    shap_parser.add_argument("--only-conflicts", action="store_true")

    explain_disagree_parser = subparsers.add_parser("explain-disagree")
    explain_disagree_parser.add_argument("--run-dir", required=True)
    explain_disagree_parser.add_argument("--top-k", type=int, default=20)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--run-dir", required=True)
    plot_parser.add_argument("--out-dir")
    plot_parser.add_argument("--top-n", type=int, default=20)
    plot_parser.add_argument("--paper-dir")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--run-dirs", nargs="+", required=True)
    aggregate_parser.add_argument("--out-dir", required=True)
    aggregate_parser.add_argument("--n-bootstrap", type=int, default=10_000)
    aggregate_parser.add_argument("--seed", type=int, default=42)

    reevaluate_parser = subparsers.add_parser("reevaluate")
    reevaluate_parser.add_argument("--run-dir", required=True)
    reevaluate_parser.add_argument("--batch-size", type=int, default=1024)
    reevaluate_parser.add_argument("--device", default="auto")

    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        run_training(
            config,
            seeds=args.seeds,
            subset_fractions=args.subset_fractions,
            hidden_dims_variants=_parse_hidden_dims_variants(args.hidden_dims_variants),
            activation_variants=_parse_activation_variants(args.activation_variants),
            origin_seed=args.origin_seed,
        )
    elif args.command == "disagree":
        compute_disagreement(
            args.run_dir,
            rashomon_tolerance=args.rashomon_tolerance,
            rashomon_metric=args.rashomon_metric,
        )
    elif args.command == "shap":
        compute_shap(
            args.run_dir,
            max_background=args.max_background,
            max_explain=args.max_explain,
            only_conflicts=args.only_conflicts,
        )
    elif args.command == "explain-disagree":
        compute_explanation_disagreement(args.run_dir, top_k=args.top_k)
    elif args.command == "plot":
        plot_all(args.run_dir, out_dir=args.out_dir, top_n=args.top_n, paper_dir=args.paper_dir)
    elif args.command == "aggregate":
        aggregate_runs(args.run_dirs, args.out_dir, n_bootstrap=args.n_bootstrap, seed=args.seed)
    elif args.command == "reevaluate":
        reconstruct_validation_metrics(args.run_dir, batch_size=args.batch_size, device_name=args.device)


if __name__ == "__main__":
    main()
