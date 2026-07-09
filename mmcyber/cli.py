from __future__ import annotations

import argparse

from mmcyber.disagreement import compute_disagreement
from mmcyber.explanation_disagreement import compute_explanation_disagreement
from mmcyber.plotting import plot_all
from mmcyber.shap_analysis import compute_shap
from mmcyber.train import run_training
from mmcyber.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="mmcyber")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--seeds", nargs="*", type=int)
    train_parser.add_argument("--subset-fractions", nargs="*", type=float)

    disagree_parser = subparsers.add_parser("disagree")
    disagree_parser.add_argument("--run-dir", required=True)
    disagree_parser.add_argument("--rashomon-tolerance", type=float, default=0.015)
    disagree_parser.add_argument("--rashomon-metric", default="accuracy")

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

    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        run_training(config, seeds=args.seeds, subset_fractions=args.subset_fractions)
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
        plot_all(args.run_dir, out_dir=args.out_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
