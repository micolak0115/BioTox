#!/usr/bin/env python3
"""Summarize ridge-penalty selection across repeated nested CV outer folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
DEFAULT_INPUT = Path(CONFIG["input_files"]["repeated_cv_run"])
DEFAULT_OUTPUT = (HERE / CONFIG["output"]["directory"]).resolve()
GRID_LOWER = 1e-4
GRID_UPPER = 1e8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_fold_results(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("repeated_nested_cv_standardized_*__*.csv"))
    paths += sorted(input_dir.glob("repeated_nested_cv_residualized_*__*.csv"))
    if len(paths) != 96:
        raise RuntimeError(f"Expected 96 fold-result files, found {len(paths)}")

    frames = [pd.read_csv(path) for path in paths]
    folds = pd.concat(frames, ignore_index=True)
    if len(folds) != 9_600:
        raise RuntimeError(f"Expected 9,600 outer-fold rows, found {len(folds)}")
    if not folds["final_success"].astype(bool).all():
        raise RuntimeError("At least one final outer-fold fit was unsuccessful")
    return folds


def format_power(value: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return "NA"
    exponent = np.log10(value)
    if np.isclose(exponent, round(exponent), atol=1e-10):
        return f"1e{int(round(exponent)):+d}"
    return f"{value:.3g}"


def summarize_group(group: pd.DataFrame) -> pd.Series:
    selected = group["alpha_star_outer"].astype(float)
    minimum = group["alpha_min_loss"].astype(float)
    log_selected = np.log10(selected)
    counts = selected.value_counts()
    mode = float(counts[counts == counts.max()].index.max())
    quantiles = log_selected.quantile([0.25, 0.50, 0.75])

    return pd.Series(
        {
            "n_outer_folds": len(group),
            "lambda_mode": mode,
            "lambda_log10_q1": quantiles.loc[0.25],
            "lambda_log10_median": quantiles.loc[0.50],
            "lambda_log10_q3": quantiles.loc[0.75],
            "lambda_geometric_median": 10 ** quantiles.loc[0.50],
            "lambda_minimum": selected.min(),
            "lambda_maximum": selected.max(),
            "n_unique_lambda": selected.nunique(),
            "one_se_stronger_n": int((selected > minimum).sum()),
            "one_se_stronger_pct": 100.0 * (selected > minimum).mean(),
            "lower_boundary_n": int(np.isclose(selected, GRID_LOWER).sum()),
            "lower_boundary_pct": 100.0 * np.isclose(selected, GRID_LOWER).mean(),
            "upper_boundary_n": int(np.isclose(selected, GRID_UPPER).sum()),
            "upper_boundary_pct": 100.0 * np.isclose(selected, GRID_UPPER).mean(),
            "saturated_no_bio_signal_n": int(
                group["beta_saturated_no_bio_signal"].astype(bool).sum()
            ),
            "saturated_no_bio_signal_pct": 100.0
            * group["beta_saturated_no_bio_signal"].astype(bool).mean(),
            "all_final_fits_successful": bool(group["final_success"].all()),
        }
    )


def build_summary(folds: pd.DataFrame) -> pd.DataFrame:
    summary = (
        folds.groupby(["variant", "task", "cell"], sort=True, observed=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )
    summary.insert(
        0,
        "family",
        np.where(summary["task"].str.startswith("NR-"), "NR", "SR"),
    )
    summary["lambda_selected_display"] = summary.apply(
        lambda row: (
            f"{format_power(row['lambda_geometric_median'])} "
            f"[{format_power(10 ** row['lambda_log10_q1'])}, "
            f"{format_power(10 ** row['lambda_log10_q3'])}]"
        ),
        axis=1,
    )
    return summary


def build_frequency(folds: pd.DataFrame) -> pd.DataFrame:
    frequency = (
        folds.groupby(
            ["variant", "task", "cell", "alpha_star_outer"],
            sort=True,
            observed=True,
        )
        .size()
        .rename("selection_n")
        .reset_index()
        .rename(columns={"alpha_star_outer": "lambda"})
    )
    frequency["selection_pct"] = frequency["selection_n"]
    frequency["selection_pct"] = 100.0 * frequency["selection_pct"] / 100.0
    return frequency


def build_overall(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in folds.groupby("variant", sort=True):
        rows.append(
            {
                "scope": variant,
                "n_outer_folds": len(group),
                "upper_boundary_n": int(
                    np.isclose(group["alpha_star_outer"], GRID_UPPER).sum()
                ),
                "upper_boundary_pct": 100.0
                * np.isclose(group["alpha_star_outer"], GRID_UPPER).mean(),
                "lower_boundary_n": int(
                    np.isclose(group["alpha_star_outer"], GRID_LOWER).sum()
                ),
                "lower_boundary_pct": 100.0
                * np.isclose(group["alpha_star_outer"], GRID_LOWER).mean(),
                "saturated_no_bio_signal_n": int(
                    group["beta_saturated_no_bio_signal"].astype(bool).sum()
                ),
                "saturated_no_bio_signal_pct": 100.0
                * group["beta_saturated_no_bio_signal"].astype(bool).mean(),
                "one_se_stronger_n": int(
                    (group["alpha_star_outer"] > group["alpha_min_loss"]).sum()
                ),
                "one_se_stronger_pct": 100.0
                * (group["alpha_star_outer"] > group["alpha_min_loss"]).mean(),
            }
        )
    group = folds
    rows.append(
        {
            "scope": "all",
            "n_outer_folds": len(group),
            "upper_boundary_n": int(
                np.isclose(group["alpha_star_outer"], GRID_UPPER).sum()
            ),
            "upper_boundary_pct": 100.0
            * np.isclose(group["alpha_star_outer"], GRID_UPPER).mean(),
            "lower_boundary_n": int(
                np.isclose(group["alpha_star_outer"], GRID_LOWER).sum()
            ),
            "lower_boundary_pct": 100.0
            * np.isclose(group["alpha_star_outer"], GRID_LOWER).mean(),
            "saturated_no_bio_signal_n": int(
                group["beta_saturated_no_bio_signal"].astype(bool).sum()
            ),
            "saturated_no_bio_signal_pct": 100.0
            * group["beta_saturated_no_bio_signal"].astype(bool).mean(),
            "one_se_stronger_n": int(
                (group["alpha_star_outer"] > group["alpha_min_loss"]).sum()
            ),
            "one_se_stronger_pct": 100.0
            * (group["alpha_star_outer"] > group["alpha_min_loss"]).mean(),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    folds = load_fold_results(args.input_dir)
    summary = build_summary(folds)
    frequency = build_frequency(folds)
    overall = build_overall(folds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "lambda_selection_by_analysis.csv", index=False)
    frequency.to_csv(args.output_dir / "lambda_selection_frequency.csv", index=False)
    overall.to_csv(args.output_dir / "lambda_selection_overall.csv", index=False)

    compact = summary[
        [
            "family",
            "task",
            "cell",
            "variant",
            "n_outer_folds",
            "lambda_selected_display",
            "lambda_mode",
            "one_se_stronger_pct",
            "upper_boundary_pct",
            "saturated_no_bio_signal_pct",
        ]
    ]
    compact.to_csv(args.output_dir / "lambda_selection_compact.csv", index=False)


if __name__ == "__main__":
    main()
