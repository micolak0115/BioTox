"""Publication summary for repeated calibrated nested scaffold CV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from stage2.aggregate_offset_logistic import benjamini_hochberg  # noqa: E402
from stage2.repeated_cv_inference import (  # noqa: E402
    repeated_cv_nadeau_bengio_test,
)


TASK_ORDER = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)
CELL_ORDER = ("HA1E", "HEPG2", "HT29", "MCF7")
FAMILY_COLORS = {"NR": "#5B8DB8", "SR": "#D77A7D"}
MODEL_SPEC = "frozen_offset_fixed_calibration_intercept_ridge_biology_v3"


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("Invalid boolean values in repeated-CV fold table")
    return mapped


def _validate_group(
    frame: pd.DataFrame,
    source: Path,
    expected_repeats: int,
    expected_folds: int,
) -> None:
    required = {
        "task",
        "cell",
        "variant",
        "repeat_id",
        "fold_id",
        "outer_fold_id",
        "outer_seed_requested",
        "outer_split_seed",
        "outer_split_signature",
        "chemical_offset_frozen",
        "chemical_offset_coefficient",
        "stage2_model_spec",
        "intercept_penalized",
        "calibration_intercept_frozen",
        "null_intercept",
        "bio_intercept",
        "use_sample_weight",
        "stage2_likelihood",
        "n_outer_train",
        "n_outer_test",
        "auprc_chem",
        "auprc_bio",
        "delta_auprc",
        "auroc_chem",
        "auroc_bio",
        "delta_auroc",
        "log_loss_chem",
        "log_loss_bio",
        "delta_log_loss",
        "final_success",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    if len(frame) != expected_repeats * expected_folds:
        raise ValueError(
            f"{source} must contain {expected_repeats * expected_folds} rows"
        )
    matrix = frame.groupby("repeat_id")["fold_id"].nunique()
    if len(matrix) != expected_repeats or not np.all(matrix == expected_folds):
        raise ValueError(f"{source} has an incomplete repeat-fold matrix")
    if frame.duplicated(["repeat_id", "fold_id"]).any():
        raise ValueError(f"{source} contains duplicate repeat-fold rows")
    if not np.array_equal(frame["fold_id"], frame["outer_fold_id"]):
        raise ValueError(f"{source} has inconsistent outer-fold identifiers")
    signatures = frame.groupby("repeat_id")["outer_split_signature"].nunique()
    if not np.all(signatures == 1):
        raise ValueError(f"{source} has inconsistent split signatures")
    if frame.groupby("repeat_id")["outer_split_signature"].first().duplicated().any():
        raise ValueError(f"{source} reuses an identical outer partition")
    if set(frame["stage2_model_spec"]) != {MODEL_SPEC}:
        raise ValueError(f"{source} has an unexpected Stage-2 specification")
    if not _as_bool(frame["chemical_offset_frozen"]).all():
        raise ValueError(f"{source} contains an unfrozen chemical offset")
    if not np.allclose(frame["chemical_offset_coefficient"], 1.0):
        raise ValueError(f"{source} changes the chemical offset coefficient")
    if _as_bool(frame["intercept_penalized"]).any():
        raise ValueError(f"{source} penalizes the calibration intercept")
    if not _as_bool(frame["calibration_intercept_frozen"]).all():
        raise ValueError(f"{source} refits the calibration intercept during ridge fitting")
    if not np.array_equal(frame["null_intercept"], frame["bio_intercept"]):
        raise ValueError(f"{source} changes the shared calibration intercept")
    if _as_bool(frame["use_sample_weight"]).any():
        raise ValueError(f"{source} is not the unweighted primary fit")
    if set(frame["stage2_likelihood"]) != {"unweighted_publication_primary"}:
        raise ValueError(f"{source} has an unexpected likelihood")
    if not _as_bool(frame["final_success"]).all():
        raise ValueError(f"{source} contains failed fits")


def load_repeated_folds(input_dir: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = input_dir / "RUN_COMPLETE.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing completed-run marker: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_repeats = int(manifest["n_repeats"])
    expected_folds = int(manifest["outer_folds_per_repeat"])
    variants = list(manifest["variants"])
    tasks = list(manifest["tasks"])
    cells = list(manifest["cells"])

    frames = []
    for variant in variants:
        for cell in cells:
            for task in tasks:
                source = (
                    input_dir
                    / f"repeated_nested_cv_{variant}_{task}__{cell}.csv"
                )
                if not source.is_file():
                    raise FileNotFoundError(source)
                frame = pd.read_csv(source)
                _validate_group(
                    frame,
                    source,
                    expected_repeats,
                    expected_folds,
                )
                frames.append(frame)
    folds = pd.concat(frames, ignore_index=True)
    if folds.duplicated(
        ["variant", "task", "cell", "repeat_id", "fold_id"]
    ).any():
        raise ValueError("Duplicate repeated-CV fold rows were found")
    return folds, manifest


def _summarize_group(frame: pd.DataFrame) -> dict:
    row = {
        "variant": frame["variant"].iloc[0],
        "task": frame["task"].iloc[0],
        "cell": frame["cell"].iloc[0],
        "n_repeats": int(frame["repeat_id"].nunique()),
        "outer_folds_per_repeat": int(frame["fold_id"].nunique()),
        "n_paired_outer_draws": int(len(frame)),
        "n_unique_compounds": int(
            round(frame["n_outer_test"].sum() / frame["repeat_id"].nunique())
        ),
        "n_test_evaluations": int(frame["n_outer_test"].sum()),
    }
    for metric in ("auprc", "auroc", "log_loss"):
        nb = repeated_cv_nadeau_bengio_test(
            frame[f"delta_{metric}"].to_numpy(dtype=float),
            frame["n_outer_train"].to_numpy(dtype=float),
            frame["n_outer_test"].to_numpy(dtype=float),
        )
        row.update(
            {
                f"mean_{metric}_chem": float(frame[f"{metric}_chem"].mean()),
                f"sd_{metric}_chem": float(frame[f"{metric}_chem"].std(ddof=1)),
                f"mean_{metric}_bio": float(frame[f"{metric}_bio"].mean()),
                f"sd_{metric}_bio": float(frame[f"{metric}_bio"].std(ddof=1)),
                f"mean_delta_{metric}": nb["mean"],
                f"sd_delta_{metric}": float(
                    frame[f"delta_{metric}"].std(ddof=1)
                ),
                f"nb_{metric}_corrected_se": nb["corrected_se"],
                f"nb_{metric}_ci_low": nb["ci_low"],
                f"nb_{metric}_ci_high": nb["ci_high"],
                f"nb_{metric}_t_stat": nb["t_stat"],
                f"nb_{metric}_df": nb["df"],
                f"nb_{metric}_p_value": nb["p_value"],
                f"nb_{metric}_inference_valid": nb["inference_valid"],
                f"nb_{metric}_variance_degenerate": nb[
                    "variance_degenerate"
                ],
                f"nb_{metric}_correction_factor": nb["correction_factor"],
                f"nb_{metric}_mean_test_train_ratio": nb[
                    "mean_test_train_ratio"
                ],
            }
        )
    return row


def build_summary(folds: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            _summarize_group(group)
            for _, group in folds.groupby(
                ["variant", "task", "cell"],
                sort=False,
            )
        ]
    )
    for metric in ("auprc", "auroc", "log_loss"):
        summary[f"nb_{metric}_q_value"] = np.nan
        for (_, cell), indices in summary.groupby(
            ["variant", "cell"]
        ).groups.items():
            summary.loc[indices, f"nb_{metric}_q_value"] = benjamini_hochberg(
            summary.loc[indices, f"nb_{metric}_p_value"].fillna(1.0)
            )
        summary[f"significant_positive_{metric}_fdr"] = (
            (summary[f"nb_{metric}_q_value"] < 0.05)
            & (summary[f"mean_delta_{metric}"] > 0)
        )
        if metric == "log_loss":
            summary["significant_improvement_log_loss_fdr"] = (
                (summary["nb_log_loss_q_value"] < 0.05)
                & (summary["mean_delta_log_loss"] < 0)
            )
    ranks = {
        "variant": {
            value: index
            for index, value in enumerate(summary["variant"].drop_duplicates())
        },
        "task": {value: index for index, value in enumerate(TASK_ORDER)},
        "cell": {value: index for index, value in enumerate(CELL_ORDER)},
    }
    return (
        summary.assign(
            _variant=summary["variant"].map(ranks["variant"]),
            _task=summary["task"].map(ranks["task"]),
            _cell=summary["cell"].map(ranks["cell"]),
        )
        .sort_values(["_variant", "_task", "_cell"])
        .drop(columns=["_variant", "_task", "_cell"])
        .reset_index(drop=True)
    )


def _write_markdown(summary: pd.DataFrame, destination: Path) -> None:
    table = pd.DataFrame(
        {
            "Endpoint": summary["task"],
            "Cell": summary["cell"],
            "N": summary["n_unique_compounds"],
            "Chem AUPRC": summary["mean_auprc_chem"].map(lambda x: f"{x:.3f}"),
            "BioTox AUPRC": summary["mean_auprc_bio"].map(lambda x: f"{x:.3f}"),
            "Delta AUPRC [NB 95% CI]": summary.apply(
                lambda row: (
                    f"{row.mean_delta_auprc:+.3f} "
                    f"[{row.nb_auprc_ci_low:+.3f}, "
                    f"{row.nb_auprc_ci_high:+.3f}]"
                ),
                axis=1,
            ),
            "Nominal p": summary["nb_auprc_p_value"].map(
                lambda x: f"{x:.3g}"
            ),
            "BH q": summary["nb_auprc_q_value"].map(lambda x: f"{x:.3g}"),
        }
    )
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_forest(summary: pd.DataFrame, output_dir: Path, variant: str) -> None:
    data = summary.loc[summary["variant"] == variant]
    cells = [cell for cell in CELL_ORDER if cell in set(data["cell"])]
    tasks = [task for task in TASK_ORDER if task in set(data["task"])]
    figure, axes = plt.subplots(
        1,
        len(cells),
        figsize=(4.2 * len(cells), 7.0),
        sharey=True,
        squeeze=False,
    )
    for column, cell in enumerate(cells):
        ax = axes[0, column]
        cell_data = data.set_index(["cell", "task"])
        for position, task in enumerate(tasks):
            row = cell_data.loc[(cell, task)]
            color = FAMILY_COLORS[task.split("-", maxsplit=1)[0]]
            mean = float(row["mean_delta_auprc"])
            low = float(row["nb_auprc_ci_low"])
            high = float(row["nb_auprc_ci_high"])
            ax.errorbar(
                mean,
                position,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                markersize=6,
                color=color,
                markeredgecolor="#111111",
                markeredgewidth=0.8,
                ecolor="#222222",
                elinewidth=1.3,
                capsize=3,
                zorder=3,
            )
            ax.text(
                0.98,
                position,
                f"p={row['nb_auprc_p_value']:.3g}; "
                f"q={row['nb_auprc_q_value']:.3g}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=7.3,
                color="#555555",
            )
        ax.axvline(
            0,
            color="#8A8A8A",
            linewidth=1.0,
            linestyle=(0, (2, 3)),
            zorder=0,
        )
        ax.axhline(6.5, color="#B8B8B8", linewidth=0.8)
        ax.set_title(cell, fontweight="bold")
        ax.set_xlabel("Delta AUPRC (BioTox - Chem-only)")
        ax.set_yticks(np.arange(len(tasks)))
        ax.set_yticklabels(tasks)
        ax.set_ylim(len(tasks) - 0.4, -0.6)
        ax.grid(axis="x", color="#E0E0E0", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Tox21 endpoint")
    figure.suptitle(
        "Repeated nested scaffold-CV paired effects",
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.012,
        (
            "Points are mean paired outer-fold effects; whiskers are "
            "Nadeau-Bengio corrected 95% confidence intervals. "
            "Nominal p-values and within-cell BH q-values are shown."
        ),
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.12,
        right=0.99,
        top=0.92,
        bottom=0.10,
        wspace=0.10,
    )
    stem = output_dir / f"figure_repeated_cv_nb_delta_auprc_{variant}"
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(
            stem.with_suffix(suffix),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def summarize(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing repeated-CV summary: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    folds, manifest = load_repeated_folds(input_dir)
    folds.to_csv(output_dir / "repeated_outer_fold_metrics.csv", index=False)
    summary = build_summary(folds)
    summary.to_csv(
        output_dir / "repeated_cv_nadeau_bengio_summary.csv",
        index=False,
    )
    for variant in manifest["variants"]:
        subset = summary.loc[summary["variant"] == variant]
        _write_markdown(
            subset,
            output_dir / f"table_repeated_cv_nb_{variant}.md",
        )
        _plot_forest(summary, output_dir, variant)
    summary_manifest = {
        "source_run": str(input_dir),
        "source_pipeline": manifest["pipeline"],
        "n_repeats": manifest["n_repeats"],
        "outer_folds_per_repeat": manifest["outer_folds_per_repeat"],
        "n_paired_draws_per_task_cell": (
            manifest["n_repeats"] * manifest["outer_folds_per_repeat"]
        ),
        "primary_metric": "AUPRC",
        "secondary_metric": "AUROC",
        "calibration_metric": "LogLoss (negative BioTox-minus-Chem-only delta improves fit)",
        "stage2_model_spec": MODEL_SPEC,
        "calibration_intercept_frozen": True,
        "primary_inference": "Nadeau-Bengio corrected resampled paired t-test",
        "multiple_testing": "Benjamini-Hochberg within variant and cell",
        "confirmatory_multiplicity_family": (
            "standardized variant: 12 tasks within each cell; "
            "residualized variant is a separate sensitivity family"
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(summary_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize repeated calibrated nested-CV significance."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.input_dir.resolve(), args.output_dir.resolve())
    print(
        f"[repeated-cv-summary] wrote {len(summary)} task-cell-variant rows to "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
