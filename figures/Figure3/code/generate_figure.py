"""Plot unadjusted 6 h/24 h biological-context performance as grouped bar charts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import FancyArrowPatch, Patch
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((CODE_DIR / "config.json").read_text(encoding="utf-8"))
DEFAULT_INFERENCE_SOURCE = (CODE_DIR / CONFIG["input"]["inference_source"]).resolve()
DEFAULT_PRIMARY_6H_SOURCE = (CODE_DIR / CONFIG["input"]["primary_6h_source"]).resolve()
DEFAULT_PAIRED_24H_SOURCE = (CODE_DIR / CONFIG["input"]["paired_24h_source"]).resolve()
DEFAULT_OUTPUT_DIR = (CODE_DIR / CONFIG["output"]["directory"]).resolve()

TASKS = ["NR-Aromatase", "SR-ARE", "SR-MMP", "SR-p53"]
CELLS_6H = ["HA1E", "HEPG2", "HT29", "MCF7"]
CELLS_24H = ["HA1E", "HT29", "MCF7"]
ASSAY_CELL_LINES = {
    "NR-Aromatase": "MCF7",
    "SR-ARE": "HEPG2",
    "SR-MMP": "HEPG2",
    "SR-p53": "HT29",
}
ASSAY_CELL_LABELS = {
    "NR-Aromatase": "MCF7",
    "SR-ARE": "HepG2",
    "SR-MMP": "HepG2",
    "SR-p53": "HT29",
}

BEST_COLOR = "#B22222"
EXPLORATORY_COLOR = "#C29A3A"
OTHER_COLOR = "#8C8C8C"
ASSAY_MATCH_COLOR = "#5B9BD5"
BEST_CONTEXT_COLOR = "#E06B65"
OVERLAP_COLOR = "#9A78C2"
GRID_COLOR = "#D8D8D8"
ZERO_COLOR = "#777777"
BAR_ALPHA = 0.62
CHEM_HATCH = "///"
CHEM_BIO_HATCH = ""
ACTIVE_VARIANT = "standardized"
A4_WIDTH_IN = 8.27
A4_HEIGHT_IN = 11.69


FIGURE_CAPTION = """# Figure 3 caption

**Figure 3 | Cellular and temporal context dependence of transcriptomic
information.** (A) NR-Aromatase. (B) SR-ARE. (C) SR-MMP.
(D) SR-p53. Within each assay panel, results
are grouped first by exposure duration and then by biological context. The
transcriptomic input is the unadjusted (`standardized`) landmark-gene
representation. The
6 h group contains all four primary full-cohort cell lines (HA1E, HepG2,
HT29 and MCF7), whereas the 24 h group contains the three eligible
paired-timepoint cell lines (HA1E, HT29 and MCF7). HepG2 was unavailable at
24 h because its paired cohort was excluded before endpoint modelling. A
solid vertical rule separates the exposure groups.

The left subplot in each panel reports mean outer-test area under
the precision-recall curve (AUPRC) for the shared-intercept transcriptomic-term-off
molecular comparator (left, hatched bar) and the molecular-plus-transcriptomic model
(right, solid bar). Error bars show the sample standard deviation
across 20 repeat-level means, each averaged over five outer folds. Prominent
arrows connect paired bar heights, and labels report the signed AUPRC change.
Exactly one cell-line pair within each assay-by-exposure group, selected by
the highest absolute AUPRC across both models and all eligible cell lines, is
highlighted in firebrick for both the molecular and molecular-plus-
transcriptomic bars. Its paired gain label is bold. All other bars are grey.

The open rectangular tick-label box identifies the cell line used by the
corresponding Tox21 reporter assay. No cell-line colour coding is used.

The right subplot reports mean paired improvement over the
recalibrated frozen chemical predictor (Delta AUPRC). Error bars show
Nadeau-Bengio-corrected 95% confidence intervals and the dotted horizontal
line denotes no incremental gain. Positive gains with BH q<0.05 are
confirmatory and firebrick; positive gains with 0.05<=q<0.10 are exploratory
and ochre; other contexts are grey. Stars indicate confirmatory BH q-value
tiers (*q <= 0.05, **q <= 0.01 and ***q <= 0.001); a dagger marks
exploratory 0.05<=q<0.10.

Separate legends are centered beneath the two subplot columns. The left
legend identifies the chemical and chemical-plus-transcriptomic bars and the
best-performance highlighting. The right legend identifies non-significant
(q>=0.10), exploratory (0.05<=q<0.10), and confirmatory (q<0.05) incremental
gain categories.

Coloring is descriptive and was not used for model fitting, hyperparameter
selection or statistical inference. Error bars in the absolute and
incremental subplots quantify different quantities and should not be
compared directly. BH q-values are calculated within each analysis cohort;
nominal P values remain in the source table but do not determine figure
significance. Because the full 6 h and paired 24 h
cohorts differ, visual comparisons between exposure groups are descriptive
and are not paired timepoint estimates. AUPRC, area under the
precision-recall curve.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paired 6 h/24 h context-matching bar charts."
    )
    parser.add_argument(
        "--inference-source",
        type=Path,
        default=DEFAULT_INFERENCE_SOURCE,
    )
    parser.add_argument(
        "--primary-6h-source",
        type=Path,
        default=DEFAULT_PRIMARY_6H_SOURCE,
    )
    parser.add_argument(
        "--paired-24h-source",
        type=Path,
        default=DEFAULT_PAIRED_24H_SOURCE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--legend-y",
        type=float,
        default=0.008,
        help="Figure-relative y coordinate for the bottom edge of the shared legend.",
    )
    parser.add_argument(
        "--subplot-bottom",
        type=float,
        default=0.145,
        help="Figure-relative lower boundary of the subplot grid.",
    )
    parser.add_argument(
        "--variant", choices=("standardized", "residualized"), default="standardized",
        help="Stage-2 representation; standardized is the unadjusted main default.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_columns(
    frame: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def prepare_plotting_frame(
    inference: pd.DataFrame,
    primary_6h: pd.DataFrame,
    paired_24h: pd.DataFrame,
    primary_6h_source: Path,
    paired_24h_source: Path,
) -> pd.DataFrame:
    validate_columns(
        inference,
        {
            "task",
            "analysis_cohort",
            "lincs_cell_line",
            "lincs_exposure_h",
            "chem_auprc",
            "chem_bio_auprc",
            "delta_auprc",
            "nb_ci_low",
            "nb_ci_high",
            "nominal_p_value",
        },
        "inference source",
    )
    validate_columns(
        primary_6h,
        {
            "task",
            "cell",
            "variant",
            "mean_auprc_chem",
            "sd_repeat_mean_auprc_chem",
            "mean_auprc_biotox",
            "sd_repeat_mean_auprc_biotox",
            "n_compounds",
        },
        "primary 6 h performance source",
    )
    validate_columns(
        paired_24h,
        {
            "task",
            "cell",
            "variant",
            "mean_auprc_chem",
            "sd_repeat_mean_auprc_chem",
            "mean_auprc_biotox",
            "sd_repeat_mean_auprc_biotox",
            "n_compounds",
        },
        "paired 24 h performance source",
    )
    if len(inference) != 120:
        raise ValueError(
            f"Expected 120 complete residual-context rows, found {len(inference)}"
        )

    selected_6h = inference.loc[
        inference["task"].isin(TASKS)
        & inference["analysis_cohort"].eq("Primary full 6-h cohort")
        & inference["lincs_cell_line"].isin(CELLS_6H)
        & inference["lincs_exposure_h"].eq(6)
    ].copy()
    selected_24h = inference.loc[
        inference["task"].isin(TASKS)
        & inference["analysis_cohort"].eq("Paired 6/24-h cohort")
        & inference["lincs_cell_line"].isin(CELLS_24H)
        & inference["lincs_exposure_h"].eq(24)
    ].copy()
    selected_inference = pd.concat(
        [selected_6h, selected_24h],
        ignore_index=True,
    )
    primary_performance = primary_6h.loc[
        primary_6h["task"].isin(TASKS)
        & primary_6h["cell"].isin(CELLS_6H)
        & primary_6h["variant"].eq(ACTIVE_VARIANT)
    ].copy()
    primary_performance["lincs_exposure_h"] = 6
    primary_performance["source_run"] = str(primary_6h_source.parent)
    paired_performance = paired_24h.loc[
        paired_24h["task"].isin(TASKS)
        & paired_24h["cell"].isin(CELLS_24H)
        & paired_24h["variant"].eq(ACTIVE_VARIANT)
    ].copy()
    paired_performance["lincs_exposure_h"] = 24
    paired_performance["source_run"] = str(paired_24h_source.parent)
    selected_performance = pd.concat(
        [primary_performance, paired_performance],
        ignore_index=True,
    )

    result = selected_inference.merge(
        selected_performance[
            [
                "task",
                "cell",
                "lincs_exposure_h",
                "mean_auprc_chem",
                "sd_repeat_mean_auprc_chem",
                "mean_auprc_biotox",
                "sd_repeat_mean_auprc_biotox",
                "n_compounds",
                "source_run",
            ]
        ],
        left_on=["task", "lincs_cell_line", "lincs_exposure_h"],
        right_on=["task", "cell", "lincs_exposure_h"],
        how="left",
        validate="one_to_one",
    )
    expected_rows = len(TASKS) * (len(CELLS_6H) + len(CELLS_24H))
    if len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} context rows, found {len(result)}")
    if result[
        [
            "mean_auprc_chem",
            "sd_repeat_mean_auprc_chem",
            "mean_auprc_biotox",
            "sd_repeat_mean_auprc_biotox",
        ]
    ].isna().any().any():
        missing = result.loc[
            result[
                [
                    "mean_auprc_chem",
                    "sd_repeat_mean_auprc_chem",
                    "mean_auprc_biotox",
                    "sd_repeat_mean_auprc_biotox",
                ]
            ].isna().any(axis=1),
            ["task", "lincs_cell_line", "lincs_exposure_h"],
        ]
        raise ValueError(f"Missing performance rows:\n{missing}")
    if not np.allclose(
        result["mean_auprc_biotox"],
        result["chem_bio_auprc"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "Absolute AUPRC values disagree between performance and inference audits"
        )
    if not np.allclose(
        result["mean_auprc_chem"],
        result["chem_auprc"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "Chemical AUPRC values disagree between performance and inference audits"
        )

    task_order = {task: index for index, task in enumerate(TASKS)}
    time_order = {6: 0, 24: 1}
    cell_order = {"HA1E": 0, "HEPG2": 1, "HT29": 2, "MCF7": 3}
    result["_task_order"] = result["task"].map(task_order)
    result["_time_order"] = result["lincs_exposure_h"].map(time_order)
    result["_cell_order"] = result["lincs_cell_line"].map(cell_order)
    result = result.sort_values(
        ["_task_order", "_time_order", "_cell_order"]
    ).reset_index(drop=True)

    result["best_absolute_chem"] = False
    result["best_absolute_bio"] = False
    for task in TASKS:
        for exposure_h in (6, 24):
            task_rows = result.loc[
                result["task"].eq(task)
                & result["lincs_exposure_h"].eq(exposure_h)
            ]
            values = task_rows[
                ["chem_auprc", "chem_bio_auprc"]
            ].to_numpy()
            best_flat_index = int(np.nanargmax(values))
            best_row_offset, best_model_offset = np.unravel_index(
                best_flat_index,
                values.shape,
            )
            best_row_index = task_rows.index[best_row_offset]
            best_column = (
                "best_absolute_chem"
                if best_model_offset == 0
                else "best_absolute_bio"
            )
            result.loc[best_row_index, best_column] = True
    result["best_absolute_context"] = (
        result["best_absolute_chem"] | result["best_absolute_bio"]
    )
    result["toxicity_assay_cell_line"] = result["task"].map(ASSAY_CELL_LINES)
    result["assay_cell_match"] = result["lincs_cell_line"].eq(
        result["toxicity_assay_cell_line"]
    )
    result["tick_label_category"] = np.select(
        [
            result["assay_cell_match"] & result["best_absolute_context"],
            result["best_absolute_context"],
            result["assay_cell_match"],
        ],
        [
            "assay_match_and_best",
            "best_model_context",
            "assay_cell_match",
        ],
        default="other",
    )
    result["confirmatory_significant_positive"] = (
        result["delta_auprc"].gt(0) & result["bh_q_value"].lt(0.05)
    )
    result["exploratory_significant_positive"] = (
        result["delta_auprc"].gt(0)
        & result["bh_q_value"].ge(0.05)
        & result["bh_q_value"].lt(0.10)
    )
    # Preserve the historical column name for consumers while defining it
    # from the revised q-value tiers.
    result["nominal_significant_positive"] = (
        result["confirmatory_significant_positive"]
        | result["exploratory_significant_positive"]
    )

    for task in TASKS:
        observed = result.loc[result["task"].eq(task)]
        expected = (
            [(6, cell) for cell in CELLS_6H]
            + [(24, cell) for cell in CELLS_24H]
        )
        actual = list(
            zip(
                observed["lincs_exposure_h"],
                observed["lincs_cell_line"],
            )
        )
        if actual != expected:
            raise ValueError(f"Unexpected {task} context order: {actual}")
    return result


def add_time_bracket(
    axis: plt.Axes,
    x_start: float,
    x_end: float,
    label: str,
) -> None:
    y = -0.18
    transform = axis.get_xaxis_transform()
    axis.plot(
        [x_start, x_end],
        [y, y],
        transform=transform,
        color="#222222",
        linewidth=1.25,
        clip_on=False,
    )
    axis.text(
        (x_start + x_end) / 2,
        y - 0.030,
        label,
        transform=transform,
        ha="center",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#222222",
        clip_on=False,
    )


def significance_marker(value: float) -> str:
    if pd.isna(value) or value >= 0.10:
        return ""
    if value >= 0.05:
        return "†"
    if value <= 0.001:
        return "***"
    if value <= 0.01:
        return "**"
    return "*"


def plot_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    legend_y: float,
    subplot_bottom: float,
) -> None:
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=False)
    table_dir.mkdir(parents=True, exist_ok=False)

    fig, axes = plt.subplots(
        nrows=len(TASKS),
        ncols=2,
        figsize=(A4_WIDTH_IN, A4_HEIGHT_IN),
        sharex=False,
        sharey="col",
        gridspec_kw={
            "width_ratios": [1.0, 1.0],
            "hspace": 0.55,
            "wspace": 0.28,
        },
    )
    x_positions = np.array([0, 1.1, 2.2, 3.3, 5.0, 6.1, 7.2], dtype=float)
    tick_labels = CELLS_6H + CELLS_24H
    bar_width = 0.32
    pair_offset = bar_width / 2

    for row_index, task in enumerate(TASKS):
        task_frame = frame.loc[frame["task"].eq(task)].copy()
        # Color the complete selected cell-line pair consistently. This makes
        # the red/grey encoding describe performance context, not model type.
        pair_colors = np.where(
            task_frame["best_absolute_context"],
            BEST_COLOR,
            OTHER_COLOR,
        )

        absolute_axis = axes[row_index, 0]
        absolute_axis.bar(
            x_positions - pair_offset,
            task_frame["mean_auprc_chem"],
            width=bar_width,
            color=pair_colors,
            alpha=BAR_ALPHA,
            edgecolor="#222222",
            linewidth=1.0,
            hatch=CHEM_HATCH,
            yerr=task_frame["sd_repeat_mean_auprc_chem"],
            capsize=3.5,
            error_kw={"ecolor": "#222222", "elinewidth": 1.15, "capthick": 1.15},
            zorder=3,
        )
        absolute_axis.bar(
            x_positions + pair_offset,
            task_frame["mean_auprc_biotox"],
            width=bar_width,
            color=pair_colors,
            alpha=BAR_ALPHA,
            edgecolor="#222222",
            linewidth=1.0,
            hatch=CHEM_BIO_HATCH,
            yerr=task_frame["sd_repeat_mean_auprc_biotox"],
            capsize=3.5,
            error_kw={"ecolor": "#222222", "elinewidth": 1.15, "capthick": 1.15},
            zorder=3,
        )
        for context_index, (position, (_, record)) in enumerate(
            zip(x_positions, task_frame.iterrows())
        ):
            chem_value = float(record["mean_auprc_chem"])
            bio_value = float(record["mean_auprc_biotox"])
            gain = bio_value - chem_value
            if abs(gain) > 5e-5:
                right_bar_left = (
                    position + pair_offset - bar_width / 2
                )
                arrow = FancyArrowPatch(
                    (right_bar_left, chem_value),
                    (right_bar_left, bio_value),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.2,
                    color="#202020",
                    zorder=15,
                )
                absolute_axis.add_patch(arrow)
            annotation_y = max(
                chem_value + float(record["sd_repeat_mean_auprc_chem"]),
                bio_value + float(record["sd_repeat_mean_auprc_biotox"]),
            ) + 0.018
            is_best_context = bool(record["best_absolute_context"])
            absolute_axis.text(
                position,
                annotation_y,
                f"{gain:+.3f}",
                ha="center",
                va="bottom",
                fontsize=8.6 if is_best_context else 7.0,
                fontweight="bold" if is_best_context else "normal",
                color=BEST_COLOR if is_best_context else "#333333",
                clip_on=False,
                zorder=16,
            )
        absolute_axis.set_ylim(0.0, 1.25)
        absolute_axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        absolute_axis.set_ylabel("AUPRC", fontsize=11.8)

        delta_axis = axes[row_index, 1]
        delta = task_frame["delta_auprc"].to_numpy(dtype=float)
        ci_low = task_frame["nb_ci_low"].to_numpy(dtype=float)
        ci_high = task_frame["nb_ci_high"].to_numpy(dtype=float)
        lower_error = np.where(np.isfinite(ci_low), delta - ci_low, 0.0)
        upper_error = np.where(np.isfinite(ci_high), ci_high - delta, 0.0)
        delta_colors = np.select(
            [
                task_frame["confirmatory_significant_positive"],
                task_frame["exploratory_significant_positive"],
            ],
            [BEST_COLOR, EXPLORATORY_COLOR],
            default=OTHER_COLOR,
        )
        delta_axis.bar(
            x_positions,
            delta,
            width=0.42,
            color=delta_colors,
            alpha=BAR_ALPHA,
            edgecolor="#222222",
            linewidth=1.0,
            hatch=CHEM_BIO_HATCH,
            yerr=np.vstack([lower_error, upper_error]),
            capsize=4,
            error_kw={"ecolor": "#222222", "elinewidth": 1.25, "capthick": 1.25},
            zorder=3,
        )
        delta_axis.axhline(
            0.0,
            color=ZERO_COLOR,
            linewidth=1.1,
            linestyle=(0, (2, 3)),
            zorder=1,
        )
        delta_axis.set_ylim(-0.115, 0.48)
        delta_axis.set_yticks(np.arange(-0.10, 0.41, 0.10))
        delta_axis.set_ylabel(r"$\Delta$ AUPRC", fontsize=11.8)
        for context_index, (position, (_, record)) in enumerate(
            zip(x_positions, task_frame.iterrows())
        ):
            stars = significance_marker(record["bh_q_value"])
            if not stars or record["delta_auprc"] <= 0:
                continue
            annotation_y = (
                record["nb_ci_high"]
                if pd.notna(record["nb_ci_high"])
                else record["delta_auprc"]
            )
            delta_axis.text(
                position,
                float(annotation_y) + 0.018,
                stars,
                ha="center",
                va="bottom",
                fontsize=10.0,
                fontweight="bold",
                color="#111111",
                clip_on=False,
            )

        for axis in (absolute_axis, delta_axis):
            axis.set_xlim(-0.75, 7.95)
            axis.set_xticks(x_positions)
            tick_texts = axis.set_xticklabels(tick_labels, fontsize=8.2)
            for tick_text, (_, record) in zip(
                tick_texts,
                task_frame.iterrows(),
            ):
                category = record["tick_label_category"]
                if bool(record["assay_cell_match"]):
                    tick_text.set_bbox(
                        dict(
                            boxstyle="square,pad=0.20",
                            facecolor="none",
                            edgecolor="#111111",
                            linewidth=0.85,
                        )
                    )
            axis.grid(
                axis="y",
                color=GRID_COLOR,
                linewidth=0.8,
                linestyle=(0, (2, 3)),
                zorder=0,
            )
            axis.tick_params(axis="y", labelsize=8.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#444444")
            axis.spines["bottom"].set_color("#444444")
            axis.spines["left"].set_linewidth(0.9)
            axis.spines["bottom"].set_linewidth(0.9)
            axis.axvline(
                4.15,
                color="#444444",
                linewidth=0.9,
                zorder=1,
            )
            add_time_bracket(axis, -0.35, 3.65, "6 h")
            add_time_bracket(axis, 4.65, 7.55, "24 h")

    for axis in axes[-1, :]:
        axis.set_xlabel("Biological context", fontsize=12.2, labelpad=31)

    legend_handles = [
        Patch(
            facecolor="none",
            edgecolor="#222222",
            hatch=CHEM_HATCH,
            label="Chem.",
        ),
        Patch(
            facecolor="none",
            edgecolor="#222222",
            hatch=CHEM_BIO_HATCH,
            label="Chem. + Bio.",
        ),
        Patch(
            facecolor=BEST_COLOR,
            edgecolor="#222222",
            alpha=BAR_ALPHA,
            label="Best Perf.",
        ),
        Patch(
            facecolor=OTHER_COLOR,
            edgecolor="#222222",
            alpha=BAR_ALPHA,
            label="Rest",
        ),
    ]
    left_legend_handles = [
        legend_handles[0],
        legend_handles[1],
        Patch(
            facecolor=BEST_COLOR,
            edgecolor="#222222",
            alpha=BAR_ALPHA,
            label="Best Perf.",
        ),
        Patch(
            facecolor=OTHER_COLOR,
            edgecolor="#222222",
            alpha=BAR_ALPHA,
            label="Rest",
        ),
    ]
    right_legend_handles = [
        Patch(
            facecolor=OTHER_COLOR,
            edgecolor="#222222",
            alpha=BAR_ALPHA,
            label="No sig. (q ≥ 0.10)",
        ),
        Patch(
            facecolor=EXPLORATORY_COLOR,
            edgecolor="#222222",
            alpha=BAR_ALPHA,
            label="Exploratory (0.05 ≤ q < 0.10)",
        ),
        Patch(
            facecolor=BEST_COLOR,
            edgecolor="#222222",
            alpha=BAR_ALPHA,
            label="Confirmatory (q < 0.05)",
        ),
    ]
    left_legend = fig.legend(
        handles=left_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.29, legend_y),
        ncol=2,
        frameon=False,
        fontsize=8.3,
        borderpad=0.35,
        columnspacing=0.9,
        handlelength=1.5,
        labelspacing=0.35,
    )
    right_legend = fig.legend(
        handles=right_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.78, legend_y),
        ncol=1,
        frameon=False,
        fontsize=7.8,
        borderpad=0.25,
        columnspacing=0.0,
        handlelength=1.4,
        labelspacing=0.30,
    )
    for legend in (left_legend, right_legend):
        legend.get_frame().set_visible(False)
        legend.get_frame().set_edgecolor("none")
        legend.get_frame().set_linewidth(0.0)
    fig.subplots_adjust(
        left=0.115,
        right=0.965,
        top=0.948,
        bottom=subplot_bottom,
    )
    for panel_label, axis in zip(("A", "B", "C", "D"), axes[:, 0]):
        axis.text(
            -0.21,
            1.10,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=22.0,
            fontweight="bold",
            color="#111111",
            zorder=20,
        )
    for row_index, task in enumerate(TASKS):
        combined_left = axes[row_index, 0].get_position().x0
        combined_right = axes[row_index, 1].get_position().x1
        title_x = (combined_left + combined_right) / 2
        row_top = max(
            axes[row_index, 0].get_position().y1,
            axes[row_index, 1].get_position().y1,
        )
        title_box = TextArea(
            task,
            textprops={
                "fontsize": 14.0,
                "fontweight": "bold",
                "color": "#111111",
            },
        )
        fig.add_artist(
            AnnotationBbox(
                title_box,
                (title_x, row_top + 0.004),
                xycoords=fig.transFigure,
                box_alignment=(0.5, 0.0),
                frameon=False,
                pad=0.0,
            )
        )

    stem = figure_dir / "Figure3"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)

    export_columns = [
        "family",
        "task",
        "analysis_cohort",
        "lincs_cell_line",
        "lincs_exposure_h",
        "n_compounds",
        "chem_auprc",
        "sd_repeat_mean_auprc_chem",
        "chem_bio_auprc",
        "sd_repeat_mean_auprc_biotox",
        "delta_auprc",
        "nb_ci_low",
        "nb_ci_high",
        "nominal_p_value",
        "bh_q_value",
        "best_absolute_chem",
        "best_absolute_bio",
        "best_absolute_context",
        "toxicity_assay_cell_line",
        "assay_cell_match",
        "tick_label_category",
        "nominal_significant_positive",
        "exploratory_significant_positive",
        "confirmatory_significant_positive",
        "source_run",
    ]
    exported = frame[export_columns].rename(columns={
        "chem_auprc": "chem_term_off_auprc",
        "sd_repeat_mean_auprc_chem": "sd_repeat_mean_auprc_term_off",
    })
    exported.to_csv(
        table_dir / "biological_context_matching_plotting_data.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    global ACTIVE_VARIANT
    ACTIVE_VARIANT = args.variant
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output_dir}"
        )
    inference = pd.read_csv(args.inference_source)
    inference = inference.rename(columns={"chem_term_off_auprc": "chem_auprc"})
    if "chem_residual_auprc" in inference.columns and "chem_bio_auprc" not in inference.columns:
        inference = inference.rename(columns={"chem_residual_auprc": "chem_bio_auprc"})
    primary_6h = pd.read_csv(args.primary_6h_source)
    paired_24h = pd.read_csv(args.paired_24h_source)
    for frame in (primary_6h, paired_24h):
        frame.rename(columns={
            "mean_auprc_term_off": "mean_auprc_chem",
            "sd_repeat_mean_auprc_term_off": "sd_repeat_mean_auprc_chem",
        }, inplace=True)
    frame = prepare_plotting_frame(
        inference,
        primary_6h,
        paired_24h,
        args.primary_6h_source,
        args.paired_24h_source,
    )
    plot_figure(
        frame,
        args.output_dir,
        args.dpi,
        args.legend_y,
        args.subplot_bottom,
    )
    (args.output_dir / "FIGURE_CAPTION.md").write_text(
        FIGURE_CAPTION,
        encoding="utf-8",
    )
    manifest = {
        "analysis": "biological_context_matching_barplot_a4_final",
        "revision": 27,
        "page_size": "A4 portrait",
        "figure_size_inches": [A4_WIDTH_IN, A4_HEIGHT_IN],
        "export_bbox": "fixed page canvas; no tight cropping",
        "model_refit": False,
        "variant": args.variant,
        "representation": "unadjusted" if args.variant == "standardized" else "residualized",
        "inference_source": str(args.inference_source),
        "inference_source_sha256": sha256(args.inference_source),
        "primary_6h_source": str(args.primary_6h_source),
        "primary_6h_source_sha256": sha256(args.primary_6h_source),
        "paired_24h_source": str(args.paired_24h_source),
        "paired_24h_source_sha256": sha256(args.paired_24h_source),
        "selection_rule": (
            f"{args.variant} primary full-cohort 6 h estimates for HA1E, HepG2, "
            "HT29 and MCF7 plus paired-cohort 24 h estimates for HA1E, HT29 "
            "and MCF7 across four selected assays"
        ),
        "ranking_rule": (
            "Within each assay and exposure group, highlight exactly one "
            "bar: the highest absolute AUPRC across both models and all "
            "eligible cell lines; highlight positive BH q<0.05 gains as "
            "confirmatory and 0.05<=q<0.10 gains as exploratory in the "
            "incremental subplot"
        ),
        "n_rows": int(len(frame)),
        "n_assays": int(frame["task"].nunique()),
        "n_contexts_per_assay": 7,
        "paired_bar_geometry": "Mol. and Mol. + Bio. inner edges meet without a gap",
        "gain_annotation_position": "centered above the taller paired error bar",
        "font_sizes_points": {
            "panel_letter": 22.0,
            "assay_title": 14.0,
            "assay_context": 10.5,
            "axis_labels": 11.8,
            "bottom_axis_labels": 12.2,
            "x_tick_labels": 8.2,
            "y_tick_labels": 8.8,
            "within_panel_legends": 7.5,
            "left_bottom_legend": 8.3,
            "right_bottom_legend": 7.8,
            "bottom_legend_title": 10.0
        },
        "bottom_legend_title": None,
        "bottom_legend_anchor_y": args.legend_y,
        "bottom_legend_frame": False,
        "legend_style": {
            "chem_hatch": "///",
            "chem_bio_hatch": "",
            "frame_visible": False,
            "interior_margin_removed": True,
            "left_legend_anchor_x": 0.29,
            "left_legend_ncol": 2,
            "right_legend_anchor_x": 0.78,
            "right_legend_ncol": 1,
            "right_legend_categories": [
                "No sig. (q >= 0.10)",
                "Exploratory (0.05 <= q < 0.10)",
                "Confirmatory (q < 0.05)"
            ],
        },
        "subplot_grid_bottom": args.subplot_bottom,
        "output_formats": ["png", "pdf", "svg", "csv", "md"],
    }
    (args.output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved publication figure to {args.output_dir}")


if __name__ == "__main__":
    main()
