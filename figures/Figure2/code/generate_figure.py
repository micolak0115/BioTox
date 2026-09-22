"""Final A4 portrait Figure 2 with tall publication layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


CODE_DIR = Path(__file__).resolve().parent
A4_PORTRAIT_INCHES = (8.27, 11.69)

CONFIG = json.loads((CODE_DIR / "config.json").read_text(encoding="utf-8"))
DEFAULT_INPUT = (CODE_DIR / CONFIG["input"]["topk_ensemble_curve"]).resolve()
DEFAULT_OUTPUT_DIR = (CODE_DIR / CONFIG["output"]["directory"]).resolve()

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
NR_TASKS = TASK_ORDER[:7]
SR_TASKS = TASK_ORDER[7:]
K_SELECTED = 4

CURVE_COLOR = "#4B5964"
K4_COLOR = "#C43C39"
GLOBAL_COLOR = "#111111"
NR_COLOR = "#E68613"
SR_COLOR = "#3977A8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_curve(curve: pd.DataFrame) -> None:
    required = {"task", "k", "auprc"}
    missing = required.difference(curve.columns)
    if missing:
        raise ValueError(f"Top-K curve is missing columns: {sorted(missing)}")
    if set(curve["task"]) != set(TASK_ORDER):
        raise ValueError("Top-K curve does not contain the expected 12 tasks")
    if curve.duplicated(["task", "k"]).any():
        raise ValueError("Top-K curve contains duplicate task/K rows")
    for task, group in curve.groupby("task", sort=False):
        if set(group["k"].astype(int)) != set(range(1, 13)):
            raise ValueError(f"{task} does not contain K=1,...,12")
    if not np.isfinite(curve["auprc"].to_numpy(dtype=float)).all():
        raise ValueError("Top-K AUPRC contains non-finite values")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.linewidth": 1.0,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def draw_endpoint(axis: plt.Axes, curve: pd.DataFrame, task: str) -> None:
    task_curve = curve.loc[curve["task"].eq(task)].sort_values("k")
    k_values = task_curve["k"].to_numpy(dtype=int)
    values = task_curve["auprc"].to_numpy(dtype=float)
    k4_value = float(values[np.flatnonzero(k_values == K_SELECTED)[0]])

    axis.axvline(
        K_SELECTED,
        color=K4_COLOR,
        linestyle=(0, (2, 2.5)),
        linewidth=1.15,
        alpha=0.9,
        zorder=1,
    )
    axis.plot(
        k_values,
        values,
        color=CURVE_COLOR,
        linewidth=1.65,
        marker="o",
        markersize=3.7,
        markerfacecolor="white",
        markeredgecolor=CURVE_COLOR,
        markeredgewidth=1.0,
        zorder=2,
    )
    axis.scatter(
        [K_SELECTED],
        [k4_value],
        s=48,
        marker="o",
        facecolor=K4_COLOR,
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
    )
    axis.set_title(task, fontsize=11.0, fontweight="bold", pad=4)
    axis.set_xlim(0.5, 12.5)
    axis.set_xticks([2, 4, 6, 8, 10, 12])
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.65)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=9.5, width=0.9, length=3.2, pad=2)


def build_figure(curve: pd.DataFrame):
    configure_style()
    figure = plt.figure(figsize=A4_PORTRAIT_INCHES)
    outer = figure.add_gridspec(
        3,
        1,
        height_ratios=(1.0, 2.15, 1.75),
        left=0.105,
        right=0.975,
        top=0.935,
        bottom=0.065,
        hspace=0.48,
    )
    macro_axis = figure.add_subplot(outer[0, 0])
    nr_grid = outer[1, 0].subgridspec(3, 3, wspace=0.36, hspace=0.58)
    sr_grid = outer[2, 0].subgridspec(2, 3, wspace=0.34, hspace=0.54)
    nr_axes = [figure.add_subplot(nr_grid[index // 3, index % 3]) for index in range(7)]
    sr_axes = [figure.add_subplot(sr_grid[index // 3, index % 3]) for index in range(5)]

    global_macro = (
        curve.groupby("k", as_index=False, sort=True)["auprc"]
        .mean()
        .rename(columns={"auprc": "global_macro_auprc"})
    )
    nr_macro = (
        curve.loc[curve["task"].isin(NR_TASKS)]
        .groupby("k", as_index=False, sort=True)["auprc"]
        .mean()
        .rename(columns={"auprc": "nr_macro_auprc"})
    )
    sr_macro = (
        curve.loc[curve["task"].isin(SR_TASKS)]
        .groupby("k", as_index=False, sort=True)["auprc"]
        .mean()
        .rename(columns={"auprc": "sr_macro_auprc"})
    )
    macro = global_macro.merge(nr_macro, on="k", validate="one_to_one").merge(
        sr_macro,
        on="k",
        validate="one_to_one",
    )
    k_values = macro["k"].to_numpy(dtype=int)
    handles = []
    for column, color, linewidth, markersize, label in (
        ("global_macro_auprc", GLOBAL_COLOR, 2.2, 4.8, "All endpoints"),
        ("nr_macro_auprc", NR_COLOR, 1.7, 4.1, "NR endpoints"),
        ("sr_macro_auprc", SR_COLOR, 1.7, 4.1, "SR endpoints"),
    ):
        line, = macro_axis.plot(
            k_values,
            macro[column],
            color=color,
            linewidth=linewidth,
            marker="o",
            markersize=markersize,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            label=label,
            zorder=3,
        )
        handles.append(line)

    k4_value = float(
        macro.loc[macro["k"].eq(K_SELECTED), "global_macro_auprc"].iloc[0]
    )
    macro_axis.axvline(
        K_SELECTED,
        color=K4_COLOR,
        linestyle=(0, (2, 2.5)),
        linewidth=1.3,
        alpha=0.9,
        zorder=1,
    )
    macro_axis.scatter(
        [K_SELECTED],
        [k4_value],
        s=75,
        facecolor=K4_COLOR,
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )
    macro_axis.set_title(
        "Macro-AUPRC for Ensemble selection",
        fontsize=14.0,
        fontweight="bold",
        pad=7,
    )
    macro_axis.set_xlabel("Top-K ensemble size", fontsize=11.5, labelpad=4)
    macro_axis.set_ylabel("Held-out test AUPRC", fontsize=11.5, labelpad=5)
    macro_axis.set_xlim(0.5, 12.5)
    macro_axis.set_xticks(range(1, 13))
    macro_axis.set_ylim(0.41, 0.51)
    macro_axis.set_yticks(np.arange(0.41, 0.511, 0.02))
    macro_axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    macro_axis.grid(axis="x", visible=False)
    macro_axis.spines[["top", "right"]].set_visible(False)
    macro_axis.tick_params(axis="both", labelsize=9.8, width=1.0, length=3.5, pad=2)

    for axis, task in zip(nr_axes, NR_TASKS):
        draw_endpoint(axis, curve, task)
    for axis, task in zip(sr_axes, SR_TASKS):
        draw_endpoint(axis, curve, task)
    for axes in (nr_axes, sr_axes):
        for axis in axes:
            axis.tick_params(labelsize=8.3)

    macro_position = macro_axis.get_position()
    panel_letter_x = 0.025
    figure.text(
        panel_letter_x,
        macro_position.y1 + 0.012,
        "A",
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    # Keep all panel letters on one vertical guide at the page's left edge.
    for letter, axes in (("B", nr_axes), ("C", sr_axes)):
        left = min(axis.get_position().x0 for axis in axes)
        top = max(axis.get_position().y1 for axis in axes)
        bottom = min(axis.get_position().y0 for axis in axes)
        right = max(axis.get_position().x1 for axis in axes)
        figure.text(
            panel_letter_x,
            top + 0.004,
            letter,
            fontsize=17,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        figure.text(
            (left + right) / 2,
            bottom - 0.024,
            "Top-K ensemble size",
            fontsize=11.5,
            ha="center",
            va="top",
        )
        figure.text(
            left - 0.070,
            (bottom + top) / 2,
            "Held-out test AUPRC",
            fontsize=11.5,
            rotation=90,
            ha="center",
            va="center",
        )
    macro_axis.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        facecolor="white",
        edgecolor="#777777",
        framealpha=1.0,
        ncol=1,
        handlelength=2.3,
        columnspacing=1.0,
        borderpad=0.45,
    )
    return figure, macro


def run(input_path: Path, output_dir: Path, dpi: int) -> dict[str, object]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    curve = pd.read_csv(input_path)
    validate_curve(curve)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, macro = build_figure(curve)
    stem = output_dir / "stage1_topk_ensemble_a4_publication_final"
    figure_paths = [stem.with_suffix(suffix) for suffix in (".pdf", ".png", ".svg")]
    figure.savefig(figure_paths[0], format="pdf", metadata={"Title": "Figure 2"})
    figure.savefig(figure_paths[1], format="png", dpi=dpi)
    figure.savefig(figure_paths[2], format="svg", metadata={"Title": "Figure 2"})
    plt.close(figure)

    macro_path = output_dir / "stage1_topk_global_macro_auprc.csv"
    macro.to_csv(macro_path, index=False)
    caption_path = output_dir / "stage1_topk_ensemble_a4_publication_final_caption.md"
    caption_path.write_text(
        """# Figure 2 caption

**Figure 2 | Stage-1 chemical ensemble selection across ensemble sizes.** **A**, Held-out
macro area under the precision-recall curve (AUPRC) for equal-weight Top-K
chemical ensembles with K ranging from 1 to 12. The black curve is the
unweighted mean across all 12 Tox21 endpoints; orange and blue curves show the
corresponding means across the seven nuclear-receptor (NR) and five
stress-response (SR) endpoints. **B**, Endpoint-specific held-out AUPRC for the
seven NR assays. **C**, Endpoint-specific held-out AUPRC for the five SR assays.
Candidate chemical models were ranked separately within each endpoint, and
predictions from the highest-ranked K models were averaged with equal weight.
The filled red marker and vertical dotted line identify K = 4, which maximized
global macro-AUPRC and was frozen for Stage-2 analyses. Panel A uses an AUPRC
range of 0.41-0.51; endpoint-specific panels use a common 0-1 range. Panels B
and C use compact multi-row endpoint grids to preserve readable axes on a
tall publication page. Panel A is vertically expanded while retaining a
separate gutter above Panel B. The legend occupies the upper page margin
outside all plotting axes, and the figure is rendered natively on an A4
portrait canvas.
""",
        encoding="utf-8",
    )

    peak_row = macro.loc[macro["global_macro_auprc"].idxmax()]
    selected_macro = float(
        macro.loc[macro["k"].eq(K_SELECTED), "global_macro_auprc"].iloc[0]
    )
    manifest = {
        "analysis": "stage1_topk_ensemble_a4_portrait_publication_final",
        "supersedes": "Figure2_A4_final_v4",
        "source_curve": str(input_path),
        "source_curve_sha256": sha256(input_path),
        "canvas_inches": list(A4_PORTRAIT_INCHES),
        "paper": "A4",
        "orientation": "portrait",
        "n_tasks": len(TASK_ORDER),
        "selected_k": K_SELECTED,
        "selected_k_global_macro_auprc": selected_macro,
        "peak_k": int(peak_row["k"]),
        "peak_global_macro_auprc": float(peak_row["global_macro_auprc"]),
        "panel_a_title": "Macro-AUPRC for Ensemble selection",
        "panel_layout": {
            "A": "one vertically expanded full-width macro panel",
            "B": "seven NR subplots in a three-column portrait grid",
            "C": "five SR subplots in a three-column portrait grid",
        },
        "legend_location": "inside Panel A below the title",
        "figure_paths": [str(path) for path in figure_paths],
        "figure_sha256": {path.name: sha256(path) for path in figure_paths},
        "macro_data": str(macro_path),
        "caption": str(caption_path),
        "complete": True,
    }
    (output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the final non-overlapping A4 Figure 2."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments.input, arguments.output_dir, arguments.dpi), indent=2))
