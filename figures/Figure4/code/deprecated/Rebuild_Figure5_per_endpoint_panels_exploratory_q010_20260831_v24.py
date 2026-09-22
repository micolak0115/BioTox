#!/usr/bin/env python3
"""Figure 5 v24: exploratory q<0.10 fractions with confirmatory q<0.05 markers.

This is a plotting-only revision of v23. It reads the same audited v21 GSEA
tables and does not recompute enrichment statistics.

Display contract:
  - Per-endpoint compound panels show the top five active compounds by |NES|.
  - ``*`` marks BH-FDR q<0.05 within the endpoint-pathway compound family.
  - ``x`` marks exploratory 0.05<=q<0.10; confirmatory hits retain ``*``.
  - The heatmap color is the fraction of active compounds with q<0.10.
  - Heatmap ``*`` significance remains confirmatory: at least one compound with
    q<0.05 in that endpoint-pathway family.
  - Heatmap columns remain restricted to pathways with a confirmatory q<0.05
    compound in at least one endpoint, matching v23's column inclusion rule.
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/biotox_figure5_v24_q010_mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd


V21_TABLES = Path(
    "/home/kyungan/scripts/BioTox/publication/final/figures_revised/Figure5/"
    "unadjusted_global_zbeta_heterogeneity_20260828_v21/tables"
)
OUT_DIR = Path(
    "/home/kyungan/scripts/BioTox/publication/final/figures_revised/Figure5/"
    "unadjusted_global_zbeta_exploratory_q010_20260831_v24"
)
FIG_STEM = "Figure5_per_endpoint_panels_A4_exploratory_q010_20260831_v24"
DATA_FILE_PREFIX = "Figure5"

FIG_SIZE_IN = (11.69, 15.6)
CONFIRMATORY_Q = 0.05
EXPLORATORY_Q = 0.10
TOP_N_COMPOUNDS = 5
EXPLORATORY_MARKER = "x"
EXPLORATORY_MARKER_FONTSIZE = 13
ROW_WIDTH_RATIOS = [0.42, 2.28]
ROW_WSPACE = 0.07
GENE_BOX_ASPECT: float | None = 1.0
GENE_HEIGHT_SCALE = 1.0
GENE_VERTICAL_SHIFT = 0.0
DISPLAY_GENE_SET_NAMES: dict[str, str] = {}

OUTER_HSPACE = 0.95
OUTER_LEFT = 0.085
OUTER_RIGHT = 0.93
OUTER_TOP = 0.985
OUTER_BOTTOM = 0.145
ENDPOINT_LABELPAD = 9
ENDPOINT_LABEL_X: float | None = None
PANEL_LETTER_FONTSIZE = 26
PANEL_LETTER_Y_OFFSET = 0.032
ALIGN_PANEL_LETTERS_TO_TITLES = False
SHOW_FOOTNOTE = True
FOOTNOTE_Y = 0.012
SHOW_DIRECTION_LEGEND = False
DIRECTION_LEGEND_Y = 0.991
DIRECTION_LEGEND_FONTSIZE = 10.5
SHOW_TOP_COEFFICIENTS_SUBTITLE = True
PANEL_E_VERTICAL_SHIFT = 0.0
SHARE_GENE_XAXIS = False

GENE_XLABEL_FONTSIZE = 9.0
GENE_TICK_FONTSIZE = 8.0
GENE_LABEL_FONTSIZE = 9.6
GENE_LABEL_FONTWEIGHT = "bold"
GENE_LABEL_FONTSTYLE = "italic"
ENDPOINT_FONTSIZE = 12.5
SUBTITLE_FONTSIZE = 9.6
GSEA_XLABEL_FONTSIZE = 8.0
GSEA_YLABEL_FONTSIZE = 9.2
GSEA_TICK_FONTSIZE = 8.0
SIGNIFICANCE_MARKER_FONTSIZE = 15
HEATMAP_ENDPOINT_FONTSIZE = 11
HEATMAP_SELECTED_LABEL_FONTSIZE = 8.6
HEATMAP_LABEL_FONTSIZE = 8.0
HEATMAP_MARKER_FONTSIZE = 15
HEATMAP_TITLE_FONTSIZE = 11.5
HEATMAP_DIVIDER_AFTER_GENE_SET: str | None = None
HEATMAP_DIVIDER_COLOR = "black"
HEATMAP_DIVIDER_LINEWIDTH = 1.6
COLORBAR_TICK_FONTSIZE = 8.2
COLORBAR_LABEL_FONTSIZE = 9.0
COLORBAR_LABEL = "Fraction with BH q < 0.10 (exploratory)"
FOOTNOTE_FONTSIZE = 8.2

TASKS = ["NR-Aromatase", "SR-ARE", "SR-MMP", "SR-p53"]
INK = "#20252B"
GRID = "#D6DBE0"
BLUE = "#3E78A8"
RED = "#B44F4B"
LETTER_X = 0.006

PANEL_PATHWAY = {
    "NR-Aromatase": "Cholesterol Homeostasis",
    "SR-ARE": "Reactive Oxygen Species Pathway",
    "SR-MMP": "TNF-alpha Signaling via NF-kB",
    "SR-p53": "p53 Pathway",
}


def wrap_label(name: str, width: int) -> str:
    return textwrap.fill(name, width=width, break_long_words=False, break_on_hyphens=False)


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def marker_for_q(q: float) -> str:
    if q < CONFIRMATORY_Q:
        return "*"
    if q < EXPLORATORY_Q:
        return EXPLORATORY_MARKER
    return ""


def style(axis: plt.Axes, tick: float) -> None:
    axis.grid(color=GRID, linestyle=(0, (2, 2)), linewidth=0.6, alpha=0.9)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=tick, length=3, width=0.8, pad=2.5)


def adjust_gene_axis_position(ax: plt.Axes) -> None:
    """Resize the gene axis vertically without changing the paired GSEA axis."""
    if GENE_HEIGHT_SCALE == 1.0 and GENE_VERTICAL_SHIFT == 0.0:
        return
    position = ax.get_position()
    new_height = position.height * GENE_HEIGHT_SCALE
    new_y0 = position.y0 - (new_height - position.height) / 2 + GENE_VERTICAL_SHIFT
    ax.set_position([position.x0, new_y0, position.width, new_height])


def panel_global_coefficient(ax: plt.Axes, task: str, beta_top: pd.DataFrame) -> None:
    sub = beta_top.loc[beta_top["task"].eq(task)].sort_values("abs_beta_mean", ascending=False).head(10)
    sub = sub.sort_values("beta_mean")
    ax.barh(
        sub["gene"],
        sub["beta_mean"],
        color=np.where(sub["beta_mean"].ge(0), RED, BLUE),
        edgecolor=INK,
        linewidth=0.5,
        zorder=3,
    )
    ax.axvline(0, color=INK, linewidth=0.9, linestyle=(0, (2, 2)))
    ax.set_xlabel("Mean ridge $\\beta$", fontsize=GENE_XLABEL_FONTSIZE, fontweight="bold")
    style(ax, GENE_TICK_FONTSIZE)
    for label in ax.get_yticklabels():
        label.set_fontweight(GENE_LABEL_FONTWEIGHT)
        label.set_fontstyle(GENE_LABEL_FONTSTYLE)
        label.set_fontsize(GENE_LABEL_FONTSIZE)
    if GENE_BOX_ASPECT is not None:
        ax.set_box_aspect(GENE_BOX_ASPECT)
    ax.set_ylabel(
        task,
        fontsize=ENDPOINT_FONTSIZE,
        fontweight="bold",
        rotation=90,
        labelpad=ENDPOINT_LABELPAD,
    )
    if ENDPOINT_LABEL_X is not None:
        ax.yaxis.set_label_coords(ENDPOINT_LABEL_X, 0.5)
    if SHOW_TOP_COEFFICIENTS_SUBTITLE:
        ax.set_title("Top coefficients", fontsize=SUBTITLE_FONTSIZE, fontweight="bold", loc="center", pad=5)


def panel_compound_nes(
    ax: plt.Axes,
    task: str,
    gene_set: str,
    union_gsea: pd.DataFrame,
) -> pd.DataFrame:
    sub = union_gsea.loc[
        union_gsea["task"].eq(task) & union_gsea["gene_set"].eq(gene_set)
    ].copy()
    sub["bh_q_within_task_pathway"] = bh(sub["nominal_p"].to_numpy(float))
    sub["abs_nes"] = sub["nes"].abs()
    top5 = sub.sort_values("abs_nes", ascending=False).head(TOP_N_COMPOUNDS).reset_index(drop=True)
    top5["abs_nes_rank"] = np.arange(1, len(top5) + 1)
    top5["marker"] = top5["bh_q_within_task_pathway"].map(marker_for_q)

    x = np.arange(len(top5))
    colors = np.where(top5["nes"].ge(0), RED, BLUE)
    bars = ax.bar(x, top5["abs_nes"], color=colors, edgecolor=INK, linewidth=0.6, width=0.62, zorder=3)
    ax.set_xticks(x)
    wrapped_names = [wrap_label(name, width=22) for name in top5["drug_name"]]
    ax.set_xticklabels(
        wrapped_names,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=GSEA_XLABEL_FONTSIZE,
        fontweight="bold",
    )
    ymax = float(top5["abs_nes"].max())
    ax.set_ylim(0, ymax * 1.30)
    for bar, marker in zip(bars, top5["marker"]):
        if marker:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.04,
                marker,
                ha="center",
                va="bottom",
                fontsize=(
                    SIGNIFICANCE_MARKER_FONTSIZE
                    if marker == "*"
                    else EXPLORATORY_MARKER_FONTSIZE
                ),
                fontweight="bold",
                color=INK,
                zorder=4,
            )
    ax.set_ylabel("|NES|", fontsize=GSEA_YLABEL_FONTSIZE, fontweight="bold")
    display_gene_set = DISPLAY_GENE_SET_NAMES.get(gene_set, gene_set)
    ax.set_title(
        wrap_label(display_gene_set, width=34),
        fontsize=SUBTITLE_FONTSIZE,
        fontweight="bold",
        loc="center",
        pad=5,
    )
    style(ax, GSEA_TICK_FONTSIZE)
    ax.set_xlim(-0.55, len(top5) - 0.45)

    return top5[
        [
            "task",
            "gene_set",
            "abs_nes_rank",
            "ik",
            "drug_name",
            "nes",
            "abs_nes",
            "nominal_p",
            "bh_q_within_task_pathway",
            "marker",
            "leading_edge_genes",
        ]
    ].copy()


def panel_screening_heatmap(
    ax: plt.Axes,
    fig: plt.Figure,
    union_summary: pd.DataFrame,
) -> pd.DataFrame:
    plot_df = union_summary.loc[union_summary["n_endpoints_significant"].gt(0)].copy()
    col_order = (
        plot_df[["gene_set", "n_endpoints_significant"]]
        .drop_duplicates()
        .sort_values(["n_endpoints_significant", "gene_set"], ascending=[False, True])["gene_set"]
        .tolist()
    )
    pivot = plot_df.pivot(
        index="task", columns="gene_set", values="fraction_q_lt_0_10"
    ).reindex(index=TASKS, columns=col_order)
    sig_pivot = plot_df.pivot(
        index="task", columns="gene_set", values="confirmed_enriched"
    ).reindex(index=TASKS, columns=col_order)
    vmax = max(0.20, float(np.nanmax(pivot.to_numpy(float))))
    image = ax.imshow(pivot.to_numpy(float), cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
    if HEATMAP_DIVIDER_AFTER_GENE_SET is not None:
        if HEATMAP_DIVIDER_AFTER_GENE_SET not in pivot.columns:
            raise ValueError(
                "Heatmap divider pathway not found: "
                f"{HEATMAP_DIVIDER_AFTER_GENE_SET}"
            )
        divider_index = pivot.columns.get_loc(HEATMAP_DIVIDER_AFTER_GENE_SET)
        ax.axvline(
            divider_index + 0.5,
            color=HEATMAP_DIVIDER_COLOR,
            linewidth=HEATMAP_DIVIDER_LINEWIDTH,
            zorder=5,
        )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=HEATMAP_ENDPOINT_FONTSIZE, fontweight="bold")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([])

    selected_names = set(PANEL_PATHWAY.values())
    for j, gene_set in enumerate(pivot.columns):
        is_selected = gene_set in selected_names
        display_gene_set = DISPLAY_GENE_SET_NAMES.get(gene_set, gene_set)
        ax.text(
            j,
            -0.05,
            wrap_label(display_gene_set, width=26),
            transform=ax.get_xaxis_transform(),
            rotation=45,
            rotation_mode="anchor",
            ha="right",
            va="top",
            fontsize=(
                HEATMAP_SELECTED_LABEL_FONTSIZE if is_selected else HEATMAP_LABEL_FONTSIZE
            ),
            linespacing=1.15,
            fontweight="bold" if is_selected else "normal",
            color=INK if is_selected else "#6B7280",
        )
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            sig = bool(sig_pivot.iloc[i, j]) if pd.notna(sig_pivot.iloc[i, j]) else False
            ax.text(
                j,
                i,
                "*" if pd.notna(val) and sig else "",
                ha="center",
                va="center",
                fontsize=HEATMAP_MARKER_FONTSIZE,
                fontweight="bold",
                color="white" if sig and val >= vmax * 0.55 else INK,
            )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.8%", pad=0.35)
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    cbar.set_label(
        COLORBAR_LABEL,
        fontsize=COLORBAR_LABEL_FONTSIZE,
        fontweight="bold",
    )
    ax.set_title(
        "Cross-endpoint GSEA screening",
        fontsize=HEATMAP_TITLE_FONTSIZE,
        fontweight="bold",
        loc="center",
        pad=7,
    )

    audit = plot_df[
        [
            "task",
            "gene_set",
            "n_active",
            "q_lt_0_05_n",
            "q_lt_0_10_n",
            "fraction_q_lt_0_05",
            "fraction_q_lt_0_10",
            "confirmed_enriched",
            "n_endpoints_significant",
        ]
    ].copy()
    audit.insert(2, "source_gene_set", audit["gene_set"])
    audit["gene_set"] = audit["gene_set"].replace(DISPLAY_GENE_SET_NAMES)
    return audit


def validate_summary(union_summary: pd.DataFrame) -> None:
    required = {
        "n_active",
        "q_lt_0_05_n",
        "q_lt_0_10_n",
        "fraction_q_lt_0_05",
        "fraction_q_lt_0_10",
        "confirmed_enriched",
        "n_endpoints_significant",
    }
    missing = required.difference(union_summary.columns)
    if missing:
        raise ValueError(f"Missing required union-summary columns: {sorted(missing)}")
    expected_q10_fraction = union_summary["q_lt_0_10_n"] / union_summary["n_active"]
    if not np.allclose(union_summary["fraction_q_lt_0_10"], expected_q10_fraction):
        raise ValueError("fraction_q_lt_0_10 does not equal q_lt_0_10_n / n_active")
    expected_confirmed = union_summary["q_lt_0_05_n"].gt(0)
    if not expected_confirmed.equals(union_summary["confirmed_enriched"].astype(bool)):
        raise ValueError("confirmed_enriched is inconsistent with q_lt_0_05_n > 0")


def main() -> None:
    beta_top = pd.read_csv(V21_TABLES / f"{DATA_FILE_PREFIX}_top_global_coefficients.csv")
    union_gsea = pd.read_csv(V21_TABLES / f"{DATA_FILE_PREFIX}_pathway_gsea_per_compound.csv")
    union_summary = pd.read_csv(V21_TABLES / f"{DATA_FILE_PREFIX}_pathway_selection.csv")
    validate_summary(union_summary)

    for task, gene_set in PANEL_PATHWAY.items():
        n = len(union_gsea[union_gsea["task"].eq(task) & union_gsea["gene_set"].eq(gene_set)])
        if n == 0:
            raise ValueError(f"No cached GSEA rows for {task}/{gene_set}")

    fig = plt.figure(figsize=FIG_SIZE_IN, facecolor="white")
    outer = GridSpec(
        5,
        1,
        figure=fig,
        hspace=OUTER_HSPACE,
        left=OUTER_LEFT,
        right=OUTER_RIGHT,
        top=OUTER_TOP,
        bottom=OUTER_BOTTOM,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 1.55],
    )

    top5_tables: list[pd.DataFrame] = []
    letter_axes: list[tuple[str, plt.Axes]] = []
    gene_axes: list[plt.Axes] = []
    for row_i, task in enumerate(TASKS):
        row_gs = GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=outer[row_i],
            width_ratios=ROW_WIDTH_RATIOS,
            wspace=ROW_WSPACE,
        )
        ax_left = fig.add_subplot(row_gs[0])
        ax_right = fig.add_subplot(row_gs[1])
        adjust_gene_axis_position(ax_left)
        panel_global_coefficient(ax_left, task, beta_top)
        gene_axes.append(ax_left)
        top5_tables.append(panel_compound_nes(ax_right, task, PANEL_PATHWAY[task], union_gsea))
        letter_axes.append(("ABCD"[row_i], ax_left))

    if SHARE_GENE_XAXIS:
        shared_limit = max(abs(bound) for axis in gene_axes for bound in axis.get_xlim())
        for axis in gene_axes:
            axis.set_xlim(-shared_limit, shared_limit)

    ax_e = fig.add_subplot(outer[4])
    if PANEL_E_VERTICAL_SHIFT != 0.0:
        position = ax_e.get_position()
        ax_e.set_position(
            [position.x0, position.y0 + PANEL_E_VERTICAL_SHIFT, position.width, position.height]
        )
    heatmap_table = panel_screening_heatmap(ax_e, fig, union_summary)
    letter_axes.append(("E", ax_e))

    if SHOW_DIRECTION_LEGEND:
        fig.legend(
            handles=[
                Patch(facecolor=RED, edgecolor=INK, linewidth=0.6, label="Increased toxicity"),
                Patch(facecolor=BLUE, edgecolor=INK, linewidth=0.6, label="Decreased toxicity"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, DIRECTION_LEGEND_Y),
            ncol=2,
            frameon=False,
            fontsize=DIRECTION_LEGEND_FONTSIZE,
            handlelength=1.7,
            columnspacing=2.4,
        )

    if ALIGN_PANEL_LETTERS_TO_TITLES:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inverse_figure = fig.transFigure.inverted()
        for letter, axis in letter_axes:
            title_box = axis.title.get_window_extent(renderer=renderer)
            title_center_y = title_box.y0 + title_box.height / 2
            figure_y = inverse_figure.transform((0, title_center_y))[1]
            fig.text(
                LETTER_X,
                figure_y,
                letter,
                fontsize=PANEL_LETTER_FONTSIZE,
                fontweight="bold",
                va="center",
            )
    else:
        for letter, axis in letter_axes:
            fig.text(
                LETTER_X,
                axis.get_position().y1 + PANEL_LETTER_Y_OFFSET,
                letter,
                fontsize=PANEL_LETTER_FONTSIZE,
                fontweight="bold",
            )

    if SHOW_FOOTNOTE:
        fig.text(
            0.5,
            FOOTNOTE_Y,
            f"Bar markers: * BH-FDR q<0.05; {EXPLORATORY_MARKER} exploratory 0.05<=q<0.10.  "
            "Heatmap color: fraction q<0.10; heatmap *: >=1 compound q<0.05.",
            ha="center",
            va="bottom",
            fontsize=FOOTNOTE_FONTSIZE,
            color=INK,
        )

    figures_dir = OUT_DIR / "figures"
    tables_dir = OUT_DIR / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    pdf = figures_dir / f"{FIG_STEM}.pdf"
    png = figures_dir / f"{FIG_STEM}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=340, bbox_inches="tight")
    plt.close(fig)

    top5_audit = pd.concat(top5_tables, ignore_index=True)
    top5_audit.to_csv(tables_dir / "Figure5_per_endpoint_top5_gsea_q010_marker_audit.csv", index=False)
    heatmap_table.to_csv(tables_dir / "Figure5_heatmap_q010_fraction_q005_significance_audit.csv", index=False)

    print("saved", png)
    print("saved", pdf)
    print("top5 marker counts", top5_audit["marker"].value_counts(dropna=False).to_dict())
    print(
        "heatmap confirmatory cells",
        int(heatmap_table["confirmed_enriched"].sum()),
        "of",
        len(heatmap_table),
    )


if __name__ == "__main__":
    main()
