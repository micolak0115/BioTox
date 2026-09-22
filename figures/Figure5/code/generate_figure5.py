#!/usr/bin/env python3
"""Final A4 Figure 5: complementation scatterplots plus rescue/correction heatmaps."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory


CODE_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((CODE_DIR / "config.json").read_text(encoding="utf-8"))
OUT = (CODE_DIR / CONFIG["output"]["directory"]).resolve()
SUMMARY_PATH = (CODE_DIR / CONFIG["input"]["compound_selection_statistics"]).resolve()
DISPLAYED_PATH = (CODE_DIR / CONFIG["input"]["heatmap_displayed_compounds"]).resolve()
CELLS_PATH = (CODE_DIR / CONFIG["input"]["heatmap_cells"]).resolve()

TASKS = ["NR-Aromatase", "SR-ARE", "SR-MMP", "SR-p53"]
ROLE_BY_Y = {1: "Rescue", 0: "Correction"}
# Widened from 15: two of the four endpoints' single largest-qualifying-
# increase rescue compounds (BRD-K11451237 at rank 17, Teniposide at rank
# 16) ranked just outside a top-15 cutoff and were missing a heatmap row
# despite being individually the strongest mover for their endpoint. 20
# captures both. Two remaining endpoint maxima -- SR-ARE's rescue maximum
# (SSR-69071, rank 37) and NR-Aromatase's correction maximum (voriconazole,
# rank 72) -- rank far outside any reasonably sized top-N cutoff; rather than
# stretching the cutoff to reach them, they are shown as their own row below
# a dashed gap at the bottom of their heatmap (see top_n_heatmaps). Both
# roles share the same top-N cutoff so the two heatmap panels stay the same
# height before their outlier rows are appended.
TOP_N_PER_ROLE = 20
Y_BY_ROLE = {role: y for y, role in ROLE_BY_Y.items()}
ACTIVE = "#B8423A"
INACTIVE = "#386FA4"
INK = "#20252B"
LIGHT = "#D9DDE1"
GRID = "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9.0,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrapped_name(value: str) -> str:
    if value.startswith("BRD-") and len(value) > 13:
        return value[:4] + "\n" + value[4:]
    return value


def clean_public_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={
        "mean_p_bio_off": "mean_probability_chem",
        "mean_p_bio_on": "mean_probability_chem_plus_bio",
    })


def display_compound_name(value: str) -> str:
    text = str(value)
    return text if text.upper() == text else text.title()


def selected_summary_cells(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    output["role"] = output["y"].map(ROLE_BY_Y)
    output["favorable_change"] = output["mean_aligned_delta_p"]
    output["selected_delta_p"] = np.where(output["selected"], output["mean_aligned_delta_p"], np.nan)
    output["compound_name"] = output["drug_name"].map(display_compound_name)
    output["tox21_label_conflict"] = False
    return output


def endpoint_maxima_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    selected = selected_summary_cells(summary)
    records = []
    for task in TASKS:
        for role, y in Y_BY_ROLE.items():
            candidates = selected.loc[
                selected["task"].eq(task) & selected["y"].eq(y) & selected["selected"]
            ]
            if candidates.empty:
                continue
            maximum = candidates.loc[candidates["mean_aligned_delta_p"].idxmax()].copy()
            records.append(maximum)
    return pd.DataFrame(records).reset_index(drop=True)


def top_n_heatmaps(summary, original_displayed):
    """Rank the complete selected universe by summed qualifying endpoint change."""
    all_cells = selected_summary_cells(summary)
    names = original_displayed.set_index(['role', 'ik'])['compound_name'].to_dict()
    all_cells['compound_name'] = [names.get((r.role, r.ik), r.compound_name.replace('-', ' ') if not r.compound_name.upper() == r.compound_name else r.compound_name)
                                  for r in all_cells.itertuples()]
    candidates = all_cells.groupby(['role', 'ik'], as_index=False).agg(
        drug_name=('drug_name', 'first'), compound_name=('compound_name', 'first'),
        n_evaluable_endpoints=('task', 'nunique'), n_selected_endpoints=('selected', 'sum'),
        sum_selected_delta_p=('selected_delta_p', 'sum'),
        maximum_selected_delta_p=('selected_delta_p', 'max'))
    candidates = candidates.loc[candidates.n_selected_endpoints.gt(0)].sort_values(
        ['role', 'sum_selected_delta_p', 'maximum_selected_delta_p', 'compound_name'],
        ascending=[True, False, False, True], kind='mergesort')
    candidates['candidate_rank'] = candidates.groupby('role').cumcount() + 1
    displayed = candidates.loc[candidates.candidate_rank.le(TOP_N_PER_ROLE)].copy()
    displayed['display_rank'] = displayed.candidate_rank
    displayed['panel'] = displayed.role.map({'Rescue': 'B', 'Correction': 'C'})
    cells = all_cells.merge(displayed[['role', 'ik', 'display_rank']], on=['role', 'ik'], how='inner')
    maxima = endpoint_maxima_from_summary(summary)
    maxima = maxima.merge(candidates[['role', 'ik', 'candidate_rank']], on=['role', 'ik'])
    maxima['displayed_in_heatmap'] = maxima.candidate_rank.le(TOP_N_PER_ROLE)

    # Endpoint maxima ranked far outside the top-N cutoff (e.g. rank 37 or
    # 72) are shown as their own gap-separated row at the bottom of the
    # heatmap instead of extending the ranked cutoff to an impractical size.
    outlier_iks = maxima.loc[~maxima.displayed_in_heatmap, ['role', 'ik']].drop_duplicates()
    outliers = candidates.merge(outlier_iks, on=['role', 'ik']).sort_values(['role', 'candidate_rank'])
    outliers['display_rank'] = outliers.candidate_rank
    outliers['panel'] = outliers.role.map({'Rescue': 'B', 'Correction': 'C'})
    outlier_cells = all_cells.merge(outliers[['role', 'ik', 'display_rank']], on=['role', 'ik'], how='inner')

    maxima['cell_available_for_outline'] = maxima.displayed_in_heatmap | maxima.ik.isin(outliers.ik)
    maxima['global_summary_maximum'] = True
    assert displayed.groupby('role').size().eq(TOP_N_PER_ROLE).all()
    return displayed, cells, maxima, candidates, outliers, outlier_cells


def assign_nonoverlapping_representatives(cells: pd.DataFrame) -> pd.DataFrame:
    """Assign each displayed compound to one selected endpoint, balancing label counts."""
    selected = cells.loc[cells["selected"]].copy()
    loads = {task: 0 for task in TASKS}
    groups = [group for _, group in selected.groupby(["role", "ik"], sort=False)]
    groups.sort(key=lambda group: (len(group), -float(group["selected_delta_p"].max())))
    representatives = []
    for group in groups:
        ranked = group.assign(_load=group["task"].map(loads)).sort_values(
            ["_load", "selected_delta_p"], ascending=[True, False], kind="mergesort"
        )
        chosen = ranked.iloc[0]
        representatives.append((chosen["role"], chosen["ik"], chosen["task"]))
        loads[str(chosen["task"])] += 1
    output = cells.copy()
    keys = set(representatives)
    output["representative_label"] = [
        (role, ik, task) in keys for role, ik, task in zip(output["role"], output["ik"], output["task"])
    ]
    return output


def annotate_representatives(ax: plt.Axes, task_cells: pd.DataFrame) -> None:
    selected = task_cells.loc[task_cells["selected"]]
    for role, color in (("Rescue", ACTIVE), ("Correction", INACTIVE)):
        role_rows = selected.loc[selected["role"].eq(role)]
        ax.scatter(
            role_rows["mean_p_bio_off"], role_rows["mean_p_bio_on"],
            s=43, facecolors="none", edgecolors=color, linewidth=1.0, zorder=6,
        )
        labels = role_rows.loc[role_rows["representative_label"]].sort_values(
            ["mean_p_bio_on", "compound_name"], ascending=[False, True]
        )
        if labels.empty:
            continue
        x_text = 0.025 if role == "Rescue" else 0.975
        horizontal = "left" if role == "Rescue" else "right"
        slots = np.linspace(0.82, 0.19, len(labels)) if len(labels) > 1 else np.asarray([0.72])
        for (_, row), y_text in zip(labels.iterrows(), slots):
            ax.annotate(
                row["compound_name"],
                xy=(row["mean_p_bio_off"], row["mean_p_bio_on"]), xycoords="data",
                xytext=(x_text, float(y_text)), textcoords="axes fraction",
                ha=horizontal, va="center", fontsize=5.7, fontweight="bold", color=color,
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.5},
                bbox={"facecolor": "none", "edgecolor": "none", "pad": 0.0}, zorder=7,
            )


def draw_scatter(ax: plt.Axes, task: str, summary: pd.DataFrame, cells: pd.DataFrame, bottom: bool) -> None:
    frame = summary.loc[summary["task"].eq(task)]
    rescue = frame.loc[frame["selected"] & frame["y"].eq(1)]
    correction = frame.loc[frame["selected"] & frame["y"].eq(0)]
    ax.scatter(frame["mean_p_bio_off"], frame["mean_p_bio_on"], s=5.0, color=LIGHT, alpha=0.55, linewidth=0)
    ax.scatter(rescue["mean_p_bio_off"], rescue["mean_p_bio_on"], s=15, color=ACTIVE, alpha=0.68,
               edgecolor=INK, linewidth=0.2, zorder=3)
    ax.scatter(correction["mean_p_bio_off"], correction["mean_p_bio_on"], s=15, color=INACTIVE, alpha=0.68,
               edgecolor=INK, linewidth=0.2, zorder=3)
    task_cells = cells.loc[cells["task"].eq(task) & cells["selected"]]
    for role, color, y_text in (("Rescue", ACTIVE, 0.89), ("Correction", INACTIVE, 0.78)):
        candidates = task_cells.loc[task_cells["role"].eq(role)]
        if candidates.empty:
            continue
        highest = candidates.loc[candidates["selected_delta_p"].idxmax()]
        ax.scatter([highest["mean_p_bio_off"]], [highest["mean_p_bio_on"]], s=52,
                   facecolors="none", edgecolors="#D62728", linewidth=1.35, zorder=6)
        ax.annotate(
            highest["compound_name"],
            xy=(highest["mean_p_bio_off"], highest["mean_p_bio_on"]), xycoords="data",
            xytext=(0.96, y_text), textcoords="axes fraction", ha="right", va="center",
            fontsize=7.2, fontweight="bold", color=color,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.65,
                        "connectionstyle": "angle3,angleA=180,angleB=90"},
            bbox={"facecolor": "none", "edgecolor": "none", "pad": 0.0}, zorder=7,
        )
    ax.plot([0, 1], [0, 1], color=INK, linestyle=(0, (3, 3)), linewidth=0.55)
    ax.set_title(task, fontsize=10.5, fontweight="bold", pad=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
    ax.tick_params(labelsize=8.2, length=2.2)
    if bottom:
        ax.set_xlabel("Predicted probability (Chem.)", fontsize=9.5, fontweight="bold")
    else:
        ax.set_xticklabels([])
    ax.grid(color="#D5DADF", linestyle=(0, (2, 2)), linewidth=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    active_total = int(frame["y"].eq(1).sum())
    inactive_total = int(frame["y"].eq(0).sum())
    selected_total = len(rescue) + len(correction)
    total = len(frame)
    ax.text(
        0.98, 0.025,
        f"Rescue: {len(rescue)}/{active_total} = {len(rescue)/active_total:.1%}\n"
        f"Correction: {len(correction)}/{inactive_total} = {len(correction)/inactive_total:.1%}\n"
        f"Total: {selected_total}/{total} = {selected_total/total:.1%}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8,
        color=INK, linespacing=1.12, zorder=8,
    )


def heatmap_matrix(
    displayed: pd.DataFrame, cells: pd.DataFrame, role: str,
    outliers: pd.DataFrame, outlier_cells: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, int | None]:
    order = displayed.loc[displayed["role"].eq(role)].sort_values("display_rank")
    role_cells = cells.loc[cells["role"].eq(role)]
    values = role_cells.pivot(index="ik", columns="task", values="selected_delta_p")
    values = values.reindex(index=order["ik"], columns=TASKS)
    matrix = np.column_stack([values.to_numpy(float), order["sum_selected_delta_p"].to_numpy(float)])

    # Endpoint maxima ranked far outside the top-N cutoff get their own row
    # below a dashed gap, instead of stretching the ranked cutoff to reach
    # them (see top_n_heatmaps).
    role_outliers = outliers.loc[outliers["role"].eq(role)].sort_values("display_rank")
    gap_row_index = None
    if len(role_outliers):
        gap_row_index = matrix.shape[0]
        outlier_role_cells = outlier_cells.loc[outlier_cells["role"].eq(role)]
        outlier_values = outlier_role_cells.pivot(index="ik", columns="task", values="selected_delta_p")
        outlier_values = outlier_values.reindex(index=role_outliers["ik"], columns=TASKS)
        outlier_matrix = np.column_stack(
            [outlier_values.to_numpy(float), role_outliers["sum_selected_delta_p"].to_numpy(float)]
        )
        gap = np.full((1, matrix.shape[1]), np.nan)
        matrix = np.vstack([matrix, gap, outlier_matrix])
        gap_placeholder = pd.DataFrame({"ik": [None], "compound_name": [None]})
        order = pd.concat([order, gap_placeholder, role_outliers], ignore_index=True)
    return order, matrix, gap_row_index


def draw_heatmap(
    ax: plt.Axes, displayed: pd.DataFrame, cells: pd.DataFrame, role: str, maxima: pd.DataFrame,
    outliers: pd.DataFrame, outlier_cells: pd.DataFrame,
):
    order, matrix, gap_row_index = heatmap_matrix(displayed, cells, role, outliers, outlier_cells)
    finite = matrix[np.isfinite(matrix)]
    norm = Normalize(vmin=0.0, vmax=max(0.01, float(finite.max())))
    cmap = plt.get_cmap("Reds" if role == "Rescue" else "Blues")
    rgba = np.ones(matrix.shape + (4,), dtype=float)
    for index, value in np.ndenumerate(matrix):
        if np.isfinite(value):
            rgba[index] = cmap(norm(value))
    if gap_row_index is not None:
        # Grey fill on the gap row itself signals the rank discontinuity
        # even before the reader reaches the ellipsis/spine break marks.
        rgba[gap_row_index, :, :] = mcolors.to_rgba(LIGHT)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    for row in range(matrix.shape[0] + 1):
        ax.axhline(row - 0.5, color=GRID, linewidth=0.75)
    if gap_row_index is None:
        for column in range(matrix.shape[1] + 1):
            ax.axvline(column - 0.5, color=GRID, linewidth=0.75)
    else:
        # Column separators apply within the top-N block and the outlier
        # row separately, leaving the grey gap band plain with no lines
        # crossing through it.
        for column in range(matrix.shape[1] + 1):
            ax.vlines(column - 0.5, -0.5, gap_row_index - 0.5, color=GRID, linewidth=0.75)
            ax.vlines(column - 0.5, gap_row_index + 0.5, matrix.shape[0] - 0.5, color=GRID, linewidth=0.75)
    if gap_row_index is None:
        ax.axvline(len(TASKS) - 0.5, color=INK, linewidth=1.0)
    else:
        # Break the sum-column divider across the gap row so it doesn't
        # visually bridge the rank discontinuity it introduces.
        ax.vlines(len(TASKS) - 0.5, -0.5, gap_row_index - 0.5, color=INK, linewidth=1.0)
        ax.vlines(len(TASKS) - 0.5, gap_row_index + 0.5, matrix.shape[0] - 0.5, color=INK, linewidth=1.0)
    for row in range(matrix.shape[0]):
        if row == gap_row_index:
            continue
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                continue
            color = "white" if norm(value) >= 0.55 else INK
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=8.0,
                    color=color, fontweight="bold" if column == len(TASKS) else "normal")
    if gap_row_index is not None:
        # Ellipsis in place of heatmap cells or a compound name, to make the
        # rank discontinuity explicit rather than implying adjacency.
        ax.text(len(TASKS) / 2 - 0.5, gap_row_index, r"$\vdots$", ha="center", va="center",
                fontsize=13, color=INK, fontweight="bold")
        # Conventional broken-axis cut mark (two parallel diagonal strokes)
        # on the y-axis spine itself, instead of a text ellipsis tick label.
        break_transform = blended_transform_factory(ax.transAxes, ax.transData)
        for stroke_center in (-0.13, 0.13):
            ax.plot([-0.018, 0.018],
                    [gap_row_index + stroke_center - 0.09, gap_row_index + stroke_center + 0.09],
                    transform=break_transform, color=INK, linewidth=1.3,
                    solid_capstyle="round", clip_on=False, zorder=10)
    role_cells = maxima.loc[maxima["role"].eq(role)]
    order_lookup = {ik: row for row, ik in enumerate(order["ik"]) if pd.notna(ik)}
    for column, task in enumerate(TASKS):
        candidates = role_cells.loc[role_cells["task"].eq(task)]
        if candidates.empty:
            continue
        highest = candidates.loc[candidates["selected_delta_p"].idxmax()]
        if str(highest["ik"]) not in order_lookup:
            continue
        row = order_lookup[str(highest["ik"])]
        ax.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False,
                               edgecolor="#D62728", linewidth=2.0, zorder=8))
    ax.set_yticks(np.arange(len(order)))
    yticklabels = [
        "" if row == gap_row_index else wrapped_name(name)
        for row, name in enumerate(order["compound_name"])
    ]
    ax.set_yticklabels(yticklabels, fontsize=9.5, color=ACTIVE if role == "Rescue" else INACTIVE)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks(np.arange(5), labels=["NR-Aromatase", "SR-ARE", "SR-MMP", "SR-p53", r"$\Sigma$"])
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False,
                   length=0, pad=2, labelsize=8.6)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_rotation_mode("anchor")
        label.set_ha("left")
        label.set_va("bottom")
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(len(order) - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_color(INK); spine.set_linewidth(0.7)
    return norm, cmap


def reposition_mmp_labels(ax, summary):
    """Keep both labels inside panel A and minimize actual marker overlap."""
    from matplotlib.text import Annotation
    renderer = ax.figure.canvas.get_renderer()
    scale = ax.figure.dpi / 72
    frame = summary.loc[summary.task.eq('SR-MMP')]
    points = ax.transData.transform(frame[['mean_p_bio_off', 'mean_p_bio_on']].to_numpy())
    selected = frame.selected.to_numpy(bool)
    radii = np.where(selected, 2.1, 1.1) * scale
    annotations = [t for t in ax.texts if isinstance(t, Annotation)]
    occupied = [t.get_window_extent(renderer).padded(scale) for t in ax.texts if not isinstance(t, Annotation)]
    bounds = ax.get_window_extent().padded(scale * -1)
    audit = []
    for annotation in annotations:
        original_name = annotation.get_text()
        variants = [original_name]
        if original_name.startswith('BRD-'):
            variants.append(original_name.replace('BRD-', 'BRD-\n', 1))
        elif '-' in original_name:
            head, _, tail = original_name.partition('-')
            variants.append(f"{head}-\n{tail}")
        elif len(original_name) > 8:
            # Generic mid-word hyphenated wrap: gives the search a narrower,
            # taller box to slot into gaps a single-line label cannot reach.
            mid = len(original_name) // 2
            variants.append(f"{original_name[:mid]}-\n{original_name[mid:]}")
        choices = []
        for name in variants:
            probe = ax.text(0, 0, name, transform=ax.transAxes, ha='right', va='center',
                            fontsize=annotation.get_fontsize(), fontweight='bold')
            for y in np.arange(.23, .98, .0125):
                for x in np.arange(.96, .39, -.025):
                    probe.set_position((x,y))
                    box = probe.get_window_extent(renderer).padded(.6*scale)
                    if not (bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1)):
                        continue
                    if any(box.overlaps(other) for other in occupied):
                        continue
                    hit = ((points[:,0]+radii >= box.x0)&(points[:,0]-radii <= box.x1)&
                           (points[:,1]+radii >= box.y0)&(points[:,1]-radii <= box.y1))
                    selected_hits, all_hits = int((hit&selected).sum()), int(hit.sum())
                    anchor = ax.transData.transform(annotation.xy)
                    distance = np.linalg.norm(np.array([box.x0, (box.y0+box.y1)/2])-anchor)
                    score = (selected_hits,all_hits,name.count('\n'),distance)
                    choices.append((score, float(x),float(y),name,box))
            probe.remove()
        if not choices:
            raise RuntimeError('No in-panel SR-MMP label location')
        score,x,y,name,box=min(choices,key=lambda item:item[0])
        annotation.set_text(name)
        annotation.set_ha('right')
        annotation.set_position((x,y))
        occupied.append(box.padded(scale))
        audit.append({'compound':original_name, 'display_label':name,'label_axes_position':[x,y],
                      'label_fully_inside_scatter':True, 'overlapping_selected_markers':score[0],
                      'overlapping_all_markers':score[1], 'overlaps_heatmap_labels':False})
    return audit


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_dir = OUT / "source_data"
    generation_dir = OUT / "generation"
    source_dir.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_PATH)
    # SUMMARY_PATH may be this script's own previously published source_data
    # (already renamed to the public column names by clean_public_columns
    # below) rather than the original internal-name working file; restore
    # the internal names the rest of the pipeline expects.
    summary = summary.rename(columns={
        "mean_probability_chem": "mean_p_bio_off",
        "mean_probability_chem_plus_bio": "mean_p_bio_on",
    })
    displayed, cells, maxima_audit, candidates, outliers, outlier_cells = top_n_heatmaps(summary, pd.read_csv(DISPLAYED_PATH))
    figure = plt.figure(figsize=(8.27, 8.65), facecolor='white')
    scatter_axes = []
    plot_top, plot_bottom = 0.842, 0.112
    scatter_width = 0.166
    scatter_height = scatter_width * 8.27 / 8.65
    step = (plot_top - plot_bottom - scatter_height) / 3
    for index, task in enumerate(TASKS):
        axis = figure.add_axes([0.067, plot_top-scatter_height-index*step, scatter_width, scatter_height])
        draw_scatter(axis, task, summary, maxima_audit, bottom=index == 3)
        scatter_axes.append(axis)
    rescue_ax = figure.add_axes([0.402, plot_bottom, 0.228, plot_top-plot_bottom])
    correction_ax = figure.add_axes([0.765, plot_bottom, 0.228, plot_top-plot_bottom])
    rescue_norm, rescue_cmap = draw_heatmap(rescue_ax, displayed, cells, 'Rescue', maxima_audit, outliers, outlier_cells)
    correction_norm, correction_cmap = draw_heatmap(correction_ax, displayed, cells, 'Correction', maxima_audit, outliers, outlier_cells)
    figure.canvas.draw()
    mmp_audit = reposition_mmp_labels(scatter_axes[2], summary)
    renderer = figure.canvas.get_renderer()
    header_top = max(t.get_window_extent(renderer).y1 for ax in [rescue_ax, correction_ax] for t in ax.get_xticklabels())
    title_y = header_top / figure.bbox.height + 0.010
    title_artists = []
    for x, label in [(0.012, 'A'), (0.362, 'B'), (0.725, 'C')]:
        title_artists.append(figure.text(x, title_y, label, fontsize=16, fontweight='bold', va='bottom'))
    for x, title, color in [(0.156, 'Complementation', INK), (0.516, 'Rescue', ACTIVE), (0.879, 'Correction', INACTIVE)]:
        title_artists.append(figure.text(x, title_y, title, fontsize=10.5, fontweight='bold',
                                       color=color, va='bottom', ha='center'))
    ylabel = figure.text(0.013, 0.50, 'Predicted probability (Chem. + Bio.)', rotation=90,
                        fontsize=9.5, fontweight='bold', ha='center', va='center')

    for axis, norm, cmap, label in (
        (rescue_ax, rescue_norm, rescue_cmap, "Probability increase\n" + r"$(P_{\mathrm{Chem.+Bio.}} - P_{\mathrm{Chem.}})$"),
        (correction_ax, correction_norm, correction_cmap, "Probability decrease\n" + r"$(P_{\mathrm{Chem.}} - P_{\mathrm{Chem.+Bio.}})$"),
    ):
        position = axis.get_position()
        colorbar_axis = figure.add_axes([position.x0, position.y0 - 0.027, position.width, 0.014])
        colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label(label, fontsize=9.0, fontweight="bold", labelpad=1, linespacing=0.95)
        colorbar.ax.tick_params(labelsize=8.0, length=2.0, pad=1)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    b_box = rescue_ax.get_window_extent(renderer)
    c_tick_boxes = [t.get_window_extent(renderer) for t in correction_ax.get_yticklabels()]
    b_tick_boxes = [t.get_window_extent(renderer) for t in rescue_ax.get_yticklabels()]
    scatter_right = max(ax.get_window_extent(renderer).x1 for ax in scatter_axes)
    bc_gap = min(box.x0 for box in c_tick_boxes) - b_box.x1
    ab_gap = min(box.x0 for box in b_tick_boxes) - scatter_right
    title_bottom = min(t.get_window_extent(renderer).y0 for t in title_artists)
    header_gap = title_bottom - header_top
    assert bc_gap > 3, f'Correction labels overlap rescue heatmap: {bc_gap}'
    assert ab_gap > 3, f'Rescue labels overlap scatterplots: {ab_gap}'
    assert header_gap > 3, f'Subtitles overlap endpoint headers: {header_gap}'
    layout_audit = {'rows_per_heatmap': displayed.groupby('role').size().to_dict(),
                    'gap_separated_outlier_rows_per_heatmap': outliers.groupby('role').size().to_dict(),
                    'subtitle_font_pt': 10.5, 'panel_letter_font_pt': 16,
                    'BC_label_clearance_pt': bc_gap*72/figure.dpi,
                    'AB_label_clearance_pt': ab_gap*72/figure.dpi,
                    'subtitle_to_endpoint_header_clearance_pt': header_gap*72/figure.dpi,
                    'SR_MMP_annotation_collision_check': mmp_audit,
                    'global_maxima_marked_in_scatter': len(maxima_audit),
                    'global_maxima_outlined_in_topN': int(maxima_audit.displayed_in_heatmap.sum()),
                    'global_maxima_outlined_total': int(maxima_audit.cell_available_for_outline.sum())}
    (generation_dir/'Figure5_layout_audit.json').write_text(json.dumps(layout_audit, indent=2)+'\n')
    candidates.to_csv(source_dir/'Figure5_all_compound_rankings.csv', index=False)
    figure.savefig(OUT/'Figure5_preview.png', dpi=150, bbox_inches='tight', pad_inches=0.04)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(OUT / f"Figure5.{suffix}", bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(figure)

    clean_public_columns(summary).to_csv(source_dir / "Figure5_compound_selection_statistics.csv", index=False)
    displayed.drop(columns=[column for column in displayed.columns if "clintox" in column.lower() or column in {"CT_TOX", "FDA_APPROVED"}], errors="ignore").to_csv(
        source_dir / "Figure5_heatmap_displayed_compounds.csv", index=False)
    clean_public_columns(cells).to_csv(source_dir / "Figure5_heatmap_cells.csv", index=False)

    caption = """# Figure 5 caption

**Figure 5 | Transcriptomic complementation and cross-endpoint rescue and correction.** **A**, held-out Chem. versus Chem. + Bio. predicted probabilities for four endpoint contexts. Red and blue points are formally selected rescue and correction calls. The compound with the largest qualifying probability change in each role and endpoint is identified by a hollow red ring and label. Lower-right statistics report selected/eligible counts and fractions for rescue, correction, and their combined total. **B**, the top 20 rescue compounds ranked by summed qualifying probability increases across endpoints. **C**, the top 20 correction compounds ranked by summed qualifying probability decreases across endpoints. A red outline identifies a global endpoint maximum when its compound falls within the top 20 for that role. Global endpoint maxima ranked far outside the top 20 are appended below a dashed gap as their own bottom row, showing their true per-endpoint probability change rather than extending the ranked cutoff to reach them. Endpoint cells are displayed only where the compound meets all selection criteria; blank cells are nonqualifying or not evaluable. The rightmost sum column contains the sum across qualifying endpoint cells. No comparator compounds are included.
"""
    methods = """# Figure 5 methods

Panel A combines rescue and correction calls without molecular or transcriptomic cluster overlays or a legend. Each endpoint scatterplot uses a square plotting region. Within each endpoint and role, the selected compound with the largest aligned probability change in the full compound-level summary is labeled in a separated annotation slot (with measured placement inside the SR-MMP scatterplot minimizing marker overlap) and marked by a hollow red ring. Lower-right statistics give the selected count, eligible denominator, percentage, and combined total.

Panels B and C contain formally selected compounds only. Selection uses the validated final statistics: one-sided compound-level signed-rank testing with BH q < 0.001 within endpoint and role, favorable direction in 20/20 repeats, and mean aligned probability change > 0.02. Candidates qualify in at least one endpoint and are ranked by the sum of qualifying endpoint changes. Exactly the first 20 compounds in each role ranking are displayed (widened from 15 so that each endpoint's single largest-qualifying-increase compound is captured wherever possible: this includes BRD-K11451237 and Teniposide, the SR-MMP and SR-p53 rescue maxima, which ranked just outside a 15-compound cutoff). Ties are broken by maximum qualifying endpoint change and then compound name. Full-summary endpoint maxima ranked far outside the top 20 -- SR-ARE's rescue maximum (SSR-69071, rank 37) and NR-Aromatase's correction maximum (voriconazole, rank 72) -- are not used to extend the ranked cutoff; instead each is appended as its own row below a dashed gap-and-ellipsis divider at the bottom of its heatmap, reporting its actual per-endpoint probability change rather than only an annotation in panel A. All global role-and-endpoint maximum cells, whether within the top 20 or in a gap-separated outlier row, are outlined in red. No SR-ARE or other in-stratum comparator is introduced.
"""
    (OUT / "caption.md").write_text(caption, encoding="utf-8")
    (OUT / "methods.md").write_text(methods, encoding="utf-8")
    clean_public_columns(maxima_audit).to_csv(generation_dir / "Figure5_global_endpoint_maxima_audit.csv", index=False)
    manifest = {
        "figure": "Figure 5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "layout": "compact portrait; four square scatterplots; height-matched rescue and correction heatmaps; tightly cropped",
        "cluster_overlays": False,
        "comparators_in_heatmaps": False,
        "annotation": "one full-summary maximum rescue and correction per endpoint; hollow red rings; separated fixed slots; matching heatmap cells outlined red",
        "heatmap_rows": "exactly top 20 per role, ranked across full selected summary by summed qualifying changes; outside-top20 endpoint maxima appended as a gap-separated outlier row instead of extending the cutoff",
        "inputs": {str(path): sha256(path) for path in (SUMMARY_PATH, DISPLAYED_PATH, CELLS_PATH)},
        "outputs": {},
    }
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "Figure5_manifest.json":
            manifest["outputs"][path.relative_to(OUT).as_posix()] = sha256(path)
    (generation_dir / "Figure5_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
