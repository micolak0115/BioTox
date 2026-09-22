#!/usr/bin/env python3
"""A4 portrait Figure S2 with uncluttered scatterplots and vertical chemical pairs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdFMCS, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


CODE_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((CODE_DIR / "config.json").read_text(encoding="utf-8"))
DEFAULT_OUT = (CODE_DIR / CONFIG["output"]["directory"]).resolve()
OUT = Path(os.environ.get("BIOTOX_FIGURE5_A4_OUTPUT", DEFAULT_OUT))
SOURCE = OUT / "source_data"
GENERATION = OUT / "generation"
UNIVERSE_PATH = (CODE_DIR / CONFIG["input"]["compound_selection_statistics"]).resolve()
EXEMPLAR_PATH = (CODE_DIR / CONFIG["input"]["displayed_vertical_structure_cases"]).resolve()

TASKS = ["NR-Aromatase", "SR-MMP", "SR-p53"]
RESCUE = "#B8423A"
CORRECTION = "#386FA4"
MOLECULAR_ALIGNMENT = "#7A3E9D"
BIOLOGICAL_ALIGNMENT = "#C87418"
SHARED_ATOM = (0.33, 0.68, 0.48)
FIRST_ATOM = (0.78, 0.32, 0.46)
SECOND_ATOM = (0.25, 0.52, 0.78)
INK = "#20252B"
GREY = "#D9DDE1"

RDLogger.DisableLog("rdApp.*")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10.0,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_name(value: object, ik: str) -> str:
    if pd.isna(value) or str(value).strip().lower() in {"", "none", "nan", "na"}:
        return ik
    value = str(value).strip()
    return value.upper() if value.upper().startswith("BRD-") else value.replace("-", " ").title()


def mcs_highlights(first: Chem.Mol, second: Chem.Mol) -> dict:
    result = rdFMCS.FindMCS(
        [first, second], timeout=20, ringMatchesRingOnly=True,
        completeRingsOnly=True, bondCompare=rdFMCS.BondCompare.CompareOrder,
    )
    query = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
    matches = [molecule.GetSubstructMatch(query) if query is not None else tuple() for molecule in (first, second)]
    atom_lists, atom_colors, bond_lists, bond_colors = [], [], [], []
    for index, (molecule, match) in enumerate(zip((first, second), matches)):
        shared = set(match)
        unique = FIRST_ATOM if index == 0 else SECOND_ATOM
        atom_lists.append(list(range(molecule.GetNumAtoms())))
        atom_colors.append({atom: SHARED_ATOM if atom in shared else unique for atom in range(molecule.GetNumAtoms())})
        bond_lists.append(list(range(molecule.GetNumBonds())))
        bond_colors.append({
            bond.GetIdx(): SHARED_ATOM
            if bond.GetBeginAtomIdx() in shared and bond.GetEndAtomIdx() in shared else unique
            for bond in molecule.GetBonds()
        })
    return {
        "atom_map": list(zip(matches[0], matches[1])),
        "mcs_smarts": result.smartsString,
        "mcs_atoms": int(result.numAtoms),
        "first_mcs_coverage": result.numAtoms / first.GetNumAtoms(),
        "second_mcs_coverage": result.numAtoms / second.GetNumAtoms(),
        "atom_lists": atom_lists,
        "atom_colors": atom_colors,
        "bond_lists": bond_lists,
        "bond_colors": bond_colors,
    }


def label_entries(task: str, cases: pd.DataFrame, universe: pd.DataFrame) -> list[dict]:
    task_cases = cases.loc[cases["task"].eq(task)]
    lookup = universe.set_index(["task", "role", "ik"])
    entries = {}
    for case in task_cases.itertuples(index=False):
        color = MOLECULAR_ALIGNMENT if case.pair_type == "V" else BIOLOGICAL_ALIGNMENT
        for ik, name in ((case.first_ik, case.first_name), (case.second_ik, case.second_name)):
            row = lookup.loc[(task, case.role, ik)]
            entries[(case.role, ik)] = {
                "role": case.role, "ik": ik, "name": name, "color": color,
                "x": float(row.mean_p_bio_off), "y": float(row.mean_p_bio_on),
                "selected": bool(row.selected),
            }
    return sorted(entries.values(), key=lambda item: (-item["y"], item["x"]))


def rectangles_overlap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    return not (first[2] < second[0] or second[2] < first[0] or first[3] < second[1] or second[3] < first[1])


LABEL_AUDIT = []


def annotate_without_overlap(ax: plt.Axes, entries: list[dict], point_x: np.ndarray, point_y: np.ndarray) -> None:
    """Use measured text extents; exhaust adjacent placements before a short leader."""
    from matplotlib.transforms import Bbox
    renderer = ax.figure.canvas.get_renderer()
    scale = ax.figure.dpi / 72
    points = ax.transData.transform(np.column_stack([point_x, point_y]))
    occupied = []
    highlighted = ax.transData.transform([(e['x'], e['y']) for e in entries])
    bounds = ax.get_window_extent().padded(-2 * scale)
    for entry in sorted(entries, key=lambda e: (-e['y'], e['name'])):
        ax.scatter([entry['x']], [entry['y']], s=62, facecolors='none',
                   edgecolors=entry['color'], linewidth=1.35, zorder=7)
        anchor = ax.transData.transform((entry['x'], entry['y']))
        other = points[np.linalg.norm(points - anchor, axis=1) > 2 * scale]
        candidates = [(7, 0, 'left'), (-7, 0, 'right'), (0, 8, 'center'), (0, -8, 'center'),
                      (6, 7, 'left'), (-6, 7, 'right'), (6, -7, 'left'), (-6, -7, 'right')]
        # Extra locations are ordered by physical distance to the target point.
        for distance in (16, 23, 32, 44, 60, 80, 105, 135):
            for angle in (0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330):
                dx = distance * np.cos(np.deg2rad(angle))
                dy = distance * np.sin(np.deg2rad(angle))
                candidates.append((dx, dy, 'left' if dx > 3 else 'right' if dx < -3 else 'center'))
        chosen = None
        for index, (dx, dy, ha) in enumerate(candidates):
            label = ax.annotate(entry['name'], (entry['x'], entry['y']),
                                xytext=(dx, dy), textcoords='offset points', ha=ha, va='center',
                                fontsize=9.0, fontweight='bold', color=entry['color'], zorder=8)
            box = label.get_window_extent(renderer).padded(1.7 * scale)
            inside = bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
            collisions = any(box.overlaps(old) for old in occupied)
            point_collision = np.any((other[:, 0] >= box.x0 - 1.8 * scale) &
                                     (other[:, 0] <= box.x1 + 1.8 * scale) &
                                     (other[:, 1] >= box.y0 - 1.8 * scale) &
                                     (other[:, 1] <= box.y1 + 1.8 * scale))
            leader_collision = False
            if index >= 8:
                end = np.array([np.clip(anchor[0], box.x0, box.x1), np.clip(anchor[1], box.y0, box.y1)])
                direction = end - anchor
                other_rings = highlighted[np.linalg.norm(highlighted-anchor, axis=1) > 2*scale]
                t = np.clip((other_rings-anchor) @ direction / max(float(direction @ direction), 1), 0, 1)
                leader_collision = np.any(np.linalg.norm(other_rings - (anchor + t[:, None]*direction), axis=1) < 6*scale)
            if inside and not collisions and not point_collision and not leader_collision:
                chosen = (index, dx, dy, ha, box, label)
                break
            label.remove()
        if chosen is None:
            raise RuntimeError(f"No collision-free label placement: {entry['name']}")
        index, dx, dy, ha, box, label = chosen
        occupied.append(box)
        if index >= 8:
            label.remove()
            ax.annotate(entry['name'], (entry['x'], entry['y']), xytext=(dx, dy),
                        textcoords='offset points', ha=ha, va='center', fontsize=9.0,
                        fontweight='bold', color=entry['color'], zorder=8,
                        arrowprops={'arrowstyle': '-', 'color': entry['color'],
                                    'linewidth': 0.7, 'shrinkA': 2, 'shrinkB': 5})
        LABEL_AUDIT.append({'task': ax.get_title(), 'name': entry['name'],
                            'leader': index >= 8, 'offset_points': [float(dx), float(dy)],
                            'bounds_pixels': list(box.extents), 'adjacent_candidates_exhausted': index >= 8})


def draw_scatter(ax: plt.Axes, task: str, universe: pd.DataFrame, cases: pd.DataFrame) -> None:
    group = universe.loc[universe["task"].eq(task)]
    selected = group.loc[group["selected"]]
    ax.scatter(group["mean_p_bio_off"], group["mean_p_bio_on"], s=4.0, color=GREY, alpha=0.43, linewidth=0)
    for role, color in (("Rescue", RESCUE), ("Correction", CORRECTION)):
        subset = selected.loc[selected["role"].eq(role)]
        ax.scatter(subset["mean_p_bio_off"], subset["mean_p_bio_on"], s=13, color=color,
                   edgecolor=INK, linewidth=0.2, alpha=0.72, zorder=3)
    ax.plot([0, 1], [0, 1], color=INK, linestyle=(0, (3, 3)), linewidth=0.55)
    rescue_n = int(selected["role"].eq("Rescue").sum())
    correction_n = int(selected["role"].eq("Correction").sum())
    # The endpoint name is no longer a plot title -- it is drawn instead as
    # part of a per-row [big letter][endpoint name] header above the whole
    # row (see draw_figure), leftmost and much larger than the endpoint name.
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
    ax.tick_params(labelsize=11.0, length=2.5)
    # Every scatterplot now carries its own x- and y-axis label instead of
    # sharing one across all three rows, so each endpoint's panel stands alone.
    ax.set_xlabel("Predicted probability (Chem.)", fontsize=13.5, fontweight="bold")
    ax.set_ylabel("Predicted probability (Chem.+Bio.)", fontsize=12.5, fontweight="bold")
    ax.grid(color="#D5DADF", linestyle=(0, (2, 2)), linewidth=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def molecule_image(molecule: Chem.Mol, highlight: dict, index: int) -> np.ndarray:
    options = rdMolDraw2D.MolDrawOptions()
    options.baseFontSize = 1.2
    options.fixedFontSize = 180
    options.minFontSize = 96
    options.maxFontSize = 384
    options.scalingFactor = 120
    options.bondLineWidth = 12
    options.highlightBondWidthMultiplier = 4
    image = Draw.MolsToGridImage(
        [molecule], molsPerRow=1, subImgSize=(3720, 1860), useSVG=False, drawOptions=options,
        highlightAtomLists=[highlight["atom_lists"][index]],
        highlightAtomColors=[highlight["atom_colors"][index]],
        highlightBondLists=[highlight["bond_lists"][index]],
        highlightBondColors=[highlight["bond_colors"][index]],
    ).convert("RGB")
    array = np.asarray(image)
    content = np.any(array < 248, axis=2)
    ys, xs = np.where(content)
    if len(xs):
        pad = 60
        array = array[max(0, ys.min()-pad):min(array.shape[0], ys.max()+pad),
                      max(0, xs.min()-pad):min(array.shape[1], xs.max()+pad)]
    return array


def _mathtext_bold(text: str) -> str:
    """Bold a text fragment for use inside an otherwise-plain mpl string.
    Any hyphen (e.g. inside "BRD-K...") is kept outside the math span, since
    mathtext renders a bare "-" inside $...$ as a wide minus-sign glyph
    rather than a normal short hyphen."""
    def escape(fragment: str) -> str:
        escaped = fragment
        for character in ('\\', '_', '^', '$', '{', '}', '#', '&', '%'):
            escaped = escaped.replace(character, '\\' + character)
        return escaped.replace(' ', r'\ ')
    return '-'.join(r'$\mathbf{' + escape(part) + '}$' for part in text.split('-'))


def compute_uniform_direction_size(
    cases: pd.DataFrame, axes_width_fraction: float, max_size: float = 12.0, min_size: float = 6.0,
) -> float:
    """The largest single-line fontsize that fits every case's "First vs. Second"
    line -- found once, across all cases, so every panel uses the identical
    size (rather than each shrinking independently to its own longest name)."""
    probe_fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
    probe_ax = probe_fig.add_axes([0.0, 0.0, axes_width_fraction, 0.1])
    probe_fig.canvas.draw()
    renderer = probe_fig.canvas.get_renderer()
    axes_width = probe_ax.get_window_extent(renderer).width
    texts = [
        probe_ax.text(
            0.5, 0.5,
            f"({_mathtext_bold(case.first_name)} vs. {_mathtext_bold(case.second_name)})",
            transform=probe_ax.transAxes, ha="center", va="center", fontsize=max_size,
        )
        for case in cases.itertuples(index=False)
    ]
    size = max_size
    probe_fig.canvas.draw()
    while size > min_size and max(t.get_window_extent(renderer).width for t in texts) > 0.94 * axes_width:
        size -= 0.3
        for t in texts:
            t.set_fontsize(size)
        probe_fig.canvas.draw()
    plt.close(probe_fig)
    return size


def draw_case(ax: plt.Axes, case: pd.Series, compact: bool, direction_size: float) -> None:
    ax.axis("off")
    first = Chem.MolFromSmiles(case["first_smiles"])
    second = Chem.MolFromSmiles(case["second_smiles"])
    highlights = mcs_highlights(first, second)
    rdDepictor.Compute2DCoords(first)
    rdDepictor.GenerateDepictionMatching2DStructure(second, first, highlights["atom_map"])
    first_coords = first.GetConformer().GetPositions()
    second_coords = second.GetConformer().GetPositions()
    assert all(np.allclose(first_coords[i], second_coords[j], atol=1e-6)
               for i, j in highlights["atom_map"]), "Shared scaffold alignment failed"
    color = MOLECULAR_ALIGNMENT if case["pair_type"] == "V" else BIOLOGICAL_ALIGNMENT
    name_size = 13.0
    detail_size = 10.5
    ax.text(0.5, 0.985, case["first_name"], transform=ax.transAxes, ha="center", va="top",
            fontsize=name_size, fontweight="bold", color=color, zorder=1000, clip_on=False)
    first_ax = ax.inset_axes([0.04, 0.640, 0.92, 0.270], zorder=1)
    first_ax.imshow(molecule_image(first, highlights, 0)); first_ax.axis("off")
    ax.text(0.5, 0.595, case["second_name"], transform=ax.transAxes, ha="center", va="top",
            fontsize=name_size, fontweight="bold", color=color, zorder=1000, clip_on=False)
    second_ax = ax.inset_axes([0.04, 0.245, 0.92, 0.270], zorder=1)
    second_ax.imshow(molecule_image(second, highlights, 1)); second_ax.axis("off")
    # Pair names use “vs.”; signed differences follow the displayed first-minus-second order.
    direction = f"({_mathtext_bold(case['first_name'])} vs. {_mathtext_bold(case['second_name'])})"
    ax.text(0.5, 0.225, direction, transform=ax.transAxes, ha="center",
            va="top", fontsize=direction_size, color=INK)
    # Each statistic is colored in the pair's alignment color only when it is
    # the defining/notable value for that selection category (matching the
    # panel B/C selection criteria): high Tanimoto and large Δp(Chem.+Bio.)
    # for molecularly conserved pairs; large Δp(Chem.) for transcriptomically
    # convergent pairs. The other statistics stay the default ink color.
    conserved = case["pair_type"] == "V"
    stat_lines = [
        (f"{_mathtext_bold('Morgan Tanimoto:')} {case['morgan_tanimoto']:.2f}", conserved),
        (f"{_mathtext_bold('Δp (Chem.):')} {case['delta_x_bio_off_probability']:+.3f}", not conserved),
        (f"{_mathtext_bold('Δp (Chem.+Bio.):')} {case['delta_y_bio_on_probability']:+.3f}", conserved),
    ]
    stat_top, stat_step = 0.170, 0.070
    for offset, (text, notable) in enumerate(stat_lines):
        ax.text(0.5, stat_top - offset * stat_step, text, transform=ax.transAxes,
                ha="center", va="top", fontsize=detail_size, color=color if notable else INK)


def draw_figure(universe: pd.DataFrame, cases: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
    molecular = cases.loc[cases['pair_type'].eq('V')].set_index('task')
    biological = cases.loc[cases['pair_type'].eq('H')].set_index('task')
    direction_size = compute_uniform_direction_size(cases, axes_width_fraction=0.257)
    scatter_axes, molecular_axes, biological_axes = [], [], []
    scatter_x0 = 0.088
    width = 0.32
    square_height = width * 210 / 297
    # Each row is a single panel allocated to one endpoint: one big letter at
    # the row's top-left corner, and one endpoint name horizontally centered
    # above the row's second (molecular-example) column, represent all three
    # horizontally-placed subplots in that row -- written once per row
    # rather than once per subplot.
    row_top0 = 0.955
    row_step = 0.300
    header_gap = 0.010
    letter_size = 26
    endpoint_size = 17
    row_letters = ['A', 'B', 'C']
    molecular_x_center = 0.442 + 0.257 / 2
    for index, task in enumerate(TASKS):
        top = row_top0 - index * row_step
        scatter_ax = figure.add_axes([scatter_x0, top - square_height, width, square_height])
        draw_scatter(scatter_ax, task, universe, cases)
        scatter_ax.set_aspect('equal', adjustable='box')
        molecular_ax = figure.add_axes([0.442, top - square_height, 0.257, square_height])
        biological_ax = figure.add_axes([0.724, top - square_height, 0.257, square_height])
        draw_case(molecular_ax, molecular.loc[task], compact=False, direction_size=direction_size)
        draw_case(biological_ax, biological.loc[task], compact=False, direction_size=direction_size)
        scatter_axes.append(scatter_ax)
        molecular_axes.append(molecular_ax)
        biological_axes.append(biological_ax)
        header_y = top + header_gap
        # The letter sits above and to the left of the y-axis label rather
        # than merely above the axes box: measure the scatterplot's actual
        # rendered left edge (which includes its y-axis label and ticks),
        # then place the letter a further clear gap to the left of that, so
        # it visibly encompasses/precedes the y-axis label rather than just
        # touching its edge.
        figure.canvas.draw()
        tight_left_px = scatter_ax.get_tightbbox(figure.canvas.get_renderer()).x0
        tight_left = figure.transFigure.inverted().transform((tight_left_px, 0))[0]
        letter_x = tight_left - 0.016
        figure.text(letter_x, header_y, row_letters[index], fontsize=letter_size,
                    fontweight='bold', va='bottom', ha='left', color=INK)
        figure.text(molecular_x_center, header_y, task, fontsize=endpoint_size,
                    fontweight='bold', va='bottom', ha='center', color=INK,
                    zorder=1000, clip_on=False)
    scatter_x_center = scatter_x0 + width / 2
    marker_handles = [
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=GREY,
               markeredgecolor='none', markersize=6.5, label='Not selected'),
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=RESCUE,
               markeredgecolor=INK, markeredgewidth=0.4, markersize=7.5, label='Rescued'),
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=CORRECTION,
               markeredgecolor=INK, markeredgewidth=0.4, markersize=7.5, label='Corrected'),
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='none',
               markeredgecolor=MOLECULAR_ALIGNMENT, markeredgewidth=1.4, markersize=9.0,
               label='Molecularly conserved'),
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='none',
               markeredgecolor=BIOLOGICAL_ALIGNMENT, markeredgewidth=1.4, markersize=9.0,
               label='Transcriptomically convergent'),
    ]
    structure_handles = [
        Line2D([0], [0], marker='s', linestyle='none', markerfacecolor=SHARED_ATOM,
               markeredgecolor='none', markersize=9.0, label='Shared substructure'),
        Line2D([0], [0], marker='s', linestyle='none', markerfacecolor=FIRST_ATOM,
               markeredgecolor='none', markersize=9.0, label='First-compound-specific'),
        Line2D([0], [0], marker='s', linestyle='none', markerfacecolor=SECOND_ATOM,
               markeredgecolor='none', markersize=9.0, label='Second-compound-specific'),
    ]
    # Measure the last row's actual rendered bottom (axes plus its x-axis
    # label) rather than assuming a fixed offset, so the legend never
    # collides with it regardless of axis-label font size.
    figure.canvas.draw()
    last_row_bottom_px = scatter_axes[-1].get_tightbbox(figure.canvas.get_renderer()).y0
    last_row_bottom = figure.transFigure.inverted().transform((0, last_row_bottom_px))[1]
    legend_row1_y = last_row_bottom - 0.018
    legend_row2_y = legend_row1_y - 0.028
    marker_legend = figure.legend(
        handles=marker_handles, loc='upper center',
        bbox_to_anchor=(0.5, legend_row1_y), bbox_transform=figure.transFigure,
        ncol=5, fontsize=9.5, frameon=False, handletextpad=0.4, columnspacing=1.3,
        borderaxespad=0.0,
    )
    figure.add_artist(marker_legend)
    figure.legend(
        handles=structure_handles, loc='upper center',
        bbox_to_anchor=(0.5, legend_row2_y), bbox_transform=figure.transFigure,
        ncol=3, fontsize=9.5, frameon=False, handletextpad=0.4, columnspacing=1.3,
        borderaxespad=0.0,
    )
    figure.canvas.draw()
    for ax, task in zip(scatter_axes, TASKS):
        group = universe.loc[universe['task'].eq(task)]
        annotate_without_overlap(ax, label_entries(task, cases, universe),
                                 group['mean_p_bio_off'].to_numpy(float),
                                 group['mean_p_bio_on'].to_numpy(float))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    compound_label_gaps = []
    for ax in molecular_axes + biological_axes:
        for label, structure_ax in zip(ax.texts[:2], ax.child_axes):
            gap = label.get_window_extent(renderer).y0 - structure_ax.get_window_extent(renderer).y1
            assert gap > 0, f"Compound label overlaps structure: {label.get_text()} ({gap}px)"
            assert label.get_zorder() > structure_ax.get_zorder()
            compound_label_gaps.append({"name": label.get_text(), "gap_px": float(gap)})
    row_bounds = [[ax.get_position().bounds for ax in triplet]
                  for triplet in zip(scatter_axes, molecular_axes, biological_axes)]
    audit = {'compound_label_clearance': compound_label_gaps, 'page_mm': [210, 297], 'labels': LABEL_AUDIT,
             'scatter_aspect_ratios': [ax.get_window_extent().width / ax.get_window_extent().height for ax in scatter_axes],
             'BC_matching_bounds': [bool(np.allclose(b.get_position().bounds[1::2], c.get_position().bounds[1::2]))
                                    for b, c in zip(molecular_axes, biological_axes)],
             'ABC_row_top_bottom_aligned': [
                 bool(np.allclose([bounds[1] for bounds in row], row[0][1]) and
                      np.allclose([bounds[1] + bounds[3] for bounds in row], row[0][1] + row[0][3]))
                 for row in row_bounds],
             'headers': {'letter_fontsize': 26, 'subtitle_fontsize': 17, 'subtitle_color': INK},
             'removed_detail_lines': ['Common substructure', 'Coverage'],
             'shared_scaffold_coordinate_alignment': 'all six pairs passed atom-coordinate equality at atol=1e-6',
             'probability_separation_validation': 'all six pairs matched Figure5 per-compound probabilities at atol=1e-12',
             'probability_annotation': 'Δp = p(first) - p(second) within each model'}
    assert all(abs(r-1) < 1e-8 for r in audit['scatter_aspect_ratios'])
    assert all(audit['BC_matching_bounds'])
    assert all(audit['ABC_row_top_bottom_aligned'])
    (GENERATION / 'FigureS2_layout_audit.json').write_text(json.dumps(audit, indent=2))
    for suffix in ('png', 'pdf', 'svg'):
        save_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.03, 'dpi': 900}
        figure.savefig(OUT / f'FigureS2.{suffix}', **save_kwargs)
    figure.savefig(OUT / 'FigureS2_preview.png', dpi=150, bbox_inches='tight', pad_inches=0.03)
    plt.close(figure)


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    GENERATION.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(UNIVERSE_PATH).rename(columns={
        "mean_probability_chem": "mean_p_bio_off",
        "mean_probability_chem_plus_bio": "mean_p_bio_on",
    })
    universe["role"] = np.where(universe["y"].eq(1), "Rescue", "Correction")
    cases = pd.read_csv(EXEMPLAR_PATH).rename(columns={
        "delta_chem_probability": "delta_x_bio_off_probability",
        "delta_chem_plus_bio_probability": "delta_y_bio_on_probability",
    })
    lookup = universe.set_index(["task", "ik"])
    for case in cases.itertuples(index=False):
        first = lookup.loc[(case.task, case.first_ik)]
        second = lookup.loc[(case.task, case.second_ik)]
        assert first.selected and second.selected
        assert np.isclose((first.mean_p_bio_off - second.mean_p_bio_off),
                          case.delta_x_bio_off_probability, atol=1e-12, rtol=0)
        assert np.isclose((first.mean_p_bio_on - second.mean_p_bio_on),
                          case.delta_y_bio_on_probability, atol=1e-12, rtol=0)
    assert cases.loc[cases["pair_type"].eq("V"), "delta_y_bio_on_probability"].gt(0).all()
    assert cases.loc[cases["pair_type"].eq("H"), "delta_x_bio_off_probability"].gt(0).all()
    draw_figure(universe, cases)

    methods = """# Figure S2 chemical-alignment-atlas methods

Each row is a single panel allocated to one endpoint (A: NR-Aromatase, B: SR-MMP, C: SR-p53; SR-ARE is omitted because no selected-selected example met the complete criteria). One large row letter at the row's top-left corner and one endpoint name, horizontally centered above the row's second (molecular-example) column, represent the entire panel; they are written once per row rather than once per sub-plot. Each row's three horizontally-placed sub-plots -- an endpoint scatterplot, a molecularly conserved/transcriptomically divergent example, and a transcriptomically convergent/structurally divergent example -- are otherwise undecorated by any further lettering. Every scatterplot carries its own x- and y-axis label (no longer shared across rows), enlarged for legibility. No cluster boundary or cluster ellipse is drawn. Example compounds are marked by hollow alignment-colored rings so the underlying rescue or correction point remains visible. Labels are first placed immediately beside their datapoints using collision checks against all points and previously placed labels. A short leader line is used only when all eight adjacent label placements collide; the nearest collision-free position is used within the scatterplot.

The molecularly conserved/transcriptomically divergent example for each endpoint satisfies absolute Chem. probability separation <=0.05, absolute Chem.+Bio. probability separation >=0.10, common endpoint/role-specific Morgan/Butina cluster membership at Tanimoto >=0.60, and different residualized-transcriptomic clusters. The transcriptomically convergent/structurally divergent example satisfies absolute Chem.+Bio. probability separation <=0.05, absolute Chem. probability separation >=0.10, common endpoint/role-specific residualized-LINCS cluster membership at Spearman correlation approximately >=0.40, and different molecular clusters. Each example reports a "(first compound vs. second compound)" line directly above its statistics, in parentheses, plain black, and joined by “vs.” (distinct from any single hyphen inside a compound's own name, e.g. BRD-K...), naming which compound is displayed first (above) and which is displayed second (below); its fontsize is reduced per case only as needed to keep it on a single line. Statistics are reported as Morgan Tanimoto similarity and Δp (Chem.) and Δp (Chem.+Bio.), signed first-minus-second probability differences; the selection criteria above use their magnitudes; the prior transcriptomic-correlation statistic has been removed. Statistic category labels are bold and their numeric values are regular weight for readability. Each statistic is colored in that pair's alignment color only when it is the defining/notable value for that selection category -- Morgan Tanimoto and Δp (Chem.+Bio.) for molecularly conserved pairs, Δp (Chem.) for transcriptomically convergent pairs -- and left in the default ink color otherwise, so the statistic that actually drove each pair's classification is visually highlighted.

No selected-selected SR-ARE pair met either complete characteristic set. Mixed selected/comparator SR-ARE candidates were excluded, and no unsupported SR-ARE example was substituted.

Purple pairs are ordered with the higher Chem.+Bio. probability first, so their signed Δp (Chem.+Bio.) is positive. Orange pairs are ordered with the higher Chem. probability first, so their signed Δp (Chem.) is positive. Highlight bands are widened to emphasize substructure differences. Molecular images are rendered at 3720 × 1860 pixels with enlarged atom labels (RDKit fixedFontSize 180 pixels on each molecular rendering canvas); all final formats are exported at 900 dpi for embedded molecular images. Structures are displayed vertically, with the second molecule constrained to the first molecule’s 2D coordinates at all matched maximum-common-substructure atoms for consistent scaffold orientation. Δp = p(first compound) − p(second compound) within each model; it does not denote the within-compound probability increment from adding Bio. Pairwise maximum common substructures were recomputed with RDKit FMCS using ring-only ring matching, complete rings, and bond-order matching. Green atoms are shared, magenta ("first-compound-specific" in the legend) atoms are unique to the compound displayed first, and blue atoms are unique to the compound displayed second. These are descriptive alignments rather than causal toxicophore assignments.

A two-row legend below the panels gives, in its first row, the not-selected/rescued/corrected scatterplot marker styles and the molecularly-conserved/transcriptomically-convergent example-ring colors, and in its second row, the shared/first-compound/second-compound substructure-highlight colors used in every molecule image.
"""
    (OUT / "methods.md").write_text(methods, encoding="utf-8")
    caption = """**Figure S2 | Chemical structures underlying selected molecular and transcriptomic alignment examples.** Each row is a single panel for one endpoint (A: NR-Aromatase, B: SR-MMP, C: SR-p53; SR-ARE is omitted because no selected-selected SR-ARE pair met the complete prespecified characteristics), identified once by a large row letter and endpoint name representing all three of its horizontally-placed sub-plots. The first sub-plot in each row shows Chem. versus Chem.+Bio. predicted probability for that endpoint, each with its own axis labels. Hollow purple and orange rings identify the compounds displayed in the other two sub-plots without obscuring their underlying point status. Compound names are placed directly beside datapoints when collision-free; short leader lines are used only when adjacent placements would overlap. The second sub-plot shows that endpoint's molecularly conserved and transcriptomically divergent selected rescue pair; the third shows its transcriptomically convergent and structurally divergent selected pair. Comparator-containing pairs are excluded. Each example names its pair as "(first compound vs. second compound)" in parentheses and plain black, identifying which compound is displayed above and which is displayed below, and reports the unsigned Tanimoto similarity of the Morgan fingerprint and the Chem. and Chem.+Bio. signed between-compound probability differences (Δp), calculated as first-displayed minus second-displayed, each colored in that pair's alignment color only when it is the defining statistic for that selection category. Purple pairs place the compound with the higher Chem.+Bio. probability above the other compound. Orange pairs place the compound with the higher Chem. probability above the other compound. Shared scaffolds are aligned to matching 2D coordinates within each pair. Δp denotes the signed first-minus-second difference within the indicated model, not the probability increment from adding Bio. Green atoms belong to the pairwise maximum common substructure; magenta and blue atoms are specific to the first- and second-displayed compound, respectively. A two-row legend below the panels gives the scatterplot marker styles and example-ring colors (row 1) and the molecule substructure-highlight colors (row 2). Relationships are exploratory and do not establish causal mechanisms.
"""
    (OUT / "caption.md").write_text(caption, encoding="utf-8")
    manifest = {
        "figure": "Figure S2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "layout": "A4-compatible tight portrait; three endpoint-aligned rows; near-square panel-A scatterplots; height-matched panels B and C",
        "sr_are_result": "no selected-selected qualifying pair; comparator-containing cases excluded",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "rdkit": rdkit.__version__},
        "inputs": {str(path): sha256(path) for path in [UNIVERSE_PATH, EXEMPLAR_PATH]},
        "outputs": {},
    }
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"FigureS2_manifest.json", "provenance.json"} and "__pycache__" not in path.parts:
            manifest["outputs"][path.relative_to(OUT).as_posix()] = sha256(path)
    (GENERATION / "FigureS2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    provenance = {"files": {
        path.relative_to(OUT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name != "provenance.json" and "__pycache__" not in path.parts
    }}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")



if __name__ == "__main__":
    main()
