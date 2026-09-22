"""Build final 20x5 publication tables and figures from canonical outputs."""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "table_paths.json"

# Populated by configure_paths() before build() runs -- either from the CLI
# entry point below or by a caller (e.g. regenerate_tables.py) that imports
# this module and sets them directly, matching how regenerate_corrected_figures.py
# drives the other publication generators.
SOURCE_SUMMARY: Path
SOURCE_RUN: Path
OUTPUT_DIR: Path


def configure_paths(config_path: Path = DEFAULT_CONFIG, output_dir: Path | None = None) -> None:
    """Load SOURCE_SUMMARY/SOURCE_RUN from `table_paths.json`, and set OUTPUT_DIR
    to `output_dir` (or a timestamped directory under the config's output_root)."""
    global SOURCE_SUMMARY, SOURCE_RUN, OUTPUT_DIR
    config = json.loads(config_path.read_text(encoding="utf-8"))["generate_table"]
    SOURCE_SUMMARY = Path(config["source_summary"])
    SOURCE_RUN = Path(config["source_run"])
    if output_dir is not None:
        OUTPUT_DIR = output_dir
    else:
        root = Path(json.loads(config_path.read_text(encoding="utf-8"))["output_root"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        OUTPUT_DIR = root / f"generate_table_{stamp}"

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
CELL_LABELS = {"HA1E": "HA1E", "HEPG2": "HepG2", "HT29": "HT29", "MCF7": "MCF7"}
VARIANT_ORDER = ("standardized", "residualized")
EXCLUDED_PAIRS = {("NR-PPAR-gamma", "HEPG2")}
FAMILY_COLORS = {"NR": "#326A9A", "SR": "#C44E52"}
SIGNIFICANT_COLOR = "#D28E00"
TEXT_COLOR = "#202124"


def _latex_task(task: str) -> str:
    if task == "NR-PPAR-gamma":
        return r"NR-PPAR-$\gamma$"
    return task


def _probability(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return r"$<0.001$"
    decimal_places = max(0, 2 - math.floor(math.log10(abs(value))))
    return f"{value:.{decimal_places}f}"


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _ci(low: float, high: float) -> str:
    if not np.isfinite(low) or not np.isfinite(high):
        return "NA"
    return f"[{low:+.3f}, {high:+.3f}]"


def _bh_adjust(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(numeric), np.nan, dtype=float)
    valid = np.isfinite(numeric)
    if not valid.any():
        return pd.Series(adjusted, index=values.index, dtype=float)
    positions = np.flatnonzero(valid)
    ordered = positions[np.argsort(numeric[valid])]
    ranked = numeric[ordered] * len(ordered) / np.arange(1, len(ordered) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[ordered] = np.clip(ranked, 0.0, 1.0)
    return pd.Series(adjusted, index=values.index, dtype=float)


def _nb_test(
    deltas: np.ndarray,
    n_train: np.ndarray,
    n_test: np.ndarray,
    variance_tolerance: float = 1e-15,
) -> dict[str, float | bool]:
    deltas = np.asarray(deltas, dtype=float)
    n_train = np.asarray(n_train, dtype=float)
    n_test = np.asarray(n_test, dtype=float)
    variance = float(np.var(deltas, ddof=1))
    mean = float(np.mean(deltas))
    if variance <= variance_tolerance:
        return {
            "mean": mean,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "variance_degenerate": True,
        }
    correction = 1.0 / len(deltas) + float(np.mean(n_test / n_train))
    corrected_se = math.sqrt(variance * correction)
    degrees_freedom = len(deltas) - 1
    t_statistic = mean / corrected_se
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    return {
        "mean": mean,
        "ci_low": mean - critical * corrected_se,
        "ci_high": mean + critical * corrected_se,
        "p_value": float(
            2.0 * stats.t.sf(abs(t_statistic), degrees_freedom)
        ),
        "variance_degenerate": False,
    }


def _validate_discrimination(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "task",
        "cell",
        "variant",
        "n_compounds",
        "n_repeats",
        "folds_per_repeat",
    }
    for metric in ("auprc", "auroc"):
        required.update(
            {
                f"mean_{metric}_chem",
                f"sd_repeat_mean_{metric}_chem",
                f"mean_{metric}_biotox",
                f"sd_repeat_mean_{metric}_biotox",
                f"mean_delta_{metric}",
                f"nb_95ci_low_delta_{metric}",
                f"nb_95ci_high_delta_{metric}",
                f"nb_p_value_delta_{metric}",
                f"bh_q_value_delta_{metric}",
            }
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Discrimination summary is missing columns: {missing}")
    if len(frame) != 96:
        raise ValueError(f"Expected 96 discrimination rows, found {len(frame)}")
    if set(frame["variant"]) != set(VARIANT_ORDER):
        raise ValueError("Unexpected transcriptomic variants")
    if set(frame["task"]) != set(TASK_ORDER):
        raise ValueError("Unexpected endpoint set")
    if set(frame["cell"]) != set(CELL_ORDER):
        raise ValueError("Unexpected cell-line set")
    if set(frame["n_repeats"]) != {20} or set(frame["folds_per_repeat"]) != {5}:
        raise ValueError("Expected a 20x5 repeated-CV design")
    frame = frame.copy()
    frame["excluded"] = [
        (task, cell) in EXCLUDED_PAIRS
        for task, cell in zip(frame["task"], frame["cell"])
    ]
    return frame


def _losses(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
    )
    log_loss = -float(
        np.mean(y * np.log(probabilities) + (1.0 - y) * np.log1p(-probabilities))
    )
    brier = float(np.mean((probabilities - y) ** 2))
    return log_loss, brier


def _calibration_for_group(
    variant: str,
    task: str,
    cell: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    oof_path = (
        SOURCE_RUN
        / f"repeated_nested_cv_oof_predictions_{variant}_{task}__{cell}.csv"
    )
    fold_path = SOURCE_RUN / f"repeated_nested_cv_{variant}_{task}__{cell}.csv"
    oof = pd.read_csv(
        oof_path,
        usecols=[
            "ik",
            "y",
            "p_chem_only",
            "p_chem_bio",
            "repeat_id",
            "outer_fold_id",
        ],
    )
    folds = pd.read_csv(
        fold_path,
        usecols=["repeat_id", "outer_fold_id", "n_outer_train", "n_outer_test"],
    )
    if len(folds) != 100 or folds.duplicated(
        ["repeat_id", "outer_fold_id"]
    ).any():
        raise ValueError(f"Invalid fold matrix: {fold_path}")
    if oof.duplicated(["repeat_id", "ik"]).any():
        raise ValueError(f"Duplicate repeat-compound predictions: {oof_path}")
    if oof["repeat_id"].nunique() != 20:
        raise ValueError(f"Expected 20 repeats: {oof_path}")

    fold_rows: list[dict[str, object]] = []
    for (repeat_id, fold_id), group in oof.groupby(
        ["repeat_id", "outer_fold_id"],
        sort=True,
    ):
        chem_log_loss, chem_brier = _losses(
            group["y"].to_numpy(), group["p_chem_only"].to_numpy()
        )
        bio_log_loss, bio_brier = _losses(
            group["y"].to_numpy(), group["p_chem_bio"].to_numpy()
        )
        fold_rows.append(
            {
                "variant": variant,
                "task": task,
                "cell": cell,
                "repeat_id": int(repeat_id),
                "outer_fold_id": int(fold_id),
                "log_loss_chem": chem_log_loss,
                "log_loss_biotox": bio_log_loss,
                "delta_log_loss": chem_log_loss - bio_log_loss,
                "brier_chem": chem_brier,
                "brier_biotox": bio_brier,
                "delta_brier": chem_brier - bio_brier,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows).merge(
        folds,
        on=["repeat_id", "outer_fold_id"],
        how="left",
        validate="one_to_one",
    )
    if fold_metrics[["n_outer_train", "n_outer_test"]].isna().any().any():
        raise ValueError(f"Missing fold sizes for {variant}/{task}/{cell}")

    repeat_rows: list[dict[str, float]] = []
    for _, group in oof.groupby("repeat_id", sort=True):
        chem_log_loss, chem_brier = _losses(
            group["y"].to_numpy(), group["p_chem_only"].to_numpy()
        )
        bio_log_loss, bio_brier = _losses(
            group["y"].to_numpy(), group["p_chem_bio"].to_numpy()
        )
        repeat_rows.append(
            {
                "log_loss_chem": chem_log_loss,
                "log_loss_biotox": bio_log_loss,
                "brier_chem": chem_brier,
                "brier_biotox": bio_brier,
            }
        )
    repeat_metrics = pd.DataFrame(repeat_rows)
    first_repeat = oof.loc[oof["repeat_id"] == oof["repeat_id"].min()]
    row: dict[str, object] = {
        "variant": variant,
        "task": task,
        "family": task.split("-", maxsplit=1)[0],
        "cell": cell,
        "n_compounds": int(first_repeat["ik"].nunique()),
        "n_active": int(first_repeat["y"].sum()),
        "n_repeats": 20,
        "folds_per_repeat": 5,
        "excluded": (task, cell) in EXCLUDED_PAIRS,
    }
    for metric in ("log_loss", "brier"):
        nb = _nb_test(
            fold_metrics[f"delta_{metric}"].to_numpy(),
            fold_metrics["n_outer_train"].to_numpy(),
            fold_metrics["n_outer_test"].to_numpy(),
        )
        row.update(
            {
                f"mean_{metric}_chem": float(
                    repeat_metrics[f"{metric}_chem"].mean()
                ),
                f"sd_repeat_mean_{metric}_chem": float(
                    repeat_metrics[f"{metric}_chem"].std(ddof=1)
                ),
                f"mean_{metric}_biotox": float(
                    repeat_metrics[f"{metric}_biotox"].mean()
                ),
                f"sd_repeat_mean_{metric}_biotox": float(
                    repeat_metrics[f"{metric}_biotox"].std(ddof=1)
                ),
                f"mean_delta_{metric}": nb["mean"],
                f"nb_95ci_low_delta_{metric}": nb["ci_low"],
                f"nb_95ci_high_delta_{metric}": nb["ci_high"],
                f"nb_p_value_delta_{metric}": nb["p_value"],
                f"variance_degenerate_delta_{metric}": nb[
                    "variance_degenerate"
                ],
            }
        )
    return row, fold_metrics


def _build_calibration() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    fold_frames: list[pd.DataFrame] = []
    for variant in VARIANT_ORDER:
        for task in TASK_ORDER:
            for cell in CELL_ORDER:
                row, folds = _calibration_for_group(variant, task, cell)
                rows.append(row)
                fold_frames.append(folds)
    summary = pd.DataFrame(rows)
    for metric in ("log_loss", "brier"):
        summary[f"bh_q_value_delta_{metric}"] = np.nan
        for (_, cell), indices in summary.groupby(
            ["variant", "cell"], sort=False
        ).groups.items():
            summary.loc[
                indices, f"bh_q_value_delta_{metric}"
            ] = _bh_adjust(summary.loc[indices, f"nb_p_value_delta_{metric}"])
    return summary, pd.concat(fold_frames, ignore_index=True)


def _table_cell(
    row: pd.Series,
    metric: str,
    *,
    loss_metric: bool = False,
) -> str:
    if bool(row["excluded"]):
        return r"---\textsuperscript{a}"
    chem = (
        f"{row[f'mean_{metric}_chem']:.3f}"
        rf" $\pm$ {row[f'sd_repeat_mean_{metric}_chem']:.3f}"
    )
    biotox = (
        f"{row[f'mean_{metric}_biotox']:.3f}"
        rf" $\pm$ {row[f'sd_repeat_mean_{metric}_biotox']:.3f}"
    )
    delta = _signed(float(row[f"mean_delta_{metric}"]))
    ci = _ci(
        float(row[f"nb_95ci_low_delta_{metric}"]),
        float(row[f"nb_95ci_high_delta_{metric}"]),
    )
    q_value = float(row[f"bh_q_value_delta_{metric}"])
    significant = (
        np.isfinite(q_value)
        and q_value < 0.05
        and float(row[f"mean_delta_{metric}"]) > 0
    )
    dagger = r"\textsuperscript{$\dagger$}" if significant else ""
    prefix = r"{\bfseries\boldmath " if significant else ""
    suffix = "}" if significant else ""
    delta_label = r"$\Delta_{\rm imp}$" if loss_metric else r"$\Delta$"
    return (
        prefix
        + r"\begin{tabular}[c]{@{}c@{}}"
        + f"$n={int(row['n_compounds'])}$"
        + r"\\ "
        + f"Chem {chem}; BioTox {biotox}"
        + r"\\ "
        + f"{delta_label} {delta} {ci}; $q={_probability(q_value)}$"
        + r"\end{tabular}"
        + suffix
        + dagger
    )


def _latex_table(
    frame: pd.DataFrame,
    *,
    variant: str,
    metric: str,
    caption: str,
    label: str,
    loss_metric: bool = False,
) -> str:
    subset = frame.loc[frame["variant"] == variant].copy()
    indexed = subset.set_index(["task", "cell"])
    lines = [
        r"\begin{landscape}",
        r"\begin{table*}[p]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\renewcommand{\arraystretch}{1.16}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Tox21 assay & HA1E & HepG2 & HT29 & MCF7 \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Nuclear-receptor endpoints}} \\",
    ]
    for index, task in enumerate(TASK_ORDER):
        if index == 7:
            lines.extend(
                [
                    r"\addlinespace[2pt]",
                    r"\multicolumn{5}{l}{\textit{Stress-response endpoints}} \\",
                ]
            )
        cells = [
            _table_cell(indexed.loc[(task, cell)], metric, loss_metric=loss_metric)
            for cell in CELL_ORDER
        ]
        lines.append(f"{_latex_task(task)} & " + " & ".join(cells) + r" \\")
    delta_definition = (
        r"For log loss and Brier score, "
        r"$\Delta_{\rm imp}=\mathrm{loss}_{Chem}-\mathrm{loss}_{BioTox}$, "
        r"so positive values indicate improved calibration."
        if loss_metric
        else (
            rf"$\Delta {metric.upper()}={metric.upper()}_{{BioTox}}"
            rf"-{metric.upper()}_{{Chem}}$."
        )
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\vspace{0.4em}",
            r"\begin{minipage}{0.99\linewidth}",
            r"\footnotesize",
            (
                "Values are means $\\pm$ SD across 20 repeat-level estimates. "
                + delta_definition
                + " Brackets give Nadeau--Bengio corrected 95\\% confidence "
                "intervals from all 100 paired outer-fold effects. BH "
                "$q$-values were calculated within each cell line across 12 "
                "endpoints for this representation. "
                r"\textsuperscript{$\dagger$}Positive effect with $q<0.05$. "
                r"\textsuperscript{a}HepG2--NR-PPAR-$\gamma$ was excluded "
                "because only 19 active compounds were available. "
                "NA denotes a variance-degenerate paired effect for which "
                "corrected inference is undefined."
            ),
            r"\end{minipage}",
            r"\end{table*}",
            r"\end{landscape}",
        ]
    )
    return "\n".join(lines)


def _write_latex_markdown(path: Path, title: str, latex: str) -> None:
    content = (
        f"# {title}\n\n"
        "Required packages: `booktabs`, `graphicx`, and `pdflscape`.\n\n"
        "```latex\n"
        f"{latex}\n"
        "```\n"
    )
    path.write_text(content, encoding="utf-8")


def _finite_plot_extent(
    frame: pd.DataFrame,
    metric: str,
    variants: tuple[str, ...],
) -> float:
    subset = frame.loc[
        frame["variant"].isin(variants) & ~frame["excluded"].astype(bool)
    ]
    values = np.concatenate(
        [
            subset[f"nb_95ci_low_delta_{metric}"].to_numpy(dtype=float),
            subset[f"nb_95ci_high_delta_{metric}"].to_numpy(dtype=float),
            subset[f"mean_delta_{metric}"].to_numpy(dtype=float),
        ]
    )
    finite = np.abs(values[np.isfinite(values)])
    return max(0.01, float(finite.max()) * 1.12)


def _forest_plot(
    frame: pd.DataFrame,
    *,
    variants: tuple[str, ...],
    metric: str,
    stem: Path,
    title: str,
    x_label: str,
    loss_metric: bool = False,
) -> None:
    n_rows = len(variants)
    figure, axes = plt.subplots(
        n_rows,
        len(CELL_ORDER),
        figsize=(15.8, 5.8 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = _finite_plot_extent(frame, metric, variants)
    y_positions = np.arange(len(TASK_ORDER))
    for row_index, variant in enumerate(variants):
        for column, cell in enumerate(CELL_ORDER):
            ax = axes[row_index, column]
            subset = (
                frame.loc[
                    (frame["variant"] == variant) & (frame["cell"] == cell)
                ]
                .set_index("task")
                .loc[list(TASK_ORDER)]
            )
            for position, (task, result) in enumerate(subset.iterrows()):
                if bool(result["excluded"]):
                    ax.text(
                        0.0,
                        position,
                        "excluded",
                        ha="center",
                        va="center",
                        fontsize=7.2,
                        color="#777777",
                        style="italic",
                    )
                    continue
                mean = float(result[f"mean_delta_{metric}"])
                low = float(result[f"nb_95ci_low_delta_{metric}"])
                high = float(result[f"nb_95ci_high_delta_{metric}"])
                q_value = float(result[f"bh_q_value_delta_{metric}"])
                significant = (
                    np.isfinite(q_value) and q_value < 0.05 and mean > 0
                )
                color = FAMILY_COLORS[task.split("-", maxsplit=1)[0]]
                if np.isfinite(low) and np.isfinite(high):
                    ax.errorbar(
                        mean,
                        position,
                        xerr=[[mean - low], [high - mean]],
                        fmt="D" if significant else "o",
                        markersize=6.4 if significant else 5.3,
                        color=SIGNIFICANT_COLOR if significant else color,
                        markeredgecolor="#111111",
                        markeredgewidth=0.8,
                        ecolor="#333333",
                        elinewidth=1.15,
                        capsize=2.5,
                        zorder=3,
                    )
                else:
                    ax.plot(
                        mean,
                        position,
                        marker="x",
                        markersize=6,
                        markeredgewidth=1.2,
                        color="#777777",
                        linestyle="none",
                        zorder=3,
                    )
                if significant:
                    ax.annotate(
                        f"q={q_value:.3g}",
                        (mean, position),
                        xytext=(6, -9),
                        textcoords="offset points",
                        fontsize=7.2,
                        fontweight="bold",
                        color="#6B4700",
                    )
            ax.axvline(0.0, color="#777777", linewidth=0.9, linestyle=(0, (3, 3)))
            ax.axhline(6.5, color="#C6C6C6", linewidth=0.8)
            ax.set_xlim(-extent, extent)
            ax.set_ylim(len(TASK_ORDER) - 0.45, -0.55)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(TASK_ORDER, fontsize=8.3)
            ax.grid(axis="x", color="#E4E4E4", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            if n_rows == 1:
                ax.set_title(
                    CELL_LABELS[cell],
                    fontsize=11,
                    fontweight="bold",
                    pad=8,
                )
            else:
                variant_label = (
                    "Standardized (primary)"
                    if variant == "standardized"
                    else "Residualized (sensitivity)"
                )
                ax.set_title(
                    f"{CELL_LABELS[cell]}\n{variant_label}",
                    fontsize=9.2,
                    fontweight="bold",
                    color=TEXT_COLOR,
                    pad=8,
                )
            if column == 0:
                ax.set_ylabel("Tox21 assay", fontsize=9.2, fontweight="bold")
            ax.text(
                0.015,
                0.985,
                f"({chr(97 + row_index * 4 + column)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
            )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FAMILY_COLORS["NR"],
            markeredgecolor="#111111",
            label="Nuclear receptor",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FAMILY_COLORS["SR"],
            markeredgecolor="#111111",
            label="Stress response",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=SIGNIFICANT_COLOR,
            markeredgecolor="#111111",
            label=r"Positive BH $q<0.05$",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#777777",
            linestyle="none",
            label="Variance-degenerate; CI undefined",
        ),
    ]
    figure.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.985,
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.004),
        fontsize=8.3,
    )
    delta_note = (
        "Positive values indicate lower loss with BioTox."
        if loss_metric
        else "Positive values favor BioTox."
    )
    figure.text(
        0.5,
        0.058 if n_rows == 1 else 0.040,
        (
            "Points are mean paired outer-fold effects; whiskers are "
            "Nadeau–Bengio corrected 95% confidence intervals. "
            + delta_note
        ),
        ha="center",
        fontsize=8.4,
        color="#555555",
    )
    figure.supxlabel(
        x_label,
        fontsize=10,
        fontweight="bold",
        y=0.087 if n_rows == 1 else 0.062,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.92 if n_rows == 1 else 0.90,
        bottom=0.145 if n_rows == 1 else 0.105,
        hspace=0.30,
        wspace=0.10,
    )
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(
            stem.with_suffix(suffix),
            dpi=450,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def _write_readme(output_dir: Path) -> None:
    text = """# Publication-primary 20x5 tables and figures

This directory was generated from the completed calibrated repeated nested
scaffold-CV analysis. The standardized AUPRC analysis is primary;
residualized AUPRC and all AUROC analyses are supplementary. Calibration
supplements use out-of-fold probabilities from the identical 20x5 splits.

## Statistical conventions

- Absolute metrics are means and SDs across 20 repeat-level estimates.
- Paired effects and Nadeau-Bengio corrected 95% confidence intervals use all
  100 outer-fold effects.
- BH correction is performed within representation and cell line across the
  12 endpoints.
- For AUPRC/AUROC, positive delta means BioTox minus Chem is positive.
- For log loss/Brier score, positive improvement means Chem minus BioTox loss
  is positive.
- HepG2--NR-PPAR-gamma is displayed as excluded because it has 19 actives.
- Variance-degenerate effects retain their point estimate, but CI and p-value
  are reported as undefined.

LaTeX tables are stored as fenced LaTeX blocks in `.md` files. Figures are
provided as PNG, vector PDF, and editable SVG.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def build() -> dict[str, object]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite output: {OUTPUT_DIR}")
    if not SOURCE_SUMMARY.is_file():
        raise FileNotFoundError(SOURCE_SUMMARY)
    if not (SOURCE_RUN / "RUN_COMPLETE.json").is_file():
        raise FileNotFoundError(SOURCE_RUN / "RUN_COMPLETE.json")

    tables_dir = OUTPUT_DIR / "tables"
    figures_dir = OUTPUT_DIR / "figures"
    data_dir = OUTPUT_DIR / "data"
    for directory in (tables_dir, figures_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=False)

    discrimination = _validate_discrimination(pd.read_csv(SOURCE_SUMMARY))
    calibration, calibration_folds = _build_calibration()
    discrimination.to_csv(data_dir / "discrimination_summary_20x5.csv", index=False)
    calibration.to_csv(data_dir / "calibration_summary_20x5.csv", index=False)
    calibration_folds.to_csv(
        data_dir / "calibration_outer_fold_metrics_20x5.csv",
        index=False,
    )

    _write_latex_markdown(
        tables_dir / "main_table_standardized_auprc.md",
        "Main table: standardized AUPRC",
        _latex_table(
            discrimination,
            variant="standardized",
            metric="auprc",
            caption=(
                "Primary standardized Stage-2 discrimination performance "
                "for each Tox21 assay and cellular context."
            ),
            label="tab:standardized-auprc-primary",
        ),
    )
    _write_latex_markdown(
        tables_dir / "supp_table_residualized_auprc.md",
        "Supplementary table: residualized AUPRC",
        _latex_table(
            discrimination,
            variant="residualized",
            metric="auprc",
            caption=(
                "Sensitivity analysis of CeViChe-residualized Stage-2 AUPRC "
                "for each Tox21 assay and cellular context."
            ),
            label="tab:residualized-auprc-sensitivity",
        ),
    )
    for variant in VARIANT_ORDER:
        _write_latex_markdown(
            tables_dir / f"supp_table_{variant}_auroc.md",
            f"Supplementary table: {variant} AUROC",
            _latex_table(
                discrimination,
                variant=variant,
                metric="auroc",
                caption=(
                    f"Secondary {variant} Stage-2 AUROC for each Tox21 "
                    "assay and cellular context."
                ),
                label=f"tab:{variant}-auroc-secondary",
            ),
        )
        _write_latex_markdown(
            tables_dir / f"supp_table_calibration_{variant}.md",
            f"Calibration supplement: {variant}",
            _latex_table(
                calibration,
                variant=variant,
                metric="log_loss",
                caption=(
                    f"Calibration supplement for the {variant} representation: "
                    "log loss."
                ),
                label=f"tab:{variant}-log-loss-calibration",
                loss_metric=True,
            )
            + "\n\n"
            + _latex_table(
                calibration,
                variant=variant,
                metric="brier",
                caption=(
                    f"Calibration supplement for the {variant} representation: "
                    "Brier score."
                ),
                label=f"tab:{variant}-brier-calibration",
                loss_metric=True,
            ),
        )

    _forest_plot(
        discrimination,
        variants=("standardized",),
        metric="auprc",
        stem=figures_dir / "main_figure_standardized_delta_auprc",
        title="Incremental value of standardized transcriptomic response",
        x_label=r"$\Delta$AUPRC (BioTox − Chem)",
    )
    _forest_plot(
        discrimination,
        variants=VARIANT_ORDER,
        metric="auroc",
        stem=figures_dir / "supp_figure_standardized_residualized_delta_auroc",
        title="Secondary AUROC analysis across transcriptomic representations",
        x_label=r"$\Delta$AUROC (BioTox − Chem)",
    )
    for metric, label in (("log_loss", "log loss"), ("brier", "Brier score")):
        _forest_plot(
            calibration,
            variants=VARIANT_ORDER,
            metric=metric,
            stem=figures_dir / f"supp_figure_calibration_delta_{metric}",
            title=f"Calibration improvement: {label}",
            x_label=rf"$\Delta_{{imp}}$ {label} (Chem − BioTox)",
            loss_metric=True,
        )

    _write_readme(OUTPUT_DIR)
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    artifacts = sorted(
        str(path.relative_to(OUTPUT_DIR))
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file()
    )
    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary": str(SOURCE_SUMMARY),
        "source_run": str(SOURCE_RUN),
        "output_dir": str(OUTPUT_DIR),
        "design": {
            "n_repeats": 20,
            "outer_folds_per_repeat": 5,
            "paired_outer_effects_per_group": 100,
        },
        "primary_analysis": {
            "representation": "standardized",
            "metric": "AUPRC",
        },
        "inference": "Nadeau-Bengio corrected resampled paired t-test",
        "multiple_testing": (
            "Benjamini-Hochberg within representation and cell line "
            "across 12 endpoints"
        ),
        "excluded_pairs": [
            {
                "task": "NR-PPAR-gamma",
                "cell": "HEPG2",
                "reason": "19 active compounds; prespecified minimum is 20",
            }
        ],
        "n_discrimination_rows": int(len(discrimination)),
        "n_calibration_rows": int(len(calibration)),
        "n_calibration_fold_rows": int(len(calibration_folds)),
        "artifacts": artifacts,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "RUN_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "created_at_utc": manifest["created_at_utc"],
                "n_artifacts": len(artifacts) + 2,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    configure_paths(args.config, args.output_dir)
    print(json.dumps(build(), indent=2))
