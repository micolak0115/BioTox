"""Build the supplementary numerical table underlying Figure 3 panels A/B."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
INPUT_FILES = CONFIG["input_files"]
DEFAULT_CHEM_PLOT_DIR = Path(INPUT_FILES["chem_plot_dir"])
DEFAULT_CHEM_INFERENCE_DIR = Path(INPUT_FILES["chem_inference_dir"])
DEFAULT_RIDGE_DIR = Path(INPUT_FILES["ridge_dir"])
DEFAULT_OUTPUT_DIR = (HERE / CONFIG["output"]["directory"]).resolve()

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


def _bh_adjust(values: pd.Series) -> pd.Series:
    p_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not valid.any():
        return pd.Series(adjusted, index=values.index, dtype=float)
    valid_positions = np.flatnonzero(valid)
    order = valid_positions[np.argsort(p_values[valid_positions])]
    ordered = p_values[order]
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    q_values = ordered * len(ordered) / ranks
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]
    adjusted[order] = np.clip(q_values, 0.0, 1.0)
    return pd.Series(adjusted, index=values.index, dtype=float)


def _family(task: str) -> str:
    return "NR" if task.startswith("NR-") else "SR"


def _format_effect(mean: float, sd: float) -> str:
    return f"{mean:+.4f} ± {sd:.4f}"


def _format_ci(low: float, high: float) -> str:
    if not np.isfinite(low) or not np.isfinite(high):
        return "NA"
    return f"[{low:+.4f}, {high:+.4f}]"


def _format_probability(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    text = str(value)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _build_table(
    chem_plot: pd.DataFrame,
    chem_inference: pd.DataFrame,
    ridge_repeats: pd.DataFrame,
    ridge_summary: pd.DataFrame,
) -> pd.DataFrame:
    expected_keys = pd.MultiIndex.from_product(
        [TASK_ORDER, CELL_ORDER], names=["task", "cell"]
    )

    panel_a = chem_plot.rename(
        columns={
            "delta_log_loss_d_minus_c_mean": "panel_a_mean",
            "delta_log_loss_d_minus_c_std": "panel_a_repeat_sd",
        }
    )[["task", "cell", "panel_a_mean", "panel_a_repeat_sd"]]
    panel_a_inference = chem_inference.rename(
        columns={
            "nb_log_loss_ci_low": "panel_a_nb_ci_low",
            "nb_log_loss_ci_high": "panel_a_nb_ci_high",
            "nb_log_loss_p_value": "panel_a_nb_p_value",
            "nb_log_loss_mean": "panel_a_nb_mean",
            "nb_log_loss_n": "panel_a_n_outer_folds",
            "n_repeats": "panel_a_n_repeats",
        }
    )[
        [
            "task",
            "cell",
            "panel_a_nb_mean",
            "panel_a_nb_ci_low",
            "panel_a_nb_ci_high",
            "panel_a_nb_p_value",
            "panel_a_n_outer_folds",
            "panel_a_n_repeats",
        ]
    ]
    panel_a = panel_a.merge(
        panel_a_inference, on=["task", "cell"], validate="one_to_one"
    )
    panel_a["panel_a_bh_q_value"] = np.nan
    for cell in CELL_ORDER:
        mask = panel_a["cell"].eq(cell)
        panel_a.loc[mask, "panel_a_bh_q_value"] = _bh_adjust(
            panel_a.loc[mask, "panel_a_nb_p_value"]
        )

    selected_repeats = ridge_repeats.loc[
        ridge_repeats["variant"].eq("residualized")
    ]
    panel_b_plot = (
        selected_repeats.groupby(["task", "cell"], as_index=False)[
            "delta_auprc_d1_minus_c1"
        ]
        .agg(panel_b_mean="mean", panel_b_repeat_sd="std")
    )
    selected_summary = ridge_summary.loc[
        ridge_summary["variant"].eq("residualized")
    ].rename(
        columns={
            "nb_auprc_d1_minus_c1_mean": "panel_b_nb_mean",
            "nb_auprc_d1_minus_c1_ci_low": "panel_b_nb_ci_low",
            "nb_auprc_d1_minus_c1_ci_high": "panel_b_nb_ci_high",
            "nb_auprc_d1_minus_c1_p_value": "panel_b_nb_p_value",
            "bh_q_auprc_d1_minus_c1": "panel_b_bh_q_value",
            "nb_auprc_d1_minus_c1_n": "panel_b_n_outer_folds",
            "n_repeats": "panel_b_n_repeats",
        }
    )
    panel_b = panel_b_plot.merge(
        selected_summary[
            [
                "task",
                "cell",
                "n_unique_compounds",
                "panel_b_nb_mean",
                "panel_b_nb_ci_low",
                "panel_b_nb_ci_high",
                "panel_b_nb_p_value",
                "panel_b_bh_q_value",
                "panel_b_n_outer_folds",
                "panel_b_n_repeats",
            ]
        ],
        on=["task", "cell"],
        validate="one_to_one",
    )

    table = panel_a.merge(panel_b, on=["task", "cell"], validate="one_to_one")
    table.insert(0, "family", table["task"].map(_family))
    table["task"] = pd.Categorical(table["task"], TASK_ORDER, ordered=True)
    table["cell"] = pd.Categorical(table["cell"], CELL_ORDER, ordered=True)
    table = table.sort_values(["task", "cell"]).reset_index(drop=True)
    table["task"] = table["task"].astype(str)
    table["cell"] = table["cell"].astype(str)

    observed_keys = pd.MultiIndex.from_frame(table[["task", "cell"]])
    if not observed_keys.equals(expected_keys):
        raise ValueError("The final table does not contain the expected 48 ordered contexts")
    if len(table) != 48:
        raise ValueError(f"Expected 48 rows, found {len(table)}")
    if not np.allclose(
        table["panel_a_mean"], table["panel_a_nb_mean"], atol=1e-12
    ):
        raise ValueError("Panel A plotted means do not match inferential means")
    if not np.allclose(
        table["panel_b_mean"], table["panel_b_nb_mean"], atol=1e-12
    ):
        raise ValueError("Panel B plotted means do not match inferential means")
    if not table["panel_a_n_repeats"].eq(20).all():
        raise ValueError("Panel A does not contain 20 repeats for every context")
    if not table["panel_b_n_repeats"].eq(20).all():
        raise ValueError("Panel B does not contain 20 repeats for every context")
    if not table["panel_a_n_outer_folds"].eq(100).all():
        raise ValueError("Panel A does not contain 100 outer folds per context")
    if not table["panel_b_n_outer_folds"].eq(100).all():
        raise ValueError("Panel B does not contain 100 outer folds per context")
    return table


def _display_table(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Family": table["family"],
            "Tox21 endpoint": table["task"],
            "Cell line": table["cell"],
            "Compounds": table["n_unique_compounds"].astype(int),
            "Panel A Δlog loss ± SD": [
                _format_effect(mean, sd)
                for mean, sd in zip(
                    table["panel_a_mean"], table["panel_a_repeat_sd"]
                )
            ],
            "Panel A NB 95% CI": [
                _format_ci(low, high)
                for low, high in zip(
                    table["panel_a_nb_ci_low"], table["panel_a_nb_ci_high"]
                )
            ],
            "Panel A P": table["panel_a_nb_p_value"].map(_format_probability),
            "Panel A BH q": table["panel_a_bh_q_value"].map(_format_probability),
            "Panel B ΔAUPRC ± SD": [
                _format_effect(mean, sd)
                for mean, sd in zip(
                    table["panel_b_mean"], table["panel_b_repeat_sd"]
                )
            ],
            "Panel B NB 95% CI": [
                _format_ci(low, high)
                for low, high in zip(
                    table["panel_b_nb_ci_low"], table["panel_b_nb_ci_high"]
                )
            ],
            "Panel B P": table["panel_b_nb_p_value"].map(_format_probability),
            "Panel B BH q": table["panel_b_bh_q_value"].map(_format_probability),
        }
    )


def _write_markdown(display: pd.DataFrame, path: Path) -> None:
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(display: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{landscape}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{longtable}{lll r cc cc cc cc}",
        r"\caption{Numerical statistics underlying Figure 3.}\label{tab:s8_figure3_statistics}\\",
        r"\toprule",
        r"& & & & \multicolumn{4}{c}{Panel A: D0 $-$ C0} & \multicolumn{4}{c}{Panel B: D1 $-$ C1} \\",
        r"\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
        r"Family & Endpoint & Cell & $n$ & $\Delta$ log loss $\pm$ SD & NB 95\% CI & $P$ & BH $q$ & $\Delta$ AUPRC $\pm$ SD & NB 95\% CI & $P$ & BH $q$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"& & & & \multicolumn{4}{c}{Panel A: D0 $-$ C0} & \multicolumn{4}{c}{Panel B: D1 $-$ C1} \\",
        r"\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
        r"Family & Endpoint & Cell & $n$ & $\Delta$ log loss $\pm$ SD & NB 95\% CI & $P$ & BH $q$ & $\Delta$ AUPRC $\pm$ SD & NB 95\% CI & $P$ & BH $q$ \\",
        r"\midrule",
        r"\endhead",
    ]
    previous_family = None
    previous_task = None
    for row in display.itertuples(index=False, name=None):
        family, task, cell, compounds, *statistics = row
        if previous_family is not None and family != previous_family:
            lines.append(r"\midrule\midrule")
        elif previous_task is not None and task != previous_task:
            lines.append(r"\addlinespace[1.5pt]")
        values = [family, task, cell, str(compounds)] + list(statistics)
        lines.append(" & ".join(_escape_latex(value) for value in values) + r" \\")
        previous_family = family
        previous_task = task
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\end{landscape}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _caption_text() -> str:
    return """# Table S8 caption

**Table S8 | Numerical statistics underlying Figure 3.** Panel A reports the
paired change in outer-test log loss after adding an unpenalized
cohort-specific intercept to the frozen chemical logit (D0 - C0). Negative
values indicate improved probability-level prediction after calibration.
Panel B reports the paired change in AUPRC between transcriptomic ridge models
fitted with calibrated and uncalibrated frozen chemical offsets (D1 - C1).
Positive values favor the calibrated-offset specification. Values before the
confidence intervals are means ± sample SD across 20 repeat-level estimates;
each repeat-level estimate averages five outer folds. Confidence intervals and
two-sided nominal P values use all 100 paired outer-fold effects and the
Nadeau--Bengio corrected resampled test. BH q values apply
Benjamini--Hochberg correction separately within each panel and cell line
across the 12 Tox21 endpoints. The compound count is the number of distinct
endpoint-labeled compounds in the corresponding Stage-2 cell-line cohort.
The analysis uses viability-adjusted transcriptomic profiles.
"""


def _latex_recipe_text() -> str:
    return """# LaTeX recipe for Table S8

`table_s8_figure3_panel_statistics.tex` is a landscape `longtable` containing
all 48 endpoint-cell contexts. Add these packages to the manuscript preamble:

```latex
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{pdflscape}
```

Insert the table where supplementary Table S8 should appear:

```latex
\\input{table_s8_figure3_panel_statistics.tex}
```

The TeX source already includes the `landscape`, `scriptsize`, spacing,
caption, label, repeated header, NR/SR block separation, and table rules.
Use `table_s8_figure3_panel_statistics_display.csv` for journal production
systems that ingest tables separately from the manuscript. The numeric CSV
retains unrounded values and is the source of record.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    chem_plot_dir: Path,
    chem_inference_dir: Path,
    ridge_dir: Path,
    output_dir: Path,
) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    sources = {
        "panel_a_plot_summary": chem_plot_dir.resolve()
        / "plotting_summary_mean_sd.csv",
        "panel_a_inference": chem_inference_dir.resolve()
        / "chem_model_c_vs_d_summary.csv",
        "panel_b_repeat_means": ridge_dir.resolve()
        / "calibration_ridge_repeat_means.csv",
        "panel_b_inference": ridge_dir.resolve()
        / "calibration_ridge_summary_all.csv",
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    table = _build_table(
        pd.read_csv(sources["panel_a_plot_summary"]),
        pd.read_csv(sources["panel_a_inference"]),
        pd.read_csv(sources["panel_b_repeat_means"]),
        pd.read_csv(sources["panel_b_inference"]),
    )
    display = _display_table(table)
    output_dir.mkdir(parents=True, exist_ok=False)

    numeric_path = output_dir / "table_s8_figure3_panel_statistics_numeric.csv"
    display_path = output_dir / "table_s8_figure3_panel_statistics_display.csv"
    markdown_path = output_dir / "table_s8_figure3_panel_statistics.md"
    latex_path = output_dir / "table_s8_figure3_panel_statistics.tex"
    caption_path = output_dir / "TABLE_CAPTION.md"
    recipe_path = output_dir / "LATEX_RECIPE.md"
    table.to_csv(numeric_path, index=False)
    display.to_csv(display_path, index=False)
    _write_markdown(display, markdown_path)
    _write_latex(display, latex_path)
    caption_path.write_text(_caption_text(), encoding="utf-8")
    recipe_path.write_text(_latex_recipe_text(), encoding="utf-8")

    outputs = [
        numeric_path,
        display_path,
        markdown_path,
        latex_path,
        caption_path,
        recipe_path,
    ]
    manifest = {
        "table": "Table S8",
        "sources": {key: str(path) for key, path in sources.items()},
        "outputs": [str(path) for path in outputs],
        "output_sha256": {path.name: _sha256(path) for path in outputs},
        "n_rows": len(table),
        "n_endpoint_cell_contexts": 48,
        "visual_unit": "20 repeat means; each averages five outer folds",
        "inference_unit": "100 paired outer-fold effects",
        "multiple_testing": "BH within panel and cell across 12 endpoints",
        "analysis_complete": True,
    }
    (output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build supplementary statistics for Figure 3 panels A/B."
    )
    parser.add_argument(
        "--chem-plot-dir", type=Path, default=DEFAULT_CHEM_PLOT_DIR
    )
    parser.add_argument(
        "--chem-inference-dir", type=Path, default=DEFAULT_CHEM_INFERENCE_DIR
    )
    parser.add_argument("--ridge-dir", type=Path, default=DEFAULT_RIDGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.chem_plot_dir,
                args.chem_inference_dir,
                args.ridge_dir,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
