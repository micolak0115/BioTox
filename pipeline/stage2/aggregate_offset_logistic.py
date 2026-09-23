# publication/aggregate_offset_logistic.py
"""
Phase 8 -- reporting: builds ONE master summary table per variant, joining
Phase 5 (nested CV), Phase 6 (permutation test), Phase 7 (Nadeau-Bengio),
and the evaluability table (Phase 5b) per (task, cell).

DELIBERATELY DOES NOT DO what the backup pipeline's same-named script did:
the backup `aggregate_offset_logistic.py` computed a precision-weighted
CROSS-CELL aggregate (one number per task, combining all 4 cells). That is
exactly what this study's design prohibits (see METHODS.md Phase 1/8 --
cells have different N and compound composition now, so pooling them is not
meaningful). Every row in this module's output stays at (task, cell)
granularity; nothing is ever combined across cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2.nadeau_bengio_test import nadeau_bengio_corrected_test  # noqa: E402
from stage2.utils import CELL_IDS, TOX21_TASKS  # noqa: E402

RUN_DIR = Path(__file__).parent / "_run_output"
NESTED_CV_DIR = RUN_DIR / "nested_cv"
PERM_DIR = RUN_DIR / "permutation_test"
EVAL_TABLE_PATH = RUN_DIR / "evaluability_table.csv"


def total_metrics_for_task_cell(nested_cv_dir: Path, variant: str, task: str, cell: str) -> dict | None:
    """Mean +/- sd across outer folds of the paired total (not just delta)
    AUPRC/AUROC for chem-only and chem+bio -- the per-fold CSV has these
    per-fold, nested_cv_summary_all only has delta stats, so this reads the
    per-fold file directly (same one nadeau_bengio_for_task_cell uses)."""
    path = nested_cv_dir / f"nested_cv_{variant}_{task}__{cell}.csv"
    if not path.exists():
        return None
    fold_df = pd.read_csv(path)
    required = {"auprc_chem", "auprc_bio", "auroc_chem", "auroc_bio", "delta_auprc", "delta_auroc"}
    if not required.issubset(fold_df.columns) or len(fold_df) == 0:
        return None
    out = {}
    for col in required:
        out[f"mean_{col}"] = float(fold_df[col].mean())
        out[f"std_{col}"] = float(fold_df[col].std(ddof=1)) if len(fold_df) > 1 else 0.0
    return out


def require_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{source} is incompatible with the current publication schema; "
            f"missing columns: {', '.join(missing)}"
        )


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, preserving missing entries."""
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return pd.Series(adjusted, index=p_values.index, dtype=float)

    valid_positions = np.flatnonzero(valid)
    ordered_positions = valid_positions[np.argsort(values[valid])]
    ordered_p = values[ordered_positions]
    ranks = np.arange(1, len(ordered_p) + 1, dtype=float)
    ordered_q = ordered_p * len(ordered_p) / ranks
    ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
    adjusted[ordered_positions] = np.clip(ordered_q, 0.0, 1.0)
    return pd.Series(adjusted, index=p_values.index, dtype=float)


def nadeau_bengio_for_task_cell(variant: str, task: str, cell: str, nested_cv_dir: Path = NESTED_CV_DIR) -> dict | None:
    """
    Applies Nadeau-Bengio to the paired per-outer-fold series
    {auprc_bio_fold_i} vs {auprc_chem_fold_i} for one (task, cell) --
    WITHIN that cell only, never across cells. Null hypothesis: "does
    bio+chem beat the chemical baseline" -- a different null from Phase 6's
    permutation test ("is biology better than chance"); both are reported,
    neither substitutes for the other (see METHODS.md Phase 6/7).
    """
    path = nested_cv_dir / f"nested_cv_{variant}_{task}__{cell}.csv"
    if not path.exists():
        return None
    fold_df = pd.read_csv(path)
    require_columns(
        fold_df,
        {"delta_auprc", "n_outer_train", "n_outer_test"},
        path,
    )
    if len(fold_df) < 2:
        return None
    deltas = fold_df["delta_auprc"].to_numpy(dtype=float)
    n_train = fold_df["n_outer_train"].to_numpy(dtype=float)
    n_test = fold_df["n_outer_test"].to_numpy(dtype=float)
    try:
        nb = nadeau_bengio_corrected_test(deltas, n_train, n_test)
    except ValueError:
        return None
    return {
        "nb_mean_delta_auprc": nb["mean"], "nb_p_value": nb["p_value"],
        "nb_ci_low": nb["ci_low"], "nb_ci_high": nb["ci_high"],
        "nb_significant_raw": nb["significant_nb"],
    }


def build_master_table(variant: str, nested_cv_dir: Path = NESTED_CV_DIR, perm_dir: Path = PERM_DIR,
                        out_dir: Path = RUN_DIR, out_prefix: str = "master_summary") -> pd.DataFrame:
    eval_df = pd.read_csv(EVAL_TABLE_PATH) if EVAL_TABLE_PATH.exists() else None

    nested_summary_path = nested_cv_dir / f"nested_cv_summary_all_{variant}.csv"
    nested = pd.read_csv(nested_summary_path) if nested_summary_path.exists() else pd.DataFrame()

    perm_summary_path = perm_dir / f"permutation_test_summary_{variant}.csv"
    perm = pd.read_csv(perm_summary_path) if perm_summary_path.exists() else pd.DataFrame()
    if not perm_summary_path.exists():
        print(f"[aggregate] {perm_summary_path} not found -- perm_* columns will be empty "
              f"(Phase 6 not yet run for this offset)")

    rows = []
    for cell in CELL_IDS:
        for task in TOX21_TASKS:
            row = {"task": task, "cell": cell, "variant": variant}
            if eval_df is not None:
                ev = eval_df[(eval_df["cell"] == cell) & (eval_df["task"] == task)]
                if len(ev):
                    row.update({
                        "n_pos": int(ev.iloc[0]["n_pos"]), "n_valid": int(ev.iloc[0]["n_valid"]),
                        "evaluable": bool(ev.iloc[0]["evaluable"]),
                    })
            if len(nested):
                nc = nested[(nested["cell"] == cell) & (nested["task"] == task)]
                if len(nc):
                    r = nc.iloc[0]
                    row.update({
                        "n_outer_folds": r["n_outer_folds"],
                        "mean_delta_auprc": r["mean_delta_auprc"], "std_delta_auprc": r["std_delta_auprc"],
                    })
            if len(perm):
                pc = perm[(perm["cell"] == cell) & (perm["task"] == task)]
                if len(pc):
                    r = pc.iloc[0]
                    row.update({
                        "perm_observed_delta_auprc": r["observed_mean_delta_auprc"],
                        "perm_p_value": r["p_value"], "perm_n_valid": r["n_perm_valid"],
                    })
            nb = nadeau_bengio_for_task_cell(variant, task, cell, nested_cv_dir=nested_cv_dir)
            if nb:
                row.update(nb)
            totals = total_metrics_for_task_cell(nested_cv_dir, variant, task, cell)
            if totals:
                row.update(totals)
            rows.append(row)

    out = pd.DataFrame(rows)
    evaluable = (
        out["evaluable"].fillna(False).astype(bool)
        if "evaluable" in out.columns
        else pd.Series(True, index=out.index)
    )

    # FDR correction is applied WITHIN each cell separately, never pooled
    # across cells -- each cell tests a different, only partially-
    # overlapping compound set (its own cohort), so treating all 4 cells'
    # p-values as one shared family would implicitly combine independent
    # experiments for significance purposes, contradicting the same
    # per-cell-independence principle enforced everywhere else in this
    # study (METHODS.md Phase 1/8 -- no cross-cell pooling or ranking).
    for prefix, p_col, delta_col in (
        ("nb", "nb_p_value", "nb_mean_delta_auprc"),
        ("perm", "perm_p_value", "perm_observed_delta_auprc"),
    ):
        q_col = f"{prefix}_q_value"
        significant_col = f"{prefix}_significant_fdr"
        p_values = out[p_col] if p_col in out.columns else pd.Series(np.nan, index=out.index)
        q_values = pd.Series(np.nan, index=out.index, dtype=float)
        for cell, idx in out.groupby("cell").groups.items():
            cell_family_p = p_values.loc[idx].where(evaluable.loc[idx])
            q_values.loc[idx] = benjamini_hochberg(cell_family_p)
        out[q_col] = q_values
        positive = (
            pd.to_numeric(out[delta_col], errors="coerce") > 0
            if delta_col in out.columns
            else pd.Series(False, index=out.index)
        )
        out[significant_col] = (out[q_col] < 0.05) & positive

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_prefix}_{variant}.csv"
    out.to_csv(out_path, index=False)
    n_cells = out["cell"].nunique()
    print(f"[aggregate] wrote {out_path} ({len(out)} (task,cell) rows, FDR corrected "
          f"WITHIN each of {n_cells} cells separately -- no cross-cell pooling)")
    return out


def main(nested_cv_dir: Path = NESTED_CV_DIR, perm_dir: Path = PERM_DIR,
         out_dir: Path = RUN_DIR, out_prefix: str = "master_summary"):
    for variant in ("standardized", "residualized"):
        build_master_table(variant, nested_cv_dir, perm_dir, out_dir, out_prefix)


if __name__ == "__main__":
    main()
