from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

sys.path.insert(0, "/home/kyungan/scripts/BioTox/publication/standalone/bio/run")
import run_stage2_calibration_ridge_ablation as _runner  # noqa: F401
from publication.repeated_cv_inference import repeated_cv_nadeau_bengio_test

RUN = Path("/data/kyungan/scripts/BioTox/publication/standalone/bio/run/methodology_audit/results/primary_6h_20x5")
OUT = Path("/data/kyungan/scripts/BioTox/publication/analyses/tableS8_standardized_reproduction_20260922")
OLD = pd.read_csv("/home/kyungan/scripts/BioTox/publication/tables/TableS8/TableS8_source_data.csv")
CELLS = ["HA1E", "HEPG2", "HT29", "MCF7"]

fold_rows = []
for path in sorted(RUN.glob("repeated_nested_cv_oof_predictions_standardized_*.csv")):
    frame = pd.read_csv(path)
    task, cell = str(frame.task.iloc[0]), str(frame.cell.iloc[0])
    n_total = frame.ik.nunique()
    for (repeat_id, fold_id), group in frame.groupby(["repeat_id", "outer_fold_id"], sort=True):
        y = group.y.to_numpy(float)
        chemical = group.p_chem_stage1_raw.to_numpy(float)
        calibrated = group.p_chem_only.to_numpy(float)
        fold_rows.append({
            "task": task, "cell": cell, "repeat_id": int(repeat_id),
            "outer_fold_id": int(fold_id), "n_outer_test": len(group),
            "n_outer_train": int(n_total - len(group)),
            "log_loss_c": log_loss(y, chemical, labels=[0, 1]),
            "log_loss_d": log_loss(y, calibrated, labels=[0, 1]),
            "auprc_c": average_precision_score(y, chemical),
            "auprc_d": average_precision_score(y, calibrated),
        })

fold = pd.DataFrame(fold_rows)
fold["delta_log_loss_d_minus_c"] = fold.log_loss_d - fold.log_loss_c
fold["delta_auprc_d_minus_c"] = fold.auprc_d - fold.auprc_c
repeat = fold.groupby(["task", "cell", "repeat_id"], as_index=False)[
    ["delta_log_loss_d_minus_c", "delta_auprc_d_minus_c"]
].mean()

def bh(values):
    values = values.to_numpy(float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    positions = np.where(valid)[0]
    order = positions[np.argsort(values[valid])]
    q = values[order] * len(order) / np.arange(1, len(order) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    result[order] = np.clip(q, 0, 1)
    return result

chem_rows = []
for (task, cell), group in fold.groupby(["task", "cell"], sort=False):
    repeats = repeat[(repeat.task == task) & (repeat.cell == cell)]
    logloss_nb = repeated_cv_nadeau_bengio_test(
        group.delta_log_loss_d_minus_c.to_numpy(),
        group.n_outer_train.to_numpy(), group.n_outer_test.to_numpy())
    auprc_nb = repeated_cv_nadeau_bengio_test(
        group.delta_auprc_d_minus_c.to_numpy(),
        group.n_outer_train.to_numpy(), group.n_outer_test.to_numpy())
    chem_rows.append({
        "family": "NR" if task.startswith("NR-") else "SR",
        "task": task, "cell": cell,
        "panel_a_mean": repeats.delta_log_loss_d_minus_c.mean(),
        "panel_a_repeat_sd": repeats.delta_log_loss_d_minus_c.std(ddof=1),
        "panel_a_auprc_d_minus_c_mean": repeats.delta_auprc_d_minus_c.mean(),
        "panel_a_auprc_d_minus_c_repeat_sd": repeats.delta_auprc_d_minus_c.std(ddof=1),
        "panel_a_nb_mean": logloss_nb["mean"],
        "panel_a_nb_ci_low": logloss_nb["ci_low"],
        "panel_a_nb_ci_high": logloss_nb["ci_high"],
        "panel_a_nb_p_value": logloss_nb["p_value"],
        "panel_a_n_outer_folds": logloss_nb["n"],
        "panel_a_n_repeats": len(repeats),
        "panel_a_auprc_nb_mean": auprc_nb["mean"],
        "panel_a_auprc_nb_ci_low": auprc_nb["ci_low"],
        "panel_a_auprc_nb_ci_high": auprc_nb["ci_high"],
        "panel_a_auprc_nb_p_value": auprc_nb["p_value"],
    })
chem = pd.DataFrame(chem_rows)
for cell in CELLS:
    mask = chem.cell.eq(cell)
    chem.loc[mask, "panel_a_bh_q_value"] = bh(chem.loc[mask, "panel_a_nb_p_value"])
    chem.loc[mask, "panel_a_auprc_bh_q_value"] = bh(chem.loc[mask, "panel_a_auprc_nb_p_value"])

ridge_repeats = pd.read_csv(OUT / "calibration_ridge_repeat_means.csv")
ridge_summary = pd.read_csv(OUT / "calibration_ridge_summary_all.csv")
ridge_repeats = ridge_repeats[ridge_repeats.variant.eq("standardized")]
ridge_summary = ridge_summary[ridge_summary.variant.eq("standardized")]
panel_b = ridge_repeats.groupby(["task", "cell"], as_index=False).agg(
    panel_b_mean=("delta_auprc_d1_minus_c1", "mean"),
    panel_b_repeat_sd=("delta_auprc_d1_minus_c1", "std"),
)
panel_b = panel_b.merge(
    ridge_summary[["task", "cell", "n_unique_compounds",
                   "nb_auprc_d1_minus_c1_mean", "nb_auprc_d1_minus_c1_ci_low",
                   "nb_auprc_d1_minus_c1_ci_high", "nb_auprc_d1_minus_c1_p_value",
                   "bh_q_auprc_d1_minus_c1", "nb_auprc_d1_minus_c1_n", "n_repeats"]],
    on=["task", "cell"], validate="one_to_one",
).rename(columns={
    "nb_auprc_d1_minus_c1_mean": "panel_b_nb_mean",
    "nb_auprc_d1_minus_c1_ci_low": "panel_b_nb_ci_low",
    "nb_auprc_d1_minus_c1_ci_high": "panel_b_nb_ci_high",
    "nb_auprc_d1_minus_c1_p_value": "panel_b_nb_p_value",
    "bh_q_auprc_d1_minus_c1": "panel_b_bh_q_value",
    "nb_auprc_d1_minus_c1_n": "panel_b_n_outer_folds",
    "n_repeats": "panel_b_n_repeats",
})

new = chem.merge(panel_b, on=["task", "cell"], validate="one_to_one").sort_values(["task", "cell"])
new.to_csv(OUT / "TableS8_standardized_numeric.csv", index=False)

comparison = OLD.merge(new, on=["task", "cell"], suffixes=("_old", "_new"))
fields = [
    "panel_a_mean", "panel_a_repeat_sd", "panel_a_auprc_d_minus_c_mean",
    "panel_a_auprc_d_minus_c_repeat_sd", "panel_b_mean", "panel_b_repeat_sd",
    "panel_a_nb_mean", "panel_a_nb_ci_low", "panel_a_nb_ci_high",
    "panel_a_nb_p_value", "panel_b_nb_mean", "panel_b_nb_ci_low",
    "panel_b_nb_ci_high", "panel_b_nb_p_value",
]
rows = []
for field in fields:
    old_values = comparison[f"{field}_old"].to_numpy(float)
    new_values = comparison[f"{field}_new"].to_numpy(float)
    difference = new_values - old_values
    rows.append({
        "field": field,
        "max_abs_diff": np.nanmax(np.abs(difference)),
        "mean_abs_diff": np.nanmean(np.abs(difference)),
        "n_changed_gt_1e-9": int(np.sum(np.abs(difference) > 1e-9)),
    })
pd.DataFrame(rows).to_csv(OUT / "TableS8_standardized_vs_archived_comparison.csv", index=False)
print(f"fold_rows={len(fold)} groups={fold.groupby(['task','cell']).ngroups} repeat_rows={len(repeat)}")
print(pd.DataFrame(rows).to_string(index=False))
print("\nstandardized Table S8 values:")
print(new[["task", "cell", "panel_a_mean", "panel_a_auprc_d_minus_c_mean", "panel_b_mean", "panel_b_nb_p_value"]].to_string(index=False))
