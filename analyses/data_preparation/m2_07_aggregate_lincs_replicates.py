# %%
"""
aggregate.py
============
Aggregate LINCS replicate measurements per unique condition
(pert_id, cell_id, pert_dose, pert_time).

All obs columns from the aligned input are preserved in the
output — only n_reps is added.  No column is dropped so that
downstream labelling and filtering can use the full metadata.
"""

import json
import argparse as _ap
import numpy as np
import pandas as pd
import scanpy as sc

_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--config", default="/home/kyungan/scripts/BioTox/bio/configs/lincs_ext.json")
_p.add_argument("--run-keys", nargs="*", default=None)
_p.add_argument("--method", default="median", choices=["mean", "median", "tukey_biweight"])
_args, _ = _p.parse_known_args()

with open(_args.config) as f:
    _lincs = json.load(f)
_all_run_keys = _args.run_keys or list(_lincs["runs"].keys())
method = _args.method

# %%
def tukey_biweight(X: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Tukey biweight robust mean over replicates.

    Parameters
    ----------
    X : (n_replicates, n_features)
    c : tuning constant  (4.685 → 95% efficiency vs mean under normality)

    Returns
    -------
    (n_features,)  — down-weights outlier replicates
    """
    if X.shape[0] == 1:
        return X[0]

    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad = np.where(mad == 0, 1e-8, mad)

    u   = (X - med) / (c * mad)
    w   = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)

    w_sum = w.sum(axis=0)
    w_sum = np.where(w_sum == 0, 1.0, w_sum)

    return (w * X).sum(axis=0) / w_sum


def aggregate_replicates(X: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return X.mean(axis=0)
    if method == "median":
        return np.median(X, axis=0)
    if method == "tukey_biweight":
        return tukey_biweight(X)
    raise ValueError(
        f"Unknown method: {method!r}. "
        "Choose 'mean', 'median', or 'tukey_biweight'."
    )


for run_key in _all_run_keys:
    _base = _lincs["runs"][run_key]["path_output"]
    adata_path  = _base.replace(".h5ad", "_filter_select_align.h5ad")
    output_path = _base.replace(".h5ad", "_filter_select_align_aggregate.h5ad")

    # ─────────────────────────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────────────────────────

    print(f"Loading: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    obs   = adata.obs.copy()

    obsm_keys = list(adata.obsm.keys())
    print(f"  Shape   : {adata.shape}")
    print(f"  obsm    : {obsm_keys}")
    print(f"  obs cols: {list(obs.columns)}")

    n_before = len(adata)

    # ─────────────────────────────────────────────────────────────
    # AGGREGATE — one row per (pert_id, cell_id, pert_dose, pert_time)
    # ─────────────────────────────────────────────────────────────

    print(f"\nAggregating replicates with method='{method}' ...")
    print("  groupby: (pert_id, cell_id, pert_dose, pert_time)")

    group_cols = [
        "pert_id",
        "cell_id",
        "pert_dose",
        "pert_time"
    ]

    groups = obs.groupby(
        group_cols,
        sort=False,
        dropna=False,
    ).indices

    print("Before aggregation:")
    print(obs["pert_dose"].nunique())
    print(sorted(obs["pert_dose"].unique()))

    obsm_arrays = {k: adata.obsm[k] for k in obsm_keys}

    obsm_agg = {k: [] for k in obsm_keys}
    obs_agg = []

    for group_key, idx in groups.items():

        pert, cell, dose, time = group_key
        idx = np.asarray(idx)

        for k in obsm_keys:
            obsm_agg[k].append(
                aggregate_replicates(
                    obsm_arrays[k][idx],
                    method,
                )
            )

        group_obs = obs.iloc[idx]

        row = {}

        for col in obs.columns:

            values = group_obs[col].dropna().unique()

            if len(values) == 1:
                row[col] = values[0]

            elif col in group_cols:
                raise ValueError(
                    f"Grouping column {col} "
                    f"is not unique within group:\n"
                    f"{group_key}\n"
                    f"{values}"
                )

            else:
                row[col] = group_obs[col].iloc[0]

        row["pert_id"] = pert
        row["cell_id"] = cell
        row["pert_dose"] = dose
        row["pert_time"] = time

        row["n_reps"] = len(idx)

        obs_agg.append(row)

    obs_df = pd.DataFrame(obs_agg)

    adata_agg = sc.AnnData(
        X   = None,
        obs = obs_df,
        var = adata.var.copy(),
    )

    for k in obsm_keys:
        adata_agg.obsm[k] = np.vstack(obsm_agg[k])

    adata_agg.obs.index = [
        f"{p}_{c}_{d}_{t}h"
        for p, c, d, t in zip(
            adata_agg.obs["pert_id"],
            adata_agg.obs["cell_id"],
            adata_agg.obs["pert_dose"],
            adata_agg.obs["pert_time"],
        )
    ]

    print("After aggregation:")
    print(obs_df["pert_dose"].nunique())
    print(sorted(obs_df["pert_dose"].unique()))

    # ─────────────────────────────────────────────────────────────
    # CARRY CELL BASELINE
    # ─────────────────────────────────────────────────────────────

    for key in ("X_cell_baseline", "X_scGPT_cell_baseline"):
        if key in adata.uns:
            adata_agg.uns[key] = adata.uns[key]
            print(f"  Forwarded uns['{key}']")

    # ─────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────

    print(f"\n{'='*55}")
    print(f"AGGREGATION COMPLETE — {run_key}")
    print(f"{'='*55}")
    print(f"  Before : {n_before:,} rows")
    print(f"  After  : {len(adata_agg):,} rows  "
          f"(unique drug x cell x dose x time conditions)")
    print(f"  obs cols: {list(adata_agg.obs.columns)}")
    print(f"  Compounds : {adata_agg.obs['pert_id'].nunique():,}")
    print(f"  Cell lines: {adata_agg.obs['cell_id'].nunique():,}")
    print(f"  Doses     : {sorted(adata_agg.obs['pert_dose'].unique())}")
    print(f"  Times (h) : {sorted(adata_agg.obs['pert_time'].unique())}")
    print(f"\n  Reps per condition:")
    rep_dist = adata_agg.obs["n_reps"].value_counts().sort_index()
    for n, count in rep_dist.items():
        print(f"    {n} rep(s): {count:,} conditions")

    # ─────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────

    print(f"\nWriting → {output_path}")
    adata_agg.write_h5ad(output_path)
    print("Done.")


# %%
