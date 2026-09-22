# %%
"""
join_ext.py
===========
Inner-join the existing aggregated adata with the extended-cell-line run
on shared pert_id values.

All paths and the output filename are derived from lincs_ext.json.
The output filename embeds all cell_ids found in both datasets.
"""

import json
import os
import numpy as np
import pandas as pd
import scanpy as sc

# %%
# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--config", default="/home/kyungan/scripts/BioTox/bio/configs/lincs_ext.json")
_p.add_argument("--run-keys", nargs="*", default=None)
_args, _ = _p.parse_known_args()

config_path = _args.config

with open(config_path) as f:
    cfg = json.load(f)

_all_run_keys = _args.run_keys or list(cfg["runs"].keys())
agg_suffix = "_filter_select_align_aggregate"

for run_key in _all_run_keys:
    run_cfg  = cfg["runs"][run_key]
    base_out = run_cfg["path_output"]

    path_orig = os.path.join(
        os.path.dirname(base_out),
        f"lincs_merge_{run_key}{agg_suffix}.h5ad",
    )
    path_ext = base_out.replace(".h5ad", f"{agg_suffix}.h5ad")

    print(f"\n[{run_key}] Original : {path_orig}")
    print(f"[{run_key}] Extended : {path_ext}")

    if not os.path.exists(path_orig):
        print(f"[{run_key}] original aggregate not found — skipping")
        continue
    if not os.path.exists(path_ext):
        print(f"[{run_key}] extended aggregate not found — skipping")
        continue

    # ─────────────────────────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────────────────────────

    print(f"\nLoading original : {path_orig}")
    adata_orig = sc.read_h5ad(path_orig)
    print(f"  shape: {adata_orig.shape}  cells: {sorted(adata_orig.obs['cell_id'].unique())}")

    print(f"Loading extended : {path_ext}")
    adata_ext = sc.read_h5ad(path_ext)
    print(f"  shape: {adata_ext.shape}  cells: {sorted(adata_ext.obs['cell_id'].unique())}")

    # ─────────────────────────────────────────────────────────────
    # INNER JOIN on pert_id
    # ─────────────────────────────────────────────────────────────

    perts_orig = set(adata_orig.obs["pert_id"].unique())
    perts_ext  = set(adata_ext.obs["pert_id"].unique())
    shared     = perts_orig & perts_ext

    print(f"\npert_id counts:")
    print(f"  original : {len(perts_orig):,}")
    print(f"  extended : {len(perts_ext):,}")
    print(f"  shared   : {len(shared):,}")

    if len(shared) == 0:
        print(f"[{run_key}] no shared pert_ids — skipping")
        continue

    mask_orig = adata_orig.obs["pert_id"].isin(shared)
    mask_ext  = adata_ext.obs["pert_id"].isin(shared)

    adata_orig_sel = adata_orig[mask_orig].copy()
    adata_ext_sel  = adata_ext[mask_ext].copy()

    print(f"\nAfter filtering to shared pert_ids:")
    print(f"  original : {adata_orig_sel.shape[0]:,} rows")
    print(f"  extended : {adata_ext_sel.shape[0]:,} rows")

    # ─────────────────────────────────────────────────────────────
    # CONCATENATE
    # ─────────────────────────────────────────────────────────────

    common_cols = [c for c in adata_orig_sel.obs.columns if c in adata_ext_sel.obs.columns]
    adata_orig_sel.obs = adata_orig_sel.obs[common_cols]
    adata_ext_sel.obs  = adata_ext_sel.obs[common_cols]

    common_obsm = [k for k in adata_orig_sel.obsm.keys() if k in adata_ext_sel.obsm.keys()]

    obsm_orig = {k: adata_orig_sel.obsm[k] for k in common_obsm}
    obsm_ext  = {k: adata_ext_sel.obsm[k]  for k in common_obsm}

    adata_orig_sel.obs_names_make_unique()
    adata_ext_sel.obs_names_make_unique()

    adata_joined = sc.concat(
        [adata_orig_sel, adata_ext_sel],
        join="inner",
        merge="first",
    )

    for k in common_obsm:
        adata_joined.obsm[k] = np.vstack([obsm_orig[k], obsm_ext[k]])

    for uk in adata_orig.uns:
        adata_joined.uns[uk] = adata_orig.uns[uk]

    # ─────────────────────────────────────────────────────────────
    # OUTPUT PATH — embed all cell_ids dynamically
    # ─────────────────────────────────────────────────────────────

    all_cells = sorted(adata_joined.obs["cell_id"].unique())
    cell_tag  = "_".join(all_cells)
    path_out  = os.path.join(
        os.path.dirname(base_out),
        f"lincs_merge_{run_key}{agg_suffix}_{cell_tag}.h5ad",
    )

    # ─────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────

    print(f"\n{'='*55}")
    print(f"JOIN COMPLETE — {run_key}")
    print(f"{'='*55}")
    print(f"  Total rows    : {adata_joined.shape[0]:,}")
    print(f"  Genes         : {adata_joined.shape[1]:,}")
    print(f"  Cell lines    : {all_cells}")
    print(f"  Compounds     : {adata_joined.obs['pert_id'].nunique():,}")
    print(f"  obsm keys     : {list(adata_joined.obsm.keys())}")
    print(f"  obs columns   : {list(adata_joined.obs.columns)}")

    for cell in all_cells:
        n = (adata_joined.obs["cell_id"] == cell).sum()
        print(f"    {cell}: {n:,} rows")

    # ─────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────

    print(f"\nWriting -> {path_out}")
    adata_joined.write_h5ad(path_out)
    print("Done.")

# %%
