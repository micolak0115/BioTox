# %%
import os
import gc
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from joblib import Parallel, delayed
from cmapPy.pandasGEXpress.parse_gctx import parse as parse_gctx

# %%
def _get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

log = _get_logger()


def load_gene_info(path: str) -> pd.DataFrame:
    log.info(f"Loading gene info: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()
    assert "pr_gene_id" in df.columns
    df = df.set_index("pr_gene_id")
    df.index = df.index.astype(str)
    df.index.name = "gene_id"
    for col in ("pr_is_lm", "pr_is_bing"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "pr_gene_symbol" in df.columns:
        df["gene_name"] = df["pr_gene_symbol"].astype(str)
    df["is_landmark"] = df["pr_is_lm"].astype(int) == 1 \
        if "pr_is_lm" in df.columns else False
    for col in df.columns:
        sample = df[col].iloc[0] if len(df) > 0 else None
        if isinstance(sample, (pd.Series, list, dict, np.ndarray)):
            df[col] = df[col].apply(
                lambda x: str(x.tolist()) if isinstance(x, (pd.Series, np.ndarray))
                else str(x)
            )
        else:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                df[col] = df[col].astype(str)
    log.info(f"Genes: {len(df):,} | landmark: {df['is_landmark'].sum():,}")

    return df


def load_inst_info(path: str, phase_label: str, geo_label: str) -> pd.DataFrame:
    log.info(f"Loading instance info [{geo_label}]: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()
    assert "inst_id" in df.columns
    df = df.set_index("inst_id")
    df.index = df.index.astype(str)
    df.index.name = "inst_id"
    df["phase"]    = phase_label
    df["geo"]      = geo_label
    for col in ("rna_plate", "rna_well"):
        if col not in df.columns:
            df[col] = "unknown"
    df["batch_id"] = [x.split(":")[0] for x in df.index]
    df["plate_id"] = ["_".join(x.split("_")[:3]) for x in df["batch_id"]]
    for col in ("pert_dose", "pert_time"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    log.info(f"[{geo_label}] {len(df):,} instances")

    return df


def read_gctx_batch(
    gctx_path: str,
    inst_ids:  List[str],
    gene_ids:  List[str],
    dtype:     type = np.float32,
) -> sp.csr_matrix:
    gctoo = parse_gctx(gctx_path, cid=inst_ids, rid=gene_ids)
    df    = gctoo.data_df.reindex(index=gene_ids, columns=inst_ids)
    X     = sp.csr_matrix(df.values.T.astype(dtype))
    del gctoo, df
    gc.collect()
    return X


def build_group_aware_batches(
    all_inst:   Dict[str, pd.DataFrame],
    group_cols: List[str],
    batch_size: int,
    ctrl_col:   str  = "pert_type",
    ctrl_vals:  list = None,
) -> List[List[Tuple[str, str]]]:
    
    frames = []
    for geo, inst in all_inst.items():
        tmp        = inst.copy()
        tmp["geo"] = geo
        frames.append(tmp)

    obs_all = pd.concat(frames, axis=0)

    if ctrl_vals and ctrl_col in obs_all.columns:
        ctrl_mask = obs_all[ctrl_col].isin(ctrl_vals)
        ctrl_obs  = obs_all[ctrl_mask]
        trt_obs   = obs_all[~ctrl_mask]
    else:
        ctrl_obs = pd.DataFrame()
        trt_obs  = obs_all

    available_cols = [c for c in group_cols if c in trt_obs.columns]
    grouped = trt_obs.groupby(available_cols, observed=True) \
        if available_cols else [(None, trt_obs)]

    batches:       List[List[Tuple[str, str]]] = []
    current_batch: List[Tuple[str, str]]       = []
    current_size:  int                         = 0

    for _, grp_df in grouped:
        grp_records = [(row["geo"], inst_id) for inst_id, row in grp_df.iterrows()]
        n_grp = len(grp_records)
        if n_grp > batch_size:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size  = 0
            for sub_start in range(0, n_grp, batch_size):
                batches.append(grp_records[sub_start:sub_start + batch_size])
            continue
        if current_size + n_grp > batch_size and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size  = 0
        current_batch.extend(grp_records)
        current_size += n_grp

    if current_batch:
        batches.append(current_batch)

    if len(ctrl_obs) > 0:
        ctrl_records   = [(row["geo"], inst_id) for inst_id, row in ctrl_obs.iterrows()]
        n_b            = len(batches)
        ctrl_per_batch = max(1, len(ctrl_records) // n_b)
        for i, batch in enumerate(batches):
            start = i * ctrl_per_batch
            end   = start + ctrl_per_batch if i < n_b - 1 else len(ctrl_records)
            batch.extend(ctrl_records[start:end])

    n_total = sum(len(b) for b in batches)
    log.info(
        f"Built {len(batches):,} batches | "
        f"total={n_total:,} | avg={n_total // max(len(batches), 1):,}"
    )
    return batches


def _process_batch(
    batch_idx:    int,
    batch_records: List[Tuple[str, str]],
    all_inst:     Dict[str, pd.DataFrame],
    all_gctx:     Dict[str, str],
    gene_ids:     List[str],
    var:          pd.DataFrame,
    batch_dir:    str,
    n_total:      int,
    batch_size:   int,
    dtype:        type,
) -> Tuple[int, str, Optional[str]]:

    worker_log = _get_logger(f"worker.batch_{batch_idx:04d}")
    batch_path = os.path.join(batch_dir, f"batch_{batch_idx:04d}.h5ad")

    if Path(batch_path).exists():
        worker_log.info(f"Batch {batch_idx:04d} exists — skipping")
        return batch_idx, batch_path, None

    try:
        geo_groups: Dict[str, List[str]] = {}
        for geo, inst_id in batch_records:
            geo_groups.setdefault(geo, []).append(inst_id)

        X_parts:   List[sp.csr_matrix] = []
        obs_parts: List[pd.DataFrame]  = []

        for geo, inst_ids_geo in geo_groups.items():
            X_geo   = read_gctx_batch(all_gctx[geo], inst_ids_geo, gene_ids, dtype)
            obs_geo = all_inst[geo].loc[inst_ids_geo].copy()
            X_parts.append(X_geo)
            obs_parts.append(obs_geo)
            del X_geo
            gc.collect()

        batch_inst_ids = [inst_id for _, inst_id in batch_records]
        X_concat       = sp.vstack(X_parts, format="csr")
        obs_concat     = pd.concat(obs_parts, axis=0)
        loc_map        = {iid: i for i, iid in enumerate(obs_concat.index)}
        row_order      = [loc_map[iid] for iid in batch_inst_ids]
        X_ordered      = X_concat[row_order, :]
        obs_ordered    = obs_concat.iloc[row_order].copy()

        del X_parts, obs_parts, X_concat, obs_concat
        gc.collect()

        adata = ad.AnnData(
            X   = X_ordered.astype(dtype),
            obs = obs_ordered,
            var = var.copy(),
            uns = {
                "data_level":    "Level3_INF_mlr12k",
                "normalization": "Q2NORM",
                "batch_idx":     batch_idx,
                "batch_size":    batch_size,
                "n_total":       n_total,
            },
        )
        os.makedirs(batch_dir, exist_ok=True)
        adata.write_h5ad(batch_path, compression="gzip")
        worker_log.info(f"Batch {batch_idx:04d} written: {adata.shape}")
        del adata, X_ordered, obs_ordered
        gc.collect()
        return batch_idx, batch_path, None

    except Exception as e:
        import traceback
        err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        worker_log.error(f"Batch {batch_idx:04d} FAILED: {err_msg}")
        return batch_idx, batch_path, err_msg


# %%
import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--config", default="/home/kyungan/scripts/BioTox/bio/configs/lincs_ext.json")
_p.add_argument("--run-keys", nargs="*", default=None)
_args, _ = _p.parse_known_args()

config   = _args.config
dir_data = "/home/kyungan/data/LINCS"

with open(config) as fr:
    lincs = json.load(fr)

assert "runs" in lincs, (
    "lincs.json must contain a 'runs' key with run configurations. "
    "See the schema comment in this file."
)

run_keys = _args.run_keys or list(lincs["runs"].keys())
log.info(f"Runs to execute: {run_keys}")

for run_key in run_keys:
    if run_key not in lincs["runs"]:
        log.error(f"Run '{run_key}' not found in lincs.json['runs'] — skipping")
        continue
    run_cfg = lincs["runs"][run_key].copy()
    run_cfg["name"] = run_key

    pert_types = run_cfg["pert_types"]
    ctrl_vals  = run_cfg["ctrl_vals"]
    group_cols = run_cfg["group_cols"]
    batch_size = int(run_cfg["batch_size"])
    n_jobs     = int(run_cfg["n_jobs"])
    batch_dir  = run_cfg["batch_dir"]
    dtype      = np.float32

    log.info(
        f"\n{'='*60}\n"
        f"Run: {run_cfg.get('name', batch_dir)}\n"
        f"  pert_types: {pert_types}\n"
        f"  ctrl_vals:  {ctrl_vals}\n"
        f"  group_cols: {group_cols}\n"
        f"  batch_size: {batch_size:,}\n"
        f"  n_jobs:     {n_jobs}\n"
        f"  batch_dir:  {batch_dir}\n"
        f"{'='*60}"
    )

    geo_keys  = [k for k in lincs if k != "runs"]
    first_geo = geo_keys[0]
    gene_info = load_gene_info(
        os.path.join(dir_data, first_geo, lincs[first_geo]["var"])
    )
    gene_ids  = list(gene_info.index.astype(str))
    var       = gene_info.copy()
    var.index.name = "gene_id"

    all_inst: Dict[str, pd.DataFrame] = {}
    all_gctx: Dict[str, str] = {}
    for geo in geo_keys:
        cfg = lincs[geo]
        gctx_path = os.path.join(dir_data, geo, cfg["gmt"])
        if not Path(gctx_path).exists():
            log.warning(f"GCTX not found: {gctx_path} — skipping {geo}")
            continue
        inst = load_inst_info(
            os.path.join(dir_data, geo, cfg["inst"]),
            cfg["phase"], geo,
        )
        if pert_types and "pert_type" in inst.columns:
            n_before = len(inst)
            inst     = inst[inst["pert_type"].isin(pert_types)].copy()
            log.info(f"[{geo}] filter: {n_before:,} → {len(inst):,}")
        if len(inst) == 0:
            log.warning(f"[{geo}] no instances after filter — skipping")
            continue
        all_inst[geo] = inst
        all_gctx[geo] = gctx_path

    batches = build_group_aware_batches(
        all_inst   = all_inst,
        group_cols = group_cols,
        batch_size = batch_size,
        ctrl_col   = "pert_type",
        ctrl_vals  = ctrl_vals,
    )
    n_total = sum(len(b) for b in batches)
    os.makedirs(batch_dir, exist_ok=True)

    log.info(f"Step 1: writing {len(batches)} batches (n_jobs={n_jobs})")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_process_batch)(
            batch_idx     = i,
            batch_records = batch,
            all_inst      = all_inst,
            all_gctx      = all_gctx,
            gene_ids      = gene_ids,
            var           = var,
            batch_dir     = batch_dir,
            n_total       = n_total,
            batch_size    = batch_size,
            dtype         = dtype,
        )
        for i, batch in enumerate(batches)
    )
    failed  = [(idx, err) for idx, _, err in results if err is not None]
    success = [idx         for idx, _, err in results if err is None]
    log.info(f"Step 1: {len(success)} succeeded | {len(failed)} failed")
    if failed:
        for idx, err in failed:
            log.error(f"  batch_{idx:04d}: {err[:200]}")
    
# %%
