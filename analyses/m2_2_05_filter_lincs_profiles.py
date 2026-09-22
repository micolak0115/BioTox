# -*- coding: utf-8 -*-
# %%
# filter.py
# ============================================================
# QC criteria:
#   1. replicate correlation < COR_THR (0.3)      — treatment groups
#   2. n_reps < MIN_REPS (2)                      — treatment groups
#   3. MT/ribo/hb MAD-based                       — remove outlier
#                                                    control cells only
#
# Output: filtered batch files in batch_dir + "_filter/"
#         ready for merge_batches() to be run separately
#
# Controls: kept (minus MT/ribo/hb outlier cells)
#           needed as TVN anchor in align.py
# ============================================================

import os
import gc
import json
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed

from memory_guard import MemoryGuard


def _get_logger(name=__name__):
    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

log   = _get_logger()
guard = MemoryGuard(
    min_free_gb      = 10.0,
    warn_pct         = 0.75,
    kill_pct         = 0.80,
    monitor_interval = 60.0,
    kill_interval    = 5.0,
)

# ============================================================
# CONSTANTS
# ============================================================
COR_THR   = 0.3   # replicate Pearson correlation threshold
MIN_REPS  = 2     # minimum replicates per group
MT_MADS   = 3     # MADs above median for MT%   (high = bad)
RIBO_MADS = 3     # MADs below median for ribo% (low  = bad)
HB_MADS   = 3     # MADs above median for HB%   (high = bad)


# ============================================================
# WORKER — correlation QC per batch
# ============================================================

def _compute_batch_cors(
    path_batch: str,
    group_cols: list,
    ctrl_vals:  list,
) -> tuple:
    """
    Load one batch, compute replicate correlations for all groups.
    Singletons (n < MIN_REPS) recorded with np.nan correlation.
    Returns (plate_num_replicates, same_plate_cors, error_msg or None).
    """
    try:
        a   = ad.read_h5ad(path_batch)
        X   = a.X
        if sp.issparse(X):
            X = X.toarray()
        X   = np.array(X, dtype=np.float32)
        obs = a.obs.copy()
        del a
        gc.collect()

        if "plate_id" not in obs.columns:
            obs["plate_id"] = [
                "_".join(x.split(":")[0].split("_")[:3])
                for x in obs.index
            ]

        trt_mask          = ~obs["pert_type"].isin(ctrl_vals)
        trt_obs           = obs[trt_mask]
        obs_label_to_iloc = {label: i for i, label in enumerate(obs.index)}
        available         = [c for c in group_cols if c in trt_obs.columns]

        plate_num_replicates = {}
        same_plate_cors      = {}

        for grp_key, grp_df in trt_obs.groupby(available, observed=True):
            n = len(grp_df)
            plate_num_replicates[grp_key] = n
            if n < MIN_REPS:
                same_plate_cors[grp_key] = np.nan
                continue
            iloc_idx    = [obs_label_to_iloc[l] for l in grp_df.index]
            X_grp       = X[iloc_idx]
            corr_matrix = np.corrcoef(X_grp)
            upper_tri   = corr_matrix[np.triu_indices(n, k=1)]
            same_plate_cors[grp_key] = float(np.mean(upper_tri))

        del X, obs, trt_obs
        gc.collect()
        return plate_num_replicates, same_plate_cors, None

    except Exception as e:
        import traceback
        return {}, {}, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# ============================================================
# MT/RIBO/HB QC — MAD-based thresholds from control cells
# ============================================================

def _mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


# ============================================================
# WORKER — MT/ribo/HB per batch (controls only)
# ============================================================

def _compute_batch_ctrl_qc(
    path_batch: str,
    ctrl_vals:  list,
    mt_idx:     np.ndarray,
    ribo_idx:   np.ndarray,
    hb_idx:     np.ndarray,
) -> tuple:
    """
    Load one batch, compute MT/ribo/HB pct for control cells only.
    Safe to run in parallel via loky — each worker opens its own file.
    Returns (mt_vals, ribo_vals, hb_vals, error_msg or None).
    All three are plain float lists — small, safe to serialize back.
    """
    try:
        a   = ad.read_h5ad(path_batch)
        X   = a.X
        if sp.issparse(X):
            X = X.toarray()
        X   = np.array(X, dtype=np.float32)
        obs = a.obs.copy()
        del a
        gc.collect()

        ctrl_mask = obs["pert_type"].isin(ctrl_vals)
        if ctrl_mask.sum() == 0:
            del X, obs
            gc.collect()
            return [], [], [], None

        X_ctrl = X[ctrl_mask.values]
        tot    = X_ctrl.sum(axis=1)
        tot    = np.where(tot == 0, 1.0, tot)

        mt_vals   = (X_ctrl[:, mt_idx].sum(axis=1)   / tot * 100).tolist()
        ribo_vals = (X_ctrl[:, ribo_idx].sum(axis=1) / tot * 100).tolist()
        hb_vals   = (X_ctrl[:, hb_idx].sum(axis=1)   / tot * 100).tolist()

        del X, X_ctrl, obs
        gc.collect()
        return mt_vals, ribo_vals, hb_vals, None

    except Exception as e:
        import traceback
        return [], [], [], f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def _compute_ctrl_qc_thresholds(
    batch_paths: list,
    ctrl_vals:   list,
    mt_idx:      np.ndarray,
    ribo_idx:    np.ndarray,
    hb_idx:      np.ndarray,
    n_jobs:      int,
) -> dict:
    """
    Compute MAD-based MT/ribo/HB thresholds from all control cells.
    Parallel across batch files via loky — same pattern as correlation QC.
    Peak RAM per worker = one batch.
    """
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_compute_batch_ctrl_qc)(
            path_batch = str(bp),
            ctrl_vals  = ctrl_vals,
            mt_idx     = mt_idx,
            ribo_idx   = ribo_idx,
            hb_idx     = hb_idx,
        )
        for bp in batch_paths
    )

    all_mt, all_ribo, all_hb = [], [], []
    for mt_vals, ribo_vals, hb_vals, err in results:
        if err is not None:
            log.error(f"  ctrl QC worker failed: {err[:200]}")
            continue
        all_mt.extend(mt_vals)
        all_ribo.extend(ribo_vals)
        all_hb.extend(hb_vals)

    all_mt   = np.array(all_mt,   dtype=np.float32)
    all_ribo = np.array(all_ribo, dtype=np.float32)
    all_hb   = np.array(all_hb,   dtype=np.float32)

    log.info(
        f"Control QC | n_ctrl_cells={len(all_mt):,} | "
        f"MT  median={np.median(all_mt):.2f}% MAD={_mad(all_mt):.2f} | "
        f"ribo median={np.median(all_ribo):.2f}% MAD={_mad(all_ribo):.2f} | "
        f"HB  median={np.median(all_hb):.2f}% MAD={_mad(all_hb):.2f}"
    )

    return {
        "mt_median":   float(np.median(all_mt)),
        "mt_mad":      _mad(all_mt),
        "mt_high":     float(np.median(all_mt))   + MT_MADS   * _mad(all_mt),
        "ribo_median": float(np.median(all_ribo)),
        "ribo_mad":    _mad(all_ribo),
        "ribo_low":    float(np.median(all_ribo)) - RIBO_MADS * _mad(all_ribo),
        "hb_median":   float(np.median(all_hb)),
        "hb_mad":      _mad(all_hb),
        "hb_high":     float(np.median(all_hb))   + HB_MADS   * _mad(all_hb),
    }


def _filter_batch(
    path_batch:     str,
    path_out_batch: str,
    bad_plate_keys: set,
    ctrl_vals:      list,
    mt_idx:         np.ndarray,
    ribo_idx:       np.ndarray,
    hb_idx:         np.ndarray,
    thresholds:     dict,
) -> tuple:
    """
    Load one batch, apply all filters, write filtered batch.
    Returns (path_out_batch, n_trt_pass, n_trt_fail,
             n_ctrl_pass, n_ctrl_fail, error_msg or None).
    """
    try:
        a   = ad.read_h5ad(path_batch)
        X   = a.X
        if sp.issparse(X):
            X = X.toarray()
        X   = np.array(X, dtype=np.float32)
        obs = a.obs.copy()
        del a
        gc.collect()

        if "plate_id" not in obs.columns:
            obs["plate_id"] = [
                "_".join(x.split(":")[0].split("_")[:3])
                for x in obs.index
            ]

        trt_mask  = ~obs["pert_type"].isin(ctrl_vals)
        ctrl_mask =  obs["pert_type"].isin(ctrl_vals)

        # treatments: remove bad plate keys
        obs["plate_key"] = obs[["pert_id", "pert_dose", "plate_id"]].apply(tuple, axis=1)
        trt_keep         = trt_mask & ~obs["plate_key"].isin(bad_plate_keys)
        obs              = obs.drop(columns=["plate_key"])
        n_trt_pass       = int(trt_keep.sum())
        n_trt_fail       = int((trt_mask & ~trt_keep).sum())

        # controls: remove individual outlier cells by MT/ribo/hb
        ctrl_keep_arr = np.zeros(len(obs), dtype=bool)
        n_ctrl_pass   = 0
        n_ctrl_fail   = 0
        if ctrl_mask.sum() > 0:
            ctrl_iloc      = np.where(ctrl_mask.values)[0]
            X_ctrl         = X[ctrl_iloc]
            tot            = X_ctrl.sum(axis=1)
            tot            = np.where(tot == 0, 1.0, tot)
            pct_mt         = X_ctrl[:, mt_idx].sum(axis=1)   / tot * 100
            pct_ribo       = X_ctrl[:, ribo_idx].sum(axis=1) / tot * 100
            pct_hb         = X_ctrl[:, hb_idx].sum(axis=1)   / tot * 100
            ctrl_cell_keep = (
                (pct_mt   <= thresholds["mt_high"])  &
                (pct_ribo >= thresholds["ribo_low"]) &
                (pct_hb   <= thresholds["hb_high"])
            )
            ctrl_keep_arr[ctrl_iloc] = ctrl_cell_keep
            n_ctrl_pass = int(ctrl_cell_keep.sum())
            n_ctrl_fail = int((~ctrl_cell_keep).sum())
            del X_ctrl, ctrl_cell_keep

        keep = trt_keep.values | ctrl_keep_arr
        del X
        gc.collect()

        if keep.sum() == 0:
            del obs
            gc.collect()
            return path_out_batch, n_trt_pass, n_trt_fail, n_ctrl_pass, n_ctrl_fail, None

        a_tmp          = ad.read_h5ad(path_batch)
        a_filtered     = a_tmp[keep].copy()
        a_filtered.obs = obs[keep].copy()
        del a_tmp, obs
        gc.collect()

        a_filtered.write_h5ad(path_out_batch, compression="gzip")
        del a_filtered
        gc.collect()

        return path_out_batch, n_trt_pass, n_trt_fail, n_ctrl_pass, n_ctrl_fail, None

    except Exception as e:
        import traceback
        return path_out_batch, 0, 0, 0, 0, \
               f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# ============================================================
# ENTRY POINT
# ============================================================

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

assert "runs" in lincs, "lincs.json must contain a 'runs' key."
run_keys = _args.run_keys or list(lincs["runs"].keys())
log.info(
    f"Runs: {run_keys} | "
    f"COR_THR={COR_THR} | MIN_REPS={MIN_REPS} | "
    f"MT_MADS={MT_MADS} | RIBO_MADS={RIBO_MADS} | HB_MADS={HB_MADS}"
)

# %%
for run_key in run_keys:
    if run_key not in lincs["runs"]:
        log.warning(f"[{run_key}] not in config — skipping")
        continue
    run_cfg = lincs["runs"][run_key]
    ctrl_vals   = run_cfg["ctrl_vals"]
    group_cols  = run_cfg["group_cols"]
    n_jobs      = int(run_cfg["n_jobs"])
    batch_dir   = run_cfg["batch_dir"]
    path_output = run_cfg["path_output"]
    path_save   = str(path_output).replace(".h5ad", "_filter.h5ad")

    missing = [k for k in ["ctrl_vals", "group_cols", "n_jobs", "batch_dir", "path_output"]
               if k not in run_cfg]
    if missing:
        log.warning(f"[{run_key}] missing keys {missing} — skipping")
        continue

    batch_paths = sorted(Path(batch_dir).glob("batch_*.h5ad"))
    if not batch_paths:
        log.warning(f"[{run_key}] no batch files in {batch_dir} — skipping")
        continue

    log.info(
        f"\n{'='*60}\n"
        f"Filter: {run_key}\n"
        f"  batch_dir:   {batch_dir} ({len(batch_paths)} valid files)\n"
        f"  group_cols:  {group_cols}\n"
        f"  ctrl_vals:   {ctrl_vals}\n"
        f"  n_jobs:      {n_jobs}\n"
        f"  COR_THR:     {COR_THR} | MIN_REPS: {MIN_REPS}\n"
        f"  path_save:   {path_save}\n"
        f"{'='*60}"
    )

    path_pkl_cor        = str(path_output).replace(".h5ad", "_cor_results.pkl")
    path_pkl_bad_plates = str(path_output).replace(".h5ad", "_bad_plate_keys.pkl")
    path_pkl_thresholds = str(path_output).replace(".h5ad", "_ctrl_qc_thresholds.pkl")

    # ----------------------------------------------------------
    # step 1: replicate correlation QC
    # ----------------------------------------------------------
    if Path(path_pkl_cor).exists():
        log.info(f"[{run_key}] Loading correlation results from {path_pkl_cor}")
        with open(path_pkl_cor, "rb") as fr:
            cor_data = pickle.load(fr)
        plate_num_replicates = cor_data["plate_num_replicates"]
        same_plate_cors      = cor_data["same_plate_cors"]
        log.info(
            f"[{run_key}] loaded | n_groups={len(same_plate_cors):,} | "
            f"mean={np.nanmean([v for v in same_plate_cors.values() if not np.isnan(v)]):.3f}"
        )
    else:
        guard.check_available(
            needed_gb = guard.array_gb((10_000, 12_328)) * n_jobs,
            label     = f"{run_key} correlation QC"
        )
        with guard:
            results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
                delayed(_compute_batch_cors)(
                    path_batch = str(bp),
                    group_cols = group_cols,
                    ctrl_vals  = ctrl_vals,
                )
                for bp in batch_paths
            )

        plate_num_replicates = {}
        same_plate_cors      = {}
        for i, (reps, cors, err) in enumerate(results):
            if err is not None:
                log.error(f"  [{run_key}] batch_{i:04d}: {err[:200]}")
                continue
            plate_num_replicates.update(reps)
            same_plate_cors.update(cors)

        valid_cors = [v for v in same_plate_cors.values() if not np.isnan(v)]
        log.info(
            f"[{run_key}] n_groups={len(same_plate_cors):,} | "
            f"mean={np.mean(valid_cors):.3f} ± {np.std(valid_cors):.3f}"
        )

        with open(path_pkl_cor, "wb") as fw:
            pickle.dump({
                "plate_num_replicates": plate_num_replicates,
                "same_plate_cors":      same_plate_cors,
            }, fw)
        log.info(f"[{run_key}] correlation results saved → {path_pkl_cor}")
        guard.collect()

    # ----------------------------------------------------------
    # step 2: bad plate keys
    # ----------------------------------------------------------
    if Path(path_pkl_bad_plates).exists():
        log.info(f"[{run_key}] Loading bad_plate_keys from {path_pkl_bad_plates}")
        with open(path_pkl_bad_plates, "rb") as fr:
            bad_plate_keys = pickle.load(fr)
        log.info(f"[{run_key}] loaded bad_plate_keys: {len(bad_plate_keys):,}")
    else:
        bad_by_cor  = {k for k, v in same_plate_cors.items()
                       if not np.isnan(v) and v < COR_THR}
        bad_by_reps = {k for k, v in plate_num_replicates.items()
                       if v < MIN_REPS}
        bad_plate_keys = bad_by_cor | bad_by_reps
        log.info(
            f"[{run_key}] bad_by_cor={len(bad_by_cor):,} | "
            f"bad_by_reps={len(bad_by_reps):,} | "
            f"total={len(bad_plate_keys):,}"
        )
        with open(path_pkl_bad_plates, "wb") as fw:
            pickle.dump(bad_plate_keys, fw)
        log.info(f"[{run_key}] bad_plate_keys saved → {path_pkl_bad_plates}")

    # ----------------------------------------------------------
    # step 3: plots
    # ----------------------------------------------------------
    qc_pass  = {k: v for k, v in same_plate_cors.items()
                if k not in bad_plate_keys and not np.isnan(v)}
    qc_fail  = {k: v for k, v in same_plate_cors.items()
                if k in bad_plate_keys and not np.isnan(v)}
    rep_pass = {k: v for k, v in plate_num_replicates.items()
                if k not in bad_plate_keys}
    rep_fail = {k: v for k, v in plate_num_replicates.items()
                if k in bad_plate_keys}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f"{run_key} — QC (COR_THR={COR_THR} | MIN_REPS={MIN_REPS})", fontsize=13)
    axes[0].hist([v for v in same_plate_cors.values() if not np.isnan(v)],
                 bins=500, color="gray")
    axes[0].axvline(COR_THR, color="firebrick", linestyle="--")
    axes[0].set_xlabel("replicate correlation")
    axes[0].set_ylabel("groups")
    axes[0].set_title(f"All groups (n={len(same_plate_cors):,})")
    axes[1].hist(list(qc_pass.values()), bins=500, color="gray")
    axes[1].set_xlabel("replicate correlation")
    axes[1].set_title(f"QC pass (n={len(qc_pass):,})")
    axes[2].scatter(list(rep_pass.values()), list(qc_pass.values()),
                    color="gray", alpha=0.5, s=10, label=f"pass ({len(qc_pass):,})")
    axes[2].scatter(list(rep_fail.values()),
                    [qc_fail.get(k, np.nan) for k in rep_fail],
                    color="red", alpha=0.3, s=5, label=f"fail ({len(rep_fail):,})")
    axes[2].axhline(COR_THR, color="firebrick", linestyle="--")
    axes[2].set_xticks(range(1, 41, 2))
    axes[2].set_xticklabels(range(1, 41, 2), rotation=90)
    axes[2].set_xlabel("number of replicates")
    axes[2].set_ylabel("replicate correlation")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Pass vs fail")
    plt.tight_layout()
    plt.show()
    plt.close()

    # ----------------------------------------------------------
    # step 4: MT/ribo/HB thresholds — parallel, controls only
    # ----------------------------------------------------------
    if Path(path_pkl_thresholds).exists():
        log.info(f"[{run_key}] Loading ctrl QC thresholds from {path_pkl_thresholds}")
        with open(path_pkl_thresholds, "rb") as fr:
            thresholds = pickle.load(fr)
        log.info(
            f"[{run_key}] loaded thresholds | "
            f"MT high={thresholds['mt_high']:.2f}% | "
            f"ribo low={thresholds['ribo_low']:.2f}% | "
            f"HB high={thresholds['hb_high']:.2f}%"
        )
        # still need gene indices for per-cell filtering
        a_tmp    = ad.read_h5ad(batch_paths[0], backed="r")
        var      = a_tmp.var.copy()
        a_tmp.file.close()
        mt_idx   = np.where(var["gene_name"].str.upper().str.startswith("MT-"))[0]
        ribo_idx = np.where(var["gene_name"].str.upper().str.startswith(("RPS", "RPL")))[0]
        hb_idx   = np.where(var["gene_name"].str.upper().str.contains(
                        r"^HB[^PSE]", regex=True))[0]
        del var, a_tmp
    else:
        a_tmp    = ad.read_h5ad(batch_paths[0], backed="r")
        var      = a_tmp.var.copy()
        a_tmp.file.close()
        mt_idx   = np.where(var["gene_name"].str.upper().str.startswith("MT-"))[0]
        ribo_idx = np.where(var["gene_name"].str.upper().str.startswith(("RPS", "RPL")))[0]
        hb_idx   = np.where(var["gene_name"].str.upper().str.contains(
                        r"^HB[^PSE]", regex=True))[0]
        del var, a_tmp
        log.info(f"[{run_key}] MT={len(mt_idx)} | ribo={len(ribo_idx)} | hb={len(hb_idx)}")

        guard.check_available(
            needed_gb = guard.array_gb((10_000, 12_328)) * n_jobs,
            label     = f"{run_key} ctrl QC"
        )
        thresholds = _compute_ctrl_qc_thresholds(
            batch_paths = batch_paths,
            ctrl_vals   = ctrl_vals,
            mt_idx      = mt_idx,
            ribo_idx    = ribo_idx,
            hb_idx      = hb_idx,
            n_jobs      = n_jobs,
        )
        with open(path_pkl_thresholds, "wb") as fw:
            pickle.dump(thresholds, fw)
        log.info(f"[{run_key}] thresholds saved → {path_pkl_thresholds}")

        log.info(
            f"[{run_key}] MT  thr_high={thresholds['mt_high']:.2f}% | "
            f"ribo thr_low={thresholds['ribo_low']:.2f}% | "
            f"HB  thr_high={thresholds['hb_high']:.2f}%"
        )
        guard.collect()

   # ----------------------------------------------------------
    # step 5: filter each batch — parallel via loky
    # ----------------------------------------------------------
    guard.log_info(f"[{run_key}] filtering batches | ")

    dir_filtered = str(batch_dir).rstrip("/") + "_filter"
    os.makedirs(dir_filtered, exist_ok=True)

    # build jobs — skip already-done batches
    jobs_to_run   = []
    jobs_to_skip  = []
    for bp in batch_paths:
        path_out = os.path.join(dir_filtered, bp.name)
        if Path(path_out).exists():
            try:
                a_check = ad.read_h5ad(path_out, backed="r")
                a_check.file.close()
                jobs_to_skip.append(path_out)
            except Exception:
                jobs_to_run.append((bp, path_out))
        else:
            jobs_to_run.append((bp, path_out))

    log.info(
        f"[{run_key}] batches: {len(jobs_to_skip)} skip | "
        f"{len(jobs_to_run)} to filter"
    )

    guard.check_available(
        needed_gb = guard.array_gb((10_000, 12_328)) * min(n_jobs, len(jobs_to_run)),
        label     = f"{run_key} filter batches"
    )

    with guard:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_filter_batch)(
                path_batch     = str(bp),
                path_out_batch = path_out,
                bad_plate_keys = bad_plate_keys,
                ctrl_vals      = ctrl_vals,
                mt_idx         = mt_idx,
                ribo_idx       = ribo_idx,
                hb_idx         = hb_idx,
                thresholds     = thresholds,
            )
            for bp, path_out in jobs_to_run
        )

    # assemble results
    filtered_batch_paths = list(jobs_to_skip)
    n_trt_pass  = 0
    n_trt_fail  = 0
    n_ctrl_pass = 0
    n_ctrl_fail = 0

    for path_out, ntp, ntf, ncp, ncf, err in results:
        if err is not None:
            log.error(f"  FAILED {Path(path_out).name}: {err[:200]}")
            continue
        if Path(path_out).exists():
            filtered_batch_paths.append(path_out)
        n_trt_pass  += ntp
        n_trt_fail  += ntf
        n_ctrl_pass += ncp
        n_ctrl_fail += ncf

    filtered_batch_paths = sorted(filtered_batch_paths)
    guard.collect()

    log.info(
        f"[{run_key}] filter done\n"
        f"  treatments: {n_trt_pass:,} pass | {n_trt_fail:,} removed\n"
        f"  controls:   {n_ctrl_pass:,} pass | {n_ctrl_fail:,} removed\n"
        f"  filtered batches: {len(filtered_batch_paths)} → {dir_filtered}"
    )
    log.info(f"[{run_key}] done — run merge_batches({dir_filtered!r}, {path_save!r}) next")

log.info("All runs complete.")


# %%
# select.py
# ============================================================
# Subset filter.h5ad to target cell lines + valid controls.
# Adds `control` and `plate_id` columns required by align.py.
# Preserves all obsm embeddings (X_scGPT, etc.) for the
# selected rows.
# ============================================================

import gc
import json
import logging
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path


# ────────────────────────────────────────────────────────────
# logging
# ────────────────────────────────────────────────────────────

def _get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

log = _get_logger()


# ────────────────────────────────────────────────────────────
# X layout detection
# ────────────────────────────────────────────────────────────

def _check_x_layout(src_path: str) -> dict:
    """
    Return X layout: kind, n_rows, n_cols, n_nnz, is_dense.
    Uses anndata backed read for shape to handle all obs/var
    index naming conventions across anndata versions.
    """
    a = ad.read_h5ad(src_path, backed="r")
    n_obs, n_var = a.shape
    a.file.close()
    del a

    with h5py.File(src_path, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Group):
            enc      = x.attrs.get("encoding-type", "csr_matrix")
            n_nnz    = x["data"].shape[0]
            is_dense = (n_nnz == n_obs * n_var)
            return dict(kind="csr", enc=enc,
                        n_rows=n_obs, n_cols=n_var,
                        n_nnz=n_nnz, is_dense=is_dense)
        else:
            return dict(kind="dense", enc="array",
                        n_rows=n_obs, n_cols=n_var,
                        n_nnz=n_obs * n_var, is_dense=True)


# ────────────────────────────────────────────────────────────
# X row extraction → always returns dense float32
# ────────────────────────────────────────────────────────────
def _read_rows_h5py(
    src_path: str,
    row_sel:  np.ndarray,
    n_cols:   int,
    is_dense: bool,
) -> np.ndarray:
    row_sel = np.asarray(row_sel, dtype=np.int64)
    n_out   = len(row_sel)
    out     = np.zeros((n_out, n_cols), dtype=np.float32)

    with h5py.File(src_path, "r") as f:
        x_grp = f["X"]

        if isinstance(x_grp, h5py.Group):
            data_ds = x_grp["data"]

            if is_dense:
                i = out_i = 0
                while i < n_out:
                    j = i + 1
                    while j < n_out and row_sel[j] == row_sel[j - 1] + 1:
                        j += 1
                    first_row, n_run = int(row_sel[i]), j - i
                    s     = first_row * n_cols
                    chunk = np.asarray(data_ds[s:s + n_run * n_cols],
                                       dtype=np.float32)
                    out[out_i:out_i + n_run] = chunk.reshape(n_run, n_cols)
                    i, out_i = j, out_i + n_run

            else:
                indices_ds = x_grp["indices"]
                indptr     = x_grp["indptr"][:].astype(np.int64)
                log.info(
                    f"  indptr loaded | len={len(indptr):,} | "
                    f"[-1]={indptr[-1]:,}"
                )

                sample = indptr[::max(1, len(indptr) // 10_000)]
                if np.any(np.diff(sample) < 0):
                    raise ValueError(
                        "indptr is non-monotonic — possible int32 overflow "
                        "corruption. Regenerate filter.h5ad via merge_batches."
                    )

                i = out_i = 0
                while i < n_out:
                    j = i + 1
                    while j < n_out and row_sel[j] == row_sel[j - 1] + 1:
                        j += 1
                    first_row = int(row_sel[i])
                    last_row  = int(row_sel[j - 1])
                    n_run     = j - i
                    s = int(indptr[first_row])
                    e = int(indptr[last_row + 1])

                    if s < 0 or e < s:
                        raise ValueError(
                            f"Invalid indptr rows {first_row}..{last_row}: "
                            f"s={s:,}, e={e:,} — source file corrupted."
                        )

                    if e > s:
                        chunk_data = np.asarray(data_ds[s:e],    dtype=np.float32)
                        chunk_idx  = np.asarray(indices_ds[s:e], dtype=np.int32)
                        local_iptr = (indptr[first_row:last_row + 2] - s
                                      ).astype(np.int32)
                        local_csr  = sp.csr_matrix(
                            (chunk_data, chunk_idx, local_iptr),
                            shape=(n_run, n_cols),
                        )
                        out[out_i:out_i + n_run] = local_csr.toarray()

                    i, out_i = j, out_i + n_run

        else:
            i = out_i = 0
            while i < n_out:
                j = i + 1
                while j < n_out and row_sel[j] == row_sel[j - 1] + 1:
                    j += 1
                first_row, n_run = int(row_sel[i]), j - i
                out[out_i:out_i + n_run] = np.asarray(
                    x_grp[first_row:first_row + n_run], dtype=np.float32
                )
                i, out_i = j, out_i + n_run

    return out


# ────────────────────────────────────────────────────────────
# obsm row extraction
# ────────────────────────────────────────────────────────────
def _read_obsm_h5py(src_path: str, row_sel: np.ndarray) -> dict:
    """
    Read selected rows from every dense dataset in /obsm.
    
    Loads each obsm dataset fully into RAM then selects rows with
    numpy fancy indexing — one h5py read per key regardless of how
    scattered the selected rows are. Much faster than batched h5py
    reads for typical embedding sizes (512 dims × 577K rows ≈ 1.2 GB).
    """
    row_sel = np.asarray(row_sel, dtype=np.int64)
    result  = {}

    with h5py.File(src_path, "r") as f:
        if "obsm" not in f:
            return result

        for key in f["obsm"].keys():
            item = f["obsm"][key]
            if not isinstance(item, h5py.Dataset):
                log.warning(f"  obsm['{key}'] is not a dataset — skipping")
                continue

            size_gb = item.nbytes / 1e9
            log.info(f"  reading obsm['{key}'] ({item.shape}, {size_gb:.2f} GB) ...")

            # single read → numpy fancy index: always faster than
            # N scattered h5py slice calls
            arr = np.asarray(item[:], dtype=np.float32)   # full load
            result[key] = arr[row_sel]                     # row select
            del arr

    return result


# ────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────

def _plate_id_from_index(index) -> list:
    return ["_".join(s.split(":")[0].split("_")[:3]) for s in index]


def _atomic_write(adata: ad.AnnData, dst: str, **kwargs) -> None:
    tmp = dst + ".tmp"
    adata.write_h5ad(tmp, **kwargs)
    os.replace(tmp, dst)


def _validate(dst: str, required_cols: set) -> None:
    a = ad.read_h5ad(dst, backed="r")
    size_mb = Path(dst).stat().st_size / 1e6
    shape   = a.shape
    cols    = set(a.obs.columns)
    xtype   = type(a.X).__name__
    obsm_k  = list(a.obsm.keys())
    a.file.close()
    log.info(
        f"  ✓ {Path(dst).name} | {size_mb:.1f} MB | "
        f"shape={shape} | X={xtype} | "
        f"obsm={obsm_k} | obs_cols={sorted(cols)}"
    )
    missing = required_cols - cols
    if missing:
        raise RuntimeError(f"Missing obs columns for align.py: {missing}")


def get_control_mask(obs: pd.DataFrame, run_key: str, run_cfg: dict) -> pd.Series:
    ctrl_vals = run_cfg["ctrl_vals"]
    is_ctrl   = obs["pert_type"].isin(ctrl_vals)
    if run_key == "chemical":
        return is_ctrl & obs["pert_iname"].isin(run_cfg["ctrl_cp"])
    elif run_key == "genetic":
        valid = run_cfg.get("ctrl_sh", []) + run_cfg.get("ctrl_oe", [])
        return is_ctrl & obs["pert_iname"].isin(valid)
    elif run_key == "biologic":
        return is_ctrl & obs["pert_iname"].isin(run_cfg["ctrl_lig"])
    else:
        raise ValueError(f"Unknown run_key: {run_key!r}")


# ────────────────────────────────────────────────────────────
# config
# ────────────────────────────────────────────────────────────

# %%
# reuse _args.config and _args.run_keys from the filter section above

with open(_args.config) as fr:
    lincs = json.load(fr)

assert "runs" in lincs
run_keys = _args.run_keys or list(lincs["runs"].keys())
log.info(f"run_keys: {run_keys}")


# ────────────────────────────────────────────────────────────
# select
# ────────────────────────────────────────────────────────────

# %%
REQUIRED_COLS = {"control", "plate_id", "cell_id", "pert_type", "pert_iname"}

for run_key in run_keys:
    if run_key not in lincs["runs"]:
        log.warning(f"[{run_key}] not in config — skipping")
        continue

    run_cfg   = lincs["runs"][run_key]
    ctrl_vals = run_cfg["ctrl_vals"]

    src_path = run_cfg["path_output"].replace(".h5ad", "_filter.h5ad")
    dst_path = src_path.replace(".h5ad", "_select.h5ad")

    log.info(
        f"\n{'='*60}\n"
        f"  run_key : {run_key}\n"
        f"  src     : {src_path}  "
        f"({Path(src_path).stat().st_size / 1e9:.2f} GB)\n"
        f"  dst     : {dst_path}\n"
        f"{'='*60}"
    )

    # ── step 1: inspect X layout ─────────────────────────────
    layout = _check_x_layout(src_path)
    log.info(
        f"[{run_key}] X layout | kind={layout['kind']} | "
        f"n_rows={layout['n_rows']:,} | n_cols={layout['n_cols']:,} | "
        f"n_nnz={layout['n_nnz']:,} | is_dense={layout['is_dense']}"
    )
    n_cols   = layout["n_cols"]
    is_dense = layout["is_dense"]

    # ── step 2: load metadata only ───────────────────────────
    log.info(f"[{run_key}] reading metadata ...")
    adata_meta = ad.read_h5ad(src_path, backed="r")
    try:
        obs = adata_meta.obs.copy()
        var = adata_meta.var.copy()
        uns = dict(adata_meta.uns)
    finally:
        adata_meta.file.close()
        del adata_meta

    # ── step 3: build row mask ────────────────────────────────
    cell_mask     = obs["cell_id"].isin(run_cfg["cell_id"])
    is_trt        = ~obs["pert_type"].isin(ctrl_vals)
    is_valid_ctrl = get_control_mask(obs, run_key, run_cfg)
    final_mask    = cell_mask & (is_trt | is_valid_ctrl)

    n_before = len(obs)
    n_after  = int(final_mask.sum())
    log.info(
        f"[{run_key}] cells: {n_before:,} → {n_after:,} "
        f"({n_before - n_after:,} removed)"
    )

    ctrl_survived = obs.loc[final_mask & is_valid_ctrl, "pert_iname"].value_counts()
    log.info(f"  ctrl survivors:\n{ctrl_survived.to_string()}")

    if n_after == 0:
        log.warning(f"[{run_key}] no cells survive select — skipping")
        continue

    row_sel = np.sort(np.where(final_mask.values)[0]).astype(np.int64)

    # ── step 4a: read X rows → dense float32 ─────────────────
    expected_gb = n_after * n_cols * 4 / 1e9
    log.info(
        f"[{run_key}] reading X ({n_after:,} rows, "
        f"≈ {expected_gb:.2f} GB) ..."
    )
    X_sel = _read_rows_h5py(src_path, row_sel, n_cols, is_dense)
    log.info(f"  X_sel shape={X_sel.shape} dtype={X_sel.dtype}")
    gc.collect()

    # ── step 4b: read obsm rows ───────────────────────────────
    log.info(f"[{run_key}] reading obsm ...")
    obsm_sel = _read_obsm_h5py(src_path, row_sel)
    if obsm_sel:
        log.info(f"  obsm keys: {list(obsm_sel.keys())}")
    else:
        log.info("  no obsm found in source file")
    gc.collect()

    # ── step 5: build output AnnData ─────────────────────────
    obs_sel = obs.iloc[row_sel].copy()

    if "control" not in obs_sel.columns:
        obs_sel["control"] = (
            obs_sel["pert_type"].isin(ctrl_vals).astype(np.int8)
        )
        log.info("  added 'control' column")

    if "plate_id" not in obs_sel.columns:
        obs_sel["plate_id"] = _plate_id_from_index(obs_sel.index)
        log.info("  added 'plate_id' column")

    adata_select = ad.AnnData(X=X_sel, obs=obs_sel, var=var, uns=uns)

    for k, v in obsm_sel.items():
        adata_select.obsm[k] = v

    del X_sel, obs_sel, obsm_sel
    gc.collect()

    # ── step 6: atomic write ──────────────────────────────────
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    log.info(f"[{run_key}] writing → {dst_path}")
    _atomic_write(adata_select, dst_path)
    del adata_select
    gc.collect()

    # ── step 7: validate ─────────────────────────────────────
    _validate(dst_path, REQUIRED_COLS)
    log.info(f"[{run_key}] done ✓")

log.info("All select runs complete.")
# %%