# %%
# merge_batches.py
# ============================================================
# Merge filtered batch h5ad files into one merged h5ad.
#
# Parallelism:
#   step 2: shape scan     — parallel reads (loky)
#   step 3: obs + X_scGPT — parallel reads (loky)
#   step 6: X copy         — sequential (single output file)
# ============================================================

import os
import gc
import json
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import h5py
from pathlib import Path
from joblib import Parallel, delayed

from memory_guard import MemoryGuard


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

# %%
# ============================================================
# WORKER — scan one batch shape
# ============================================================

def _scan_batch_shape(
    path_batch: str,
    index_key:  str,
) -> tuple:
    """
    Read shape metadata from one batch via h5py.
    Read-only — safe to parallelize.
    Returns (n_obs, nnz, n_vars, has_scgpt, error_msg or None).
    """
    try:
        with h5py.File(path_batch, "r") as f:
            x         = f["X"]
            nnz_b     = int(x["data"].shape[0])
            n_obs_b   = int(x["indptr"].shape[0]) - 1
            shape_a   = x.attrs.get("shape", None)
            n_vars_b  = int(shape_a[1]) if shape_a is not None \
                        else int(f["var"][index_key].shape[0])
            has_scgpt = "obsm" in f and "X_scGPT" in f["obsm"]
        return n_obs_b, nnz_b, n_vars_b, has_scgpt, None
    except Exception as e:
        import traceback
        return 0, 0, 0, False, \
               f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# ============================================================
# WORKER — read obs + X_scGPT from one batch
# ============================================================

def _read_batch_obs_emb(
    path_batch: str,
    has_scgpt:  bool,
) -> tuple:
    """
    Read obs DataFrame and X_scGPT embedding from one batch.
    Read-only — safe to parallelize.
    Returns (obs_df, emb_array or None, error_msg or None).
    """
    try:
        a   = ad.read_h5ad(path_batch, backed="r")
        obs = a.obs.copy()
        a.file.close()
        del a
        gc.collect()

        emb = None
        if has_scgpt:
            with h5py.File(path_batch, "r") as f:
                emb = f["obsm"]["X_scGPT"][:].astype(np.float32)

        return obs, emb, None

    except Exception as e:
        import traceback
        return None, None, \
               f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# ============================================================
# MERGE BATCHES
# ============================================================

def merge_batches(
    batch_dir:   str,
    path_output: str,
    pattern:     str         = "batch_*.h5ad",
    n_jobs:      int         = 10,
    guard:       MemoryGuard = None,
) -> None:
    """
    Merge all batch h5ad files into one merged h5ad.

    Collects:
        var    — cleaned via h5py (avoids IORegistryError)
        obs    — parallel reads via loky
        X_scGPT— parallel reads via loky, stacked
        X      — sparse CSR pre-allocated, filled batch by batch

    Parameters
    ----------
    batch_dir   : directory containing batch_NNNN.h5ad files
    path_output : output merged h5ad path
    pattern     : glob pattern for batch files
    n_jobs      : workers for parallel steps (shape scan + obs collection)
    guard       : MemoryGuard instance
    """
    if guard is None:
        guard = MemoryGuard()

    batch_paths = sorted(Path(batch_dir).glob(pattern))
    if not batch_paths:
        raise RuntimeError(f"No files matching '{pattern}' in {batch_dir}")

    log.info(
        f"Merging {len(batch_paths)} batches → {path_output} | "
        f"n_jobs={n_jobs}"
    )
    guard.log_info("merge startup | ")

    # ----------------------------------------------------------
    # 1. build clean var via h5py — avoids IORegistryError
    # ----------------------------------------------------------
    log.info("Building clean var...")
    with h5py.File(batch_paths[0], "r") as f:
        var_grp   = f["var"]
        var_keys  = list(var_grp.keys())
        index_key = var_grp.attrs.get("_index", var_keys[0])
        idx       = var_grp[index_key][:].astype(str)
        var_data  = {}
        for k in var_keys:
            if k == index_key:
                continue
            ds = var_grp[k]
            if not isinstance(ds, h5py.Dataset):
                continue
            raw = ds[:]
            var_data[k] = np.array([
                v.decode() if isinstance(v, bytes) else str(v)
                for v in raw
            ]) if raw.dtype == object or raw.dtype.kind in ("O", "S") else raw

    var_clean = pd.DataFrame(var_data, index=pd.Index(idx, name="gene_id"))
    log.info(f"var: {len(var_clean):,} genes | dtypes: {var_clean.dtypes.to_dict()}")

    # ----------------------------------------------------------
    # 2. parallel shape scan — read-only, tiny per batch
    # ----------------------------------------------------------
    log.info(f"Scanning batch shapes (n_jobs={n_jobs})...")
    scan_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_scan_batch_shape)(str(bp), index_key)
        for bp in batch_paths
    )

    batch_meta  = []
    total_n_obs = 0
    total_nnz   = 0
    n_vars      = None
    has_scgpt   = None

    for i, (n_obs_b, nnz_b, n_vars_b, has_scgpt_b, err) in enumerate(scan_results):
        if err is not None:
            raise RuntimeError(f"{batch_paths[i].name} scan failed: {err}")
        if n_vars is None:
            n_vars = n_vars_b
        else:
            assert n_vars_b == n_vars, \
                f"{batch_paths[i].name}: n_vars={n_vars_b} != {n_vars}"
        if has_scgpt is None:
            has_scgpt = has_scgpt_b
        batch_meta.append({
            "path":    batch_paths[i],
            "n_obs":   n_obs_b,
            "nnz":     nnz_b,
            "obs_ptr": total_n_obs,
            "nnz_ptr": total_nnz,
        })
        total_n_obs += n_obs_b
        total_nnz   += nnz_b

    del scan_results
    gc.collect()
    log.info(
        f"total_obs={total_n_obs:,} | n_vars={n_vars:,} | "
        f"total_nnz={total_nnz:,} | has_scgpt={has_scgpt}"
    )

    # ----------------------------------------------------------
    # 3. parallel obs + X_scGPT collection
    #    read-only — each worker opens its own file handle
    # ----------------------------------------------------------
    log.info(f"Collecting obs + X_scGPT (n_jobs={n_jobs})...")
    obs_emb_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_read_batch_obs_emb)(str(meta["path"]), has_scgpt)
        for meta in batch_meta
    )

    obs_frames = []
    emb_parts  = [] if has_scgpt else None

    for i, (obs, emb, err) in enumerate(obs_emb_results):
        if err is not None:
            raise RuntimeError(f"{batch_meta[i]['path'].name} obs/emb failed: {err}")
        obs_frames.append(obs)
        if has_scgpt:
            emb_parts.append(emb)

    del obs_emb_results
    gc.collect()

    obs_all = pd.concat(obs_frames, axis=0)
    del obs_frames
    gc.collect()
    log.info(f"obs collected: {len(obs_all):,} × {len(obs_all.columns)}")

    if has_scgpt:
        emb_all = np.vstack(emb_parts)
        del emb_parts
        gc.collect()
        log.info(f"X_scGPT stacked: {emb_all.shape} | {emb_all.nbytes/1e9:.2f} GB")
        guard.check_available(
            needed_gb = emb_all.nbytes / 1e9,
            label     = "X_scGPT in RAM"
        )

    # ----------------------------------------------------------
    # 4. write skeleton h5ad (obs + var + empty X)
    # ----------------------------------------------------------
    log.info("Writing skeleton...")
    guard.check_available(needed_gb=1.0, label="skeleton")
    os.makedirs(os.path.dirname(path_output) or ".", exist_ok=True)

    skeleton = ad.AnnData(
        X   = sp.csr_matrix((total_n_obs, n_vars), dtype=np.float32),
        obs = obs_all,
        var = var_clean,
    )
    skeleton.write_h5ad(path_output, compression=None)
    del skeleton, obs_all, var_clean
    gc.collect()
    log.info("Skeleton written")

    # ----------------------------------------------------------
    # 5. pre-allocate X datasets in output
    # ----------------------------------------------------------
    log.info("Pre-allocating X...")
    with h5py.File(path_output, "r+") as f:
        del f["X"]
        xg = f.create_group("X")
        xg.attrs["encoding-type"]    = "csr_matrix"
        xg.attrs["encoding-version"] = "0.1.0"
        xg.attrs["shape"]            = np.array(
            [total_n_obs, n_vars], dtype=np.int64
        )
        xg.create_dataset("data",    shape=(total_nnz,),       dtype=np.float32)
        xg.create_dataset("indices", shape=(total_nnz,),       dtype=np.int32)
        xg.create_dataset("indptr",  shape=(total_n_obs + 1,), dtype=np.int64)
        xg["indptr"][0] = 0
    log.info("X pre-allocated")

    # ----------------------------------------------------------
    # 6. copy X batch by batch — sequential (single output file)
    # ----------------------------------------------------------
    log.info("Copying X...")
    for i, meta in enumerate(batch_meta):
        with h5py.File(meta["path"], "r") as src, \
            h5py.File(path_output,  "r+") as out:
            op, nnz_ptr = meta["obs_ptr"], meta["nnz_ptr"]
            nb, nz      = meta["n_obs"],   meta["nnz"]

            out["X"]["data"]   [nnz_ptr:nnz_ptr+nz]   = src["X"]["data"][:]
            out["X"]["indices"][nnz_ptr:nnz_ptr+nz]    = src["X"]["indices"][:]

            # ── int64 cast is mandatory ────────────────────────
            # Batch indptr is int32 (small batches fit in int32).
            # nnz_ptr grows past int32 max (~2.1 B) after ~18 batches.
            # Without explicit int64 cast, numpy may keep int32 arithmetic
            # → overflow → non-monotonic indptr in the merged file.
            src_iptr = np.array(src["X"]["indptr"][1:], dtype=np.int64)
            out["X"]["indptr"][op+1:op+1+nb] = src_iptr + np.int64(nnz_ptr)

        guard.kill_if_critical()
        if i % 10 == 0:
            log.info(f"  X [{i+1}/{len(batch_paths)}]")

    log.info("X copied")

    # ── step 7: inject X_scGPT with correct anndata encoding ──
    if has_scgpt:
        log.info(f"Injecting X_scGPT: {emb_all.shape}...")
        with h5py.File(path_output, "r+") as f:
            # create obsm group with anndata encoding attrs if absent
            if "obsm" not in f:
                grp = f.create_group("obsm")
                grp.attrs["encoding-type"]    = "dict"
                grp.attrs["encoding-version"] = "0.1.0"

            obsm_grp = f["obsm"]

            # ensure group itself has encoding (skeleton may have written
            # a plain group without attrs)
            if "encoding-type" not in obsm_grp.attrs:
                obsm_grp.attrs["encoding-type"]    = "dict"
                obsm_grp.attrs["encoding-version"] = "0.1.0"

            if "X_scGPT" in obsm_grp:
                del obsm_grp["X_scGPT"]

            ds = obsm_grp.create_dataset(
                "X_scGPT",
                data             = emb_all,
                compression      = "gzip",
                compression_opts = 1,
            )
            # dataset-level encoding so anndata reads it as ndarray
            ds.attrs["encoding-type"]    = "array"
            ds.attrs["encoding-version"] = "0.2.0"

        del emb_all
        gc.collect()
        log.info("X_scGPT injected")

    # ----------------------------------------------------------
    # 8. verify
    # ----------------------------------------------------------
    with h5py.File(path_output, "r") as f:
        shape  = tuple(f["X"].attrs["shape"])
        obsm_k = list(f["obsm"].keys()) if "obsm" in f else []
    log.info(
        f"Merge complete | X={shape} | obsm={obsm_k} | "
        f"{Path(path_output).stat().st_size/1e9:.2f} GB"
    )
    guard.collect()


# ============================================================
# ENTRY POINT
# ============================================================

# %%
guard = MemoryGuard(
    min_free_gb      = 10.0,
    warn_pct         = 0.75,
    kill_pct         = 0.80,
    monitor_interval = 60.0,
    kill_interval    = 5.0,
)

import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--config", default="/home/kyungan/scripts/BioTox/bio/configs/lincs_ext.json")
_p.add_argument("--run-keys", nargs="*", default=None)
_args, _ = _p.parse_known_args()

config = _args.config

with open(config) as fr:
    lincs = json.load(fr)

assert "runs" in lincs
run_keys = _args.run_keys or list(lincs["runs"].keys())
log.info(f"Runs: {run_keys}")

# %%
for run_key in run_keys:
    if run_key not in lincs["runs"]:
        log.warning(f"[{run_key}] not in config — skipping")
        continue

    run_cfg     = lincs["runs"][run_key]
    batch_dir   = run_cfg["batch_filter_dir"]
    path_output = run_cfg["path_output"].replace(".h5ad", "_filter.h5ad")
    n_jobs      = int(run_cfg["n_jobs"])

    if not Path(batch_dir).exists():
        log.warning(f"[{run_key}] batch_filter_dir not found: {batch_dir} — skipping")
        continue

    batch_paths = sorted(Path(batch_dir).glob("batch_*.h5ad"))
    if not batch_paths:
        log.warning(f"[{run_key}] no batch files in {batch_dir} — skipping")
        continue

    if Path(path_output).exists():
        log.info(f"[{run_key}] output exists — skipping: {path_output}")
        continue

    log.info(
        f"\n{'='*60}\n"
        f"Merge: {run_key}\n"
        f"  batch_dir:   {batch_dir} ({len(batch_paths)} files)\n"
        f"  path_output: {path_output}\n"
        f"  n_jobs:      {n_jobs}\n"
        f"{'='*60}"
    )

    with guard:
        merge_batches(
            batch_dir   = batch_dir,
            path_output = path_output,
            n_jobs      = n_jobs,
            guard       = guard,
        )

log.info("All merges complete.")
# %%