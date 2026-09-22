# %%
import os
import gc
import sys
import logging
from typing import Optional, Tuple
from joblib import Parallel, delayed
import torch
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import json 
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from bio.efaas.encoder import scGPT

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

# %%
def _embed_batch(
    path_batch: str,
    gpu_id:     int,
) -> Tuple[str, Optional[str]]:
    import h5py
    import tempfile

    worker_log = _get_logger(f"worker.embed_{Path(path_batch).stem}")

    with h5py.File(path_batch, "r") as f:
        if "obsm" in f and "X_scGPT" in f["obsm"]:
            worker_log.info(f"{Path(path_batch).name}: X_scGPT exists — skipping")
            return path_batch, None

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        adata = sc.read_h5ad(path_batch)
        scgpt = scGPT(adata)

        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
            tmp_path = tmp.name

        scgpt.embed(tmp_path)

        tmp_adata = sc.read_h5ad(tmp_path, backed="r")
        if "X_scGPT" in tmp_adata.obsm:
            X_scGPT = np.array(tmp_adata.obsm["X_scGPT"], dtype=np.float32)
        else:
            X_scGPT = tmp_adata.X
            if sp.issparse(X_scGPT):
                X_scGPT = X_scGPT.toarray()
            X_scGPT = X_scGPT.astype(np.float32)
        tmp_adata.file.close()
        os.remove(tmp_path)

        with h5py.File(path_batch, "r+") as f:
            if "obsm" not in f:
                f.create_group("obsm")
            if "X_scGPT" in f["obsm"]:
                del f["obsm"]["X_scGPT"]
            f["obsm"].create_dataset(
                "X_scGPT", data=X_scGPT,
                compression="gzip", compression_opts=1,
            )

        worker_log.info(f"{Path(path_batch).name}: X_scGPT {X_scGPT.shape}")
        del adata, scgpt, X_scGPT
        gc.collect()
        return path_batch, None

    except Exception as e:
        import traceback
        err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        worker_log.error(f"{Path(path_batch).name} FAILED: {err_msg}")
        return path_batch, err_msg

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
    n_jobs = int(run_cfg["n_jobs"])
    batch_dir = run_cfg["batch_dir"]

    batch_paths = sorted(Path(batch_dir).glob("batch_*.h5ad"))
    n_gpus = torch.cuda.device_count() - 1
    gpu_ids = [i % max(n_gpus, 1) for i in range(len(batch_paths))]

    log.info(
        f"Step 2: embedding {len(batch_paths)} batches | "
        f"n_gpus={n_gpus} | n_jobs={n_jobs}"
    )
    embed_results = Parallel(
        n_jobs=n_jobs, backend="loky", verbose=10
    )(
        delayed(_embed_batch)(str(bp), gpu_ids[i])
        for i, bp in enumerate(batch_paths)
    )
    failed_e  = [(p, err) for p, err in embed_results if err is not None]
    success_e = [p         for p, err in embed_results if err is None]
    log.info(f"Step 2: {len(success_e)} succeeded | {len(failed_e)} failed")
    if failed_e:
        for p, err in failed_e:
            log.error(f"  {Path(p).name}: {err[:200]}")
# %%
