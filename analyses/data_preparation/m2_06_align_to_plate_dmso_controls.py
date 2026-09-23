# %%
# align.py

# %%
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
# from TVN import TVN
import json
import anndata as ad


def get_embeddings(
    adata: AnnData,
    obsm_key: str = "X_scGPT",
    n_pcs: int = None,
    alpha: float = 0.5,
):
    """
    Hierarchical biological normalization (Bendidi et al. 2024).

    Embeddings produced
    -------------------
    x_raw         : raw embedding, no processing
    x_centered    : per-(cell_id, plate_id) control-mean subtracted
    x_tvn         : TVN-normalized, fit per cell line on raw controls
                    z = ((x - mu_c)/sigma_c) @ T_c
    x_tvn_delta   : x_tvn minus PLATE-matched TVN-space control mean
                    (removes cell-line basal state + residual plate shift)

    Returns
    -------
    x_raw, x_centered, x_tvn, x_tvn_delta : np.ndarray (N, D)
    cell_baseline : dict {cell_id: beta_c (D,)}  — cell-line baseline in
                    TVN space, kept for downstream toxicity context
    """
    x_bio = (
        adata.obsm[obsm_key].toarray()
        if sp.issparse(adata.obsm[obsm_key])
        else np.asarray(adata.obsm[obsm_key]).copy()
    )
    x_bio = x_bio.astype(np.float32)
    n_samples, n_features = x_bio.shape

    obs       = adata.obs
    cell_arr  = obs["cell_id"].values
    plate_arr = obs["plate_id"].values
    ctrl_arr  = obs["control"].values == 1

    # ----------------------------------------------------------
    # 1. Raw
    # ----------------------------------------------------------
    x_raw = x_bio.copy()

    # ----------------------------------------------------------
    # 2. Centering — subtract per-(cell_id, plate_id) control mean
    # ----------------------------------------------------------
    x_centered = x_bio.copy()
    for (cell_id, plate_id), _ in obs.groupby(["cell_id", "plate_id"], observed=True):
        grp_idx   = np.where((cell_arr == cell_id) & (plate_arr == plate_id))[0]
        ctrl_mask = ctrl_arr[grp_idx]
        if ctrl_mask.sum() == 0:
            continue
        mu_ctrl             = x_bio[grp_idx][ctrl_mask].mean(axis=0)
        x_centered[grp_idx] = x_bio[grp_idx] - mu_ctrl

    return x_raw, x_centered

def diagnose_batch(adata: AnnData, keys: list, batch_col="plate_id", cell_col="cell_id"):
    """Silhouette of each embedding vs batch and cell — lower batch = better."""
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    print(f"\n  Batch diagnostic (lower {batch_col}_sil = better correction):")
    for key in keys:
        emb = adata.obsm[key]
        npc = min(20, emb.shape[1], emb.shape[0] - 1)
        pcs = PCA(n_components=npc).fit_transform(emb)
        try:
            s_batch = silhouette_score(pcs, adata.obs[batch_col])
        except Exception:
            s_batch = float("nan")
        try:
            s_cell = silhouette_score(pcs, adata.obs[cell_col])
        except Exception:
            s_cell = float("nan")
        print(f"    {key:<18}  {batch_col}_sil={s_batch:+.3f}  {cell_col}_sil={s_cell:+.3f}")

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

N_PCS = None
ALPHA = 0.5

for run_key in run_keys:
    run_cfg  = lincs["runs"][run_key]
    src_path = run_cfg["path_output"].replace(".h5ad", "_filter_select.h5ad")
    dst_path = src_path.replace(".h5ad", "_align.h5ad")

    print(f"\n[{run_key}] Loading {src_path} ...")
    adata = ad.read_h5ad(src_path)

    if sp.issparse(adata.X):
        adata.obsm["X"] = adata.X.toarray().astype(np.float32)
    else:
        adata.obsm["X"] = np.asarray(adata.X, dtype=np.float32)
    del adata.X

    ctrl_vals = run_cfg["ctrl_vals"]

    if "control" not in adata.obs.columns:
        adata.obs["control"] = adata.obs["pert_type"].isin(ctrl_vals).astype(int)
        print(f"  added 'control' | ctrl_vals={ctrl_vals} | "
              f"n_ctrl={adata.obs['control'].sum():,}")

    if "plate_id" not in adata.obs.columns:
        adata.obs["plate_id"] = [
            "_".join(x.split(":")[0].split("_")[:3]) for x in adata.obs.index
        ]
        print(f"  added 'plate_id' | n_plates={adata.obs['plate_id'].nunique():,}")

    print(f"  obs cols   : {list(adata.obs.columns)}")
    print(f"  cell lines : {sorted(adata.obs['cell_id'].unique())}")
    print(f"  controls   : {adata.obs['control'].sum():,} / {len(adata.obs):,}")

    (x_raw, x_centered) = get_embeddings(
        adata, obsm_key="X", n_pcs=N_PCS, alpha=ALPHA
    )
    adata.obsm["X_raw"]       = x_raw
    adata.obsm["X_centered"]  = x_centered

    if "X_scGPT" in adata.obsm:
        (z_raw, z_centered) = get_embeddings(
            adata, obsm_key="X_scGPT", n_pcs=N_PCS, alpha=ALPHA
        )
        adata.obsm["X_scGPT_raw"]       = z_raw
        adata.obsm["X_scGPT_centered"]  = z_centered
    else:
        print(f"  [WARN] X_scGPT not found — skipping scGPT alignment")


    adata.write_h5ad(dst_path)
    print(f"[{run_key}] written: {dst_path}")

# %%
