# publication/build_gene_features.py
"""
Phase 2 -- gene-level features from LINCS L1000, per cell.

Reuses `obsm['X_centered']` directly as the DMSO-fold-change feature source
(already log2(trt) - log2(matched DMSO control), per-(cell_id,plate_id),
replicate-aggregated -- see stage2/preprocess/align.py::get_embeddings; no
recomputation from the pre-aggregate needed). Restricts to the 978 landmark
genes via this file's own var['pr_is_lm']. Dose window 8-12uM is a flat
inclusion band (any row in-window counts as "the ~10uM condition" for that
compound) -- multiple in-window replicate rows for one compound are
aggregated the same way stage2/preprocess/aggregate.py does (median, its
default method).

Emits, per cell:
  - {cell}_gene_features_raw.csv        -- pooled X_centered, 978 genes.
      Feeds the "standardized" variant: z-scoring happens LIVE inside the
      nested-CV loop (Phase 5), fit on outer-train only, never here -- doing
      it here would leak outer-test statistics into outer-train's scaling.
  - {cell}_gene_features_residualized.csv -- CeViChe (CTRP) cell-death-axis
      regressed out. This IS safe to precompute once here (not inside the CV
      loop): the projection vector w is an external, fixed coefficient table
      never fit on any compound in this study, so it carries no leakage risk
      regardless of which fold a compound later lands in.
  - {cell}_ceviche_score.csv             -- the per-compound projection
      scalar itself (useful covariate for Phase 9's diagnostic).

See METHODS.md Phase 2 for the full rationale.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1.splitter import DataSplit

COMPOUNDINFO_PATH = "/data/kyungan/LINCS/compoundinfo_beta.txt"
AGGREGATE_H5AD = "/data/kyungan/LINCS/lincs_merge_chemical_filter_select_align_aggregate.h5ad"
CEVICHE_CTRP_PATH = "/data/kyungan/Cell-death-signatures/models/ctrp.csv"
COHORT_DIR = Path(__file__).parent / "_run_output" / "matched_split_rebuilt"
DEFAULT_OUT_DIR = Path(__file__).parent / "_run_output" / "gene_features_fresh_v1"

CELL_IDS = ["HA1E", "HEPG2", "HT29", "MCF7"]
DOSE_WINDOW = (8.0, 12.0)
PERT_TIME = 6.0
REPLICATE_AGG_METHOD = "median"  # matches stage2/preprocess/aggregate.py's default


def ik_skeleton(ik) -> str | None:
    if pd.isna(ik) or not isinstance(ik, str) or len(ik) == 0:
        return None
    return ik.split("-")[0] if "-" in ik else ik


def load_cohort_iks(cell: str, cohort_dir: Path) -> set[str]:
    path = cohort_dir / f"tox21_scaffold_df_split_{cell}_8to12uM_6h.pkl"
    with open(path, "rb") as f:
        split = pickle.load(f)
    frames = [split.train, split.valid, split.test]
    iks = set()
    for df in frames:
        if len(df):
            iks |= set(df["ik"].astype(str).unique())
    return iks


def aggregate_replicates(X: np.ndarray, method: str = REPLICATE_AGG_METHOD) -> np.ndarray:
    if X.shape[0] == 1:
        return X[0]
    if method == "median":
        return np.median(X, axis=0)
    if method == "mean":
        return X.mean(axis=0)
    raise ValueError(f"unsupported method {method!r}")


def build_raw_features_for_cell(cell: str, adata, landmark_mask: np.ndarray,
                                 gene_symbols: np.ndarray, pertid_to_ik: dict,
                                 cohort_iks: set[str]) -> pd.DataFrame:
    obs = adata.obs
    ptime = pd.to_numeric(obs["pert_time"], errors="coerce")
    dose = pd.to_numeric(obs["pert_dose"], errors="coerce")
    mask = (
        (obs["cell_id"].astype(str) == cell)
        & (ptime == PERT_TIME)
        & (dose >= DOSE_WINDOW[0]) & (dose <= DOSE_WINDOW[1])
        & (obs["pert_type"].astype(str) == "trt_cp")
    )
    sub_obs = obs.loc[mask]
    row_positions = np.where(mask.to_numpy())[0]

    ik_by_pos = {}
    for pos, pert_id in zip(row_positions, sub_obs["pert_id"].astype(str)):
        ik = ik_skeleton(pertid_to_ik.get(pert_id))
        if ik is not None and ik in cohort_iks:
            ik_by_pos[pos] = ik

    ik_to_positions: dict[str, list[int]] = {}
    for pos, ik in ik_by_pos.items():
        ik_to_positions.setdefault(ik, []).append(pos)

    X_centered = adata.obsm["X_centered"]
    rows = []
    for ik, positions in ik_to_positions.items():
        X_sub = X_centered[np.array(positions)][:, landmark_mask]
        pooled = aggregate_replicates(X_sub)
        rows.append([ik] + pooled.tolist())

    cols = ["ik"] + list(gene_symbols)
    df = pd.DataFrame(rows, columns=cols)
    missing = cohort_iks - set(df["ik"])
    extra = set(df["ik"]) - cohort_iks
    duplicates = df["ik"].duplicated(keep=False)
    if missing:
        raise RuntimeError(
            f"{cell}: {len(missing)}/{len(cohort_iks)} cohort compounds have no "
            "in-window LINCS row; Phase 2 cannot silently shrink the cohort"
        )
    if extra:
        raise RuntimeError(f"{cell}: generated {len(extra)} compounds outside the Phase-1 cohort")
    if duplicates.any():
        raise RuntimeError(f"{cell}: duplicate compound rows in gene feature table")
    values = df.drop(columns="ik").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{cell}: non-finite values in generated gene features")
    return df


# CEViChE (CTRP cell-death-axis) residualization is deprecated; the pipeline
# now runs unadjusted-only. Kept commented out (not removed) for provenance.
# def load_ceviche_weight(gene_symbols: np.ndarray) -> np.ndarray:
#     scores = pd.read_csv(CEVICHE_CTRP_PATH)
#     score_map = {
#         str(g).upper(): float(c)
#         for g, c in zip(scores["pr_gene_symbol"], scores["coefficient"])
#         if pd.notna(g) and pd.notna(c)
#     }
#     w = np.array([score_map.get(str(g).upper(), 0.0) for g in gene_symbols], dtype=np.float64)
#     matched = int(np.count_nonzero(w))
#     print(f"[ceviche] matched {matched}/{len(gene_symbols)} landmark genes to CTRP coefficient table")
#     return w
#
#
# def residualize(raw_df: pd.DataFrame, gene_symbols: np.ndarray, w: np.ndarray):
#     X = raw_df[list(gene_symbols)].to_numpy(dtype=np.float64)
#     denom = float(np.dot(w, w) + 1e-8)
#     score = X @ w / denom
#     projection = score[:, None] * w[None, :]
#     resid = X - projection
#     resid_df = pd.concat(
#         [raw_df[["ik"]].reset_index(drop=True), pd.DataFrame(resid, columns=list(gene_symbols))],
#         axis=1,
#     )
#     score_df = raw_df[["ik"]].copy()
#     score_df["ceviche_score"] = score
#     return resid_df, score_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build standardized-input and CeViChe-residualized landmark-gene "
            "features for each matched cell-line cohort."
        )
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=COHORT_DIR,
        help="Directory produced by build_matched_cohort.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="New output directory. Existing paths are never overwritten.",
    )
    return parser.parse_args()


def main(cohort_dir: Path, output_dir: Path) -> None:
    cohort_dir = cohort_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing gene-feature output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    compound_info = pd.read_csv(COMPOUNDINFO_PATH, sep="\t")
    pertid_to_ik = dict(zip(compound_info["pert_id"], compound_info["inchi_key"]))

    adata = ad.read_h5ad(AGGREGATE_H5AD, backed="r")
    landmark_mask = (adata.var["pr_is_lm"].astype(int) == 1).to_numpy()
    gene_symbols_raw = adata.var.loc[landmark_mask, "pr_gene_symbol"].astype(str).to_numpy()
    # de-dup gene symbols (keep first) so downstream wide-table columns are unique
    _seen = set()
    keep_mask = np.array([not (s in _seen or _seen.add(s)) for s in gene_symbols_raw])
    if not keep_mask.all():
        print(f"[gene-features] dropping {int((~keep_mask).sum())} duplicate-symbol landmark genes")
    # re-derive landmark_mask restricted to the deduped subset, preserving order
    landmark_positions = np.where(landmark_mask)[0][keep_mask]
    landmark_mask = np.zeros(adata.n_vars, dtype=bool)
    landmark_mask[landmark_positions] = True
    gene_symbols = gene_symbols_raw[keep_mask]
    print(f"[gene-features] {len(gene_symbols)} landmark genes after dedup")

    # CEViChE weighting/residualization is deprecated; unadjusted-only below.
    # w = load_ceviche_weight(gene_symbols)

    for cell in CELL_IDS:
        cohort_iks = load_cohort_iks(cell, cohort_dir)
        raw_df = build_raw_features_for_cell(cell, adata, landmark_mask, gene_symbols, pertid_to_ik, cohort_iks)
        raw_path = output_dir / f"{cell}_gene_features_raw.csv"
        raw_df.to_csv(raw_path, index=False)
        print(f"[gene-features] {cell}: wrote {raw_path} ({len(raw_df)} compounds)")

        # resid_df, score_df = residualize(raw_df, gene_symbols, w)
        # resid_path = output_dir / f"{cell}_gene_features_residualized.csv"
        # score_path = output_dir / f"{cell}_ceviche_score.csv"
        # resid_df.to_csv(resid_path, index=False)
        # score_df.to_csv(score_path, index=False)
        # print(f"[gene-features] {cell}: wrote {resid_path} and {score_path}")

    print("[done] Phase 2 complete.")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args.cohort_dir, cli_args.output_dir)
