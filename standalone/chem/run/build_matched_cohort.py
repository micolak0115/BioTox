# publication/build_matched_cohort.py
"""
Phase 1 -- per-cell cohort construction + LINCS-union scaffold-excluded chem
fine-tuning fold.

Builds, independently per cell line (no crosscell intersection -- see
METHODS.md Phase 1 for why), the maximal LINCS-x-Tox21 overlap cohort at
6h/8-12uM, and a chemical-model fine-tuning pool of Tox21 compounds that are
Bemis-Murcko-scaffold-disjoint from everything LINCS covers for these cells
(not just from the Tox21-labeled subset), so Stage-1 chemical model
selection/fine-tuning never sees anything scaffold-related to what any
cell's cohort will later evaluate on.

The publication README and methods-aligned provenance archive document the
design rationale.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chem.splitter import DataSplit, generate_scaffold, scaffold_split_df  # noqa: E402

AGGREGATE_H5AD = "/data/kyungan/LINCS/lincs_merge_chemical_filter_select_align_aggregate.h5ad"
COMPOUNDINFO_PATH = "/data/kyungan/LINCS/compoundinfo_beta.txt"
TOX21_SMILES_PATH = "/data/kyungan/pretrain-gnns/dataset/tox21/raw/tox21_smiles.csv"
DEFAULT_OUT_DIR = (
    Path(__file__).parent / "_run_output" / "matched_split_fresh_v1"
)

CELL_IDS = ["HA1E", "HEPG2", "HT29", "MCF7"]
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
DOSE_WINDOW = (8.0, 12.0)
PERT_TIME = 6.0
COHORT_FRAC_TRAIN = 0.7
COHORT_FRAC_VALID = 0.3

# Fine-tuning-pool (1,732-compound, LINCS-unmatched, scaffold-disjoint) split.
# Frozen once for the whole study -- this is Stage 1's chemical-prior
# construction split, NOT cross-validated. train: candidate fitting.
# valid: within-candidate early stopping/checkpoint selection only.
# test: Stage-1 development-selection partition used to rank candidates for
# the task-specific K=4 arithmetic mean. It is not reused for the matched
# Stage-2 evaluation and is not presented as an unbiased post-selection test.
FINETUNE_FRAC_TRAIN = 0.6
FINETUNE_FRAC_VALID = 0.2
FINETUNE_FRAC_TEST = 0.2


def canonicalize(smiles) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def ik_skeleton(ik) -> str | None:
    if pd.isna(ik) or not isinstance(ik, str) or len(ik) == 0:
        return None
    return ik.split("-")[0] if "-" in ik else ik


def load_filtered_obs() -> pd.DataFrame:
    adata = ad.read_h5ad(AGGREGATE_H5AD, backed="r")
    obs = adata.obs
    ptime = pd.to_numeric(obs["pert_time"], errors="coerce")
    dose = pd.to_numeric(obs["pert_dose"], errors="coerce")
    mask = (
        (ptime == PERT_TIME)
        & (dose >= DOSE_WINDOW[0]) & (dose <= DOSE_WINDOW[1])
        & (obs["pert_type"].astype(str) == "trt_cp")
        & (obs["cell_id"].astype(str).isin(CELL_IDS))
    )
    return obs.loc[mask].copy()


def build_per_cell_cohorts(obs, pertid_to_ik, tox21_indexed):
    per_cell_cohort = {}
    union_iks = set()
    for cell in CELL_IDS:
        pids = set(obs.loc[obs["cell_id"].astype(str) == cell, "pert_id"].astype(str).unique())
        iks = {ik_skeleton(pertid_to_ik.get(p)) for p in pids}
        iks.discard(None)
        cohort_iks = iks & set(tox21_indexed.index)
        union_iks |= cohort_iks

        rows = []
        for ik in cohort_iks:
            row = tox21_indexed.loc[ik]
            smi_c = canonicalize(row["smiles_canon"])
            if smi_c is None:
                continue
            rec = {"ik": ik, "smiles_canon": smi_c}
            for t in TOX21_TASKS:
                rec[t] = row[t]
            rows.append(rec)
        cohort_df = (
            pd.DataFrame(rows)
            .sort_values("ik", kind="stable")
            .reset_index(drop=True)
        )
        per_cell_cohort[cell] = cohort_df
        print(f"[cohort] {cell}: {len(cohort_df)} compounds")
    return per_cell_cohort, union_iks


def write_cohort_split(
    cell: str,
    cohort_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    train_idx, valid_idx, test_idx = scaffold_split_df(
        cohort_df["smiles_canon"].tolist(),
        frac_train=COHORT_FRAC_TRAIN, frac_valid=COHORT_FRAC_VALID, frac_test=0.0,
    )
    split = DataSplit(
        train=cohort_df.iloc[train_idx].reset_index(drop=True),
        valid=cohort_df.iloc[valid_idx].reset_index(drop=True),
        # empty test -- final evaluation is nested CV (Phase 5), not this frozen split;
        # kept only so DataSplit's schema matches load_matched_split's expectations.
        test=cohort_df.iloc[test_idx].reset_index(drop=True),
    )
    out_path = out_dir / f"tox21_scaffold_df_split_{cell}_8to12uM_6h.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(split, f)
    print(f"[cohort] {cell}: wrote {out_path} (train={len(split.train)}, valid={len(split.valid)})")


def build_lincs_scaffold_union(obs, pertid_to_smiles, per_cell_cohort) -> set[str]:
    all_pids = set(obs["pert_id"].astype(str).unique())
    lincs_smiles = set()
    for p in all_pids:
        s = pertid_to_smiles.get(p)
        if isinstance(s, str) and s.strip():
            lincs_smiles.add(s)
    print(f"[union] LINCS pert_id: {len(all_pids)}, resolvable SMILES: {len(lincs_smiles)}")

    lincs_scaffolds = {generate_scaffold(s, include_chirality=True) for s in lincs_smiles}
    lincs_scaffolds.discard(None)
    print(f"[union] LINCS unique scaffolds (compoundinfo_beta.txt SMILES only): {len(lincs_scaffolds)}")

    # Belt-and-suspenders: a handful of compoundinfo_beta.txt SMILES fail to
    # parse (a few LINCS entries are malformed/blank), which would silently
    # under-populate the exclusion scaffold set even though those compounds
    # ARE confirmed LINCS-covered (they matched into a per-cell cohort via
    # InChIKey). Union in each cohort's own (already-canonicalized,
    # guaranteed-parseable) scaffolds directly so no cohort compound's
    # scaffold can ever be missing from the exclusion boundary.
    cohort_scaffolds_all = set()
    for cohort_df in per_cell_cohort.values():
        s = {generate_scaffold(smi, include_chirality=True) for smi in cohort_df["smiles_canon"]}
        s.discard(None)
        cohort_scaffolds_all |= s
    n_added = len(cohort_scaffolds_all - lincs_scaffolds)
    lincs_scaffolds |= cohort_scaffolds_all
    print(f"[union] +{n_added} scaffolds recovered from cohort SMILES that compoundinfo_beta.txt "
          f"couldn't parse -> LINCS union unique scaffolds: {len(lincs_scaffolds)}")
    return lincs_scaffolds


def build_chem_finetune_pool(lincs_scaffolds: set[str], per_cell_cohort: dict) -> pd.DataFrame:
    tox21_all = pd.read_csv(TOX21_SMILES_PATH)
    tox21_all["scaffold"] = tox21_all["smiles_canon"].apply(
        lambda s: generate_scaffold(s, include_chirality=True)
    )
    excluded_mask = tox21_all["scaffold"].isin(lincs_scaffolds)
    finetune_pool = tox21_all[~excluded_mask & tox21_all["scaffold"].notna()].reset_index(drop=True)
    print(f"[finetune] Tox21 total={len(tox21_all)}, excluded={int(excluded_mask.sum())}, "
          f"surviving fine-tune pool={len(finetune_pool)}")

    overlap = set(finetune_pool["scaffold"]) & lincs_scaffolds
    assert not overlap, f"{len(overlap)} scaffolds still overlap LINCS union after filtering"

    for cell, cohort_df in per_cell_cohort.items():
        cohort_scaffolds = {generate_scaffold(s, include_chirality=True) for s in cohort_df["smiles_canon"]}
        cohort_scaffolds.discard(None)
        overlap_cell = set(finetune_pool["scaffold"]) & cohort_scaffolds
        assert not overlap_cell, f"{len(overlap_cell)} scaffolds overlap {cell}'s cohort"

    return finetune_pool


def write_finetune_split(
    finetune_pool: pd.DataFrame,
    out_dir: Path,
) -> None:
    if "ik" not in finetune_pool.columns:
        # Downstream consumers (chem_stage1_per_task_offset.py's
        # load_finetune_split/load_cohort_union, bio/run/utils.py's
        # load_matched_split) all expect an "ik" column holding the
        # truncated InChIKey connectivity skeleton -- the same identifier
        # already used as "ik" in the per-cell cohort splits
        # (build_per_cell_cohorts) and as the "ik_skel" index here. Add it
        # rather than renaming "inchikey", so nothing upstream that reads
        # "inchikey" is affected.
        finetune_pool = finetune_pool.copy()
        finetune_pool["ik"] = finetune_pool["inchikey"].apply(ik_skeleton)
    train_idx, valid_idx, test_idx = scaffold_split_df(
        finetune_pool["smiles_canon"].tolist(),
        frac_train=FINETUNE_FRAC_TRAIN, frac_valid=FINETUNE_FRAC_VALID, frac_test=FINETUNE_FRAC_TEST,
    )
    ft_split = DataSplit(
        train=finetune_pool.iloc[train_idx].reset_index(drop=True),
        valid=finetune_pool.iloc[valid_idx].reset_index(drop=True),
        test=finetune_pool.iloc[test_idx].reset_index(drop=True),
    )
    ft_out = out_dir / "tox21_scaffold_df_split_chem_finetune_pool.pkl"
    with open(ft_out, "wb") as f:
        pickle.dump(ft_split, f)
    print(f"[finetune] wrote {ft_out} (train={len(ft_split.train)}, valid={len(ft_split.valid)}, "
          f"test={len(ft_split.test)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the four matched LINCS-Tox21 cohorts and the scaffold-"
            "excluded Stage-1 chemical training split."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="New output directory. Existing paths are never overwritten.",
    )
    return parser.parse_args()


def main(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing cohort output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    compound_info = pd.read_csv(COMPOUNDINFO_PATH, sep="\t")
    pertid_to_ik = dict(zip(compound_info["pert_id"], compound_info["inchi_key"]))
    pertid_to_smiles = dict(zip(compound_info["pert_id"], compound_info["canonical_smiles"]))

    tox21 = pd.read_csv(TOX21_SMILES_PATH)
    tox21["ik_skel"] = tox21["inchikey"].apply(ik_skeleton)
    tox21_indexed = tox21.dropna(subset=["ik_skel"]).drop_duplicates("ik_skel").set_index("ik_skel")

    obs = load_filtered_obs()

    per_cell_cohort, _union_iks = build_per_cell_cohorts(obs, pertid_to_ik, tox21_indexed)
    for cell, cohort_df in per_cell_cohort.items():
        write_cohort_split(cell, cohort_df, output_dir)

    lincs_scaffolds = build_lincs_scaffold_union(obs, pertid_to_smiles, per_cell_cohort)
    finetune_pool = build_chem_finetune_pool(lincs_scaffolds, per_cell_cohort)
    write_finetune_split(finetune_pool, output_dir)

    print("[done] Phase 1 complete.")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args.output_dir)
