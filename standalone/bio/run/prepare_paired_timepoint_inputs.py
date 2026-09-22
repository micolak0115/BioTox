"""Build leakage-safe paired 6 h/24 h BioTox inputs.

The frozen Stage-1 chemical ensemble excluded the 6-hour LINCS scaffold
universe during training. A full 24-hour cohort contains additional scaffolds
that can overlap Stage-1 development data. This sensitivity analysis therefore
uses, independently within each cell line, only compounds measured at both
6 and 24 hours in the same 8-12 uM concentration window.

The paired cohort is shared by both time points. Separate gene-feature tables
are generated from the 6-hour and 24-hour X_centered profiles so downstream
repeated scaffold partitions can be identical across time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from rdkit import Chem

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from chem.splitter import DataSplit, generate_scaffold, scaffold_split_df


AGGREGATE_H5AD = Path(
    "/data/kyungan/LINCS/"
    "lincs_merge_chemical_filter_select_align_aggregate.h5ad"
)
COMPOUNDINFO_PATH = Path("/data/kyungan/LINCS/compoundinfo_beta.txt")
TOX21_SMILES_PATH = Path(
    "/data/kyungan/pretrain-gnns/dataset/tox21/raw/tox21_smiles.csv"
)
CEVICHE_CTRP_PATH = Path(
    "/data/kyungan/Cell-death-signatures/models/ctrp.csv"
)
STAGE1_SPLIT_PATH = (
    HERE
    / "_run_output"
    / "matched_split_rebuilt"
    / "tox21_scaffold_df_split_chem_finetune_pool.pkl"
)
DEFAULT_OUTPUT_ROOT = (
    HERE / "_run_output" / "paired_timepoint_inputs_6h24h_8to12uM_v1"
)

CELL_IDS = ("HA1E", "HEPG2", "HT29", "MCF7")
TOX21_TASKS = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)
TIMEPOINTS_HOURS = (6.0, 24.0)
DOSE_WINDOW_UM = (8.0, 12.0)
COHORT_FRAC_TRAIN = 0.7
COHORT_FRAC_VALID = 0.3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(smiles: object) -> str | None:
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(molecule) if molecule else None
    except Exception:
        return None


def ik_skeleton(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.split("-")[0]


def aggregate_replicates(values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 1:
        return values[0]
    return np.median(values, axis=0)


def load_stage1_scaffolds(path: Path) -> set[str]:
    with path.open("rb") as handle:
        split = pickle.load(handle)
    frames = [split.train, split.valid, split.test]
    smiles = pd.concat(frames, ignore_index=True)["smiles_canon"].dropna()
    scaffolds = {
        generate_scaffold(value, include_chirality=True) for value in smiles
    }
    scaffolds.discard(None)
    return scaffolds


def filtered_obs(obs: pd.DataFrame, pert_time: float) -> pd.DataFrame:
    times = pd.to_numeric(obs["pert_time"], errors="coerce")
    doses = pd.to_numeric(obs["pert_dose"], errors="coerce")
    mask = (
        times.eq(pert_time)
        & doses.between(DOSE_WINDOW_UM[0], DOSE_WINDOW_UM[1], inclusive="both")
        & obs["pert_type"].astype(str).eq("trt_cp")
        & obs["cell_id"].astype(str).isin(CELL_IDS)
    )
    return obs.loc[mask].copy()


def tox21_iks_for_cell(
    obs_time: pd.DataFrame,
    cell: str,
    pertid_to_ik: dict[str, object],
    tox21_iks: set[str],
) -> set[str]:
    pert_ids = (
        obs_time.loc[obs_time["cell_id"].astype(str).eq(cell), "pert_id"]
        .astype(str)
        .unique()
    )
    identifiers = {ik_skeleton(pertid_to_ik.get(value)) for value in pert_ids}
    identifiers.discard(None)
    return identifiers & tox21_iks


def build_paired_cohort(
    tox21_indexed: pd.DataFrame,
    identifiers: set[str],
) -> pd.DataFrame:
    rows = []
    for identifier in sorted(identifiers):
        source = tox21_indexed.loc[identifier]
        smiles = canonicalize(source["smiles_canon"])
        if smiles is None:
            continue
        row = {"ik": identifier, "smiles_canon": smiles}
        row.update({task: source[task] for task in TOX21_TASKS})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ik", kind="stable").reset_index(
        drop=True
    )


def write_paired_split(
    cell: str,
    cohort: pd.DataFrame,
    cohort_dir: Path,
) -> Path:
    train_idx, valid_idx, test_idx = scaffold_split_df(
        cohort["smiles_canon"].tolist(),
        frac_train=COHORT_FRAC_TRAIN,
        frac_valid=COHORT_FRAC_VALID,
        frac_test=0.0,
    )
    split = DataSplit(
        train=cohort.iloc[train_idx].reset_index(drop=True),
        valid=cohort.iloc[valid_idx].reset_index(drop=True),
        test=cohort.iloc[test_idx].reset_index(drop=True),
    )
    path = (
        cohort_dir
        / f"tox21_scaffold_df_split_{cell}_8to12uM_paired6h24h.pkl"
    )
    with path.open("xb") as handle:
        pickle.dump(split, handle)
    return path


def load_landmark_definition(adata) -> tuple[np.ndarray, np.ndarray]:
    landmark_mask = (adata.var["pr_is_lm"].astype(int) == 1).to_numpy()
    symbols = (
        adata.var.loc[landmark_mask, "pr_gene_symbol"].astype(str).to_numpy()
    )
    seen: set[str] = set()
    keep = np.array(
        [symbol not in seen and not seen.add(symbol) for symbol in symbols],
        dtype=bool,
    )
    positions = np.where(landmark_mask)[0][keep]
    deduplicated_mask = np.zeros(adata.n_vars, dtype=bool)
    deduplicated_mask[positions] = True
    return deduplicated_mask, symbols[keep]


# CEViChE (CTRP cell-death-axis) residualization is deprecated; the pipeline
# now runs unadjusted-only. Kept commented out (not removed) for provenance.
# def load_ceviche_weights(gene_symbols: np.ndarray) -> np.ndarray:
#     table = pd.read_csv(CEVICHE_CTRP_PATH)
#     score_by_gene = {
#         str(gene).upper(): float(coefficient)
#         for gene, coefficient in zip(
#             table["pr_gene_symbol"], table["coefficient"]
#         )
#         if pd.notna(gene) and pd.notna(coefficient)
#     }
#     return np.array(
#         [score_by_gene.get(str(gene).upper(), 0.0) for gene in gene_symbols],
#         dtype=np.float64,
#     )
#
#
# def residualize(
#     raw: pd.DataFrame,
#     gene_symbols: np.ndarray,
#     weights: np.ndarray,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     values = raw[list(gene_symbols)].to_numpy(dtype=np.float64)
#     denominator = float(np.dot(weights, weights) + 1e-8)
#     score = values @ weights / denominator
#     residual = values - score[:, None] * weights[None, :]
#     residual_frame = pd.concat(
#         [
#             raw[["ik"]].reset_index(drop=True),
#             pd.DataFrame(residual, columns=list(gene_symbols)),
#         ],
#         axis=1,
#     )
#     score_frame = raw[["ik"]].copy()
#     score_frame["ceviche_score"] = score
#     return residual_frame, score_frame


def build_features(
    *,
    adata,
    obs_time: pd.DataFrame,
    cell: str,
    cohort_iks: set[str],
    pertid_to_ik: dict[str, object],
    landmark_mask: np.ndarray,
    gene_symbols: np.ndarray,
) -> pd.DataFrame:
    cell_obs = obs_time.loc[
        obs_time["cell_id"].astype(str).eq(cell)
    ]
    positions = adata.obs.index.get_indexer(cell_obs.index)
    positions_by_ik: dict[str, list[int]] = {}
    for position, pert_id in zip(
        positions, cell_obs["pert_id"].astype(str)
    ):
        identifier = ik_skeleton(pertid_to_ik.get(pert_id))
        if identifier in cohort_iks:
            positions_by_ik.setdefault(identifier, []).append(int(position))

    rows = []
    matrix = adata.obsm["X_centered"]
    for identifier in sorted(cohort_iks):
        compound_positions = positions_by_ik.get(identifier, [])
        if not compound_positions:
            raise RuntimeError(
                f"{cell}: paired compound {identifier} lacks an in-window row"
            )
        values = matrix[np.asarray(compound_positions)][:, landmark_mask]
        pooled = aggregate_replicates(np.asarray(values, dtype=np.float64))
        rows.append([identifier, *pooled.tolist()])

    frame = pd.DataFrame(rows, columns=["ik", *gene_symbols.tolist()])
    feature_values = frame.drop(columns="ik").to_numpy(dtype=np.float64)
    if frame["ik"].duplicated().any():
        raise RuntimeError(f"{cell}: duplicate feature rows")
    if set(frame["ik"]) != cohort_iks:
        raise RuntimeError(f"{cell}: feature/cohort identifier mismatch")
    if not np.isfinite(feature_values).all():
        raise RuntimeError(f"{cell}: non-finite feature values")
    return frame


def task_counts(cohort: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts = {}
    for task in TOX21_TASKS:
        labels = pd.to_numeric(cohort[task], errors="coerce").dropna()
        counts[task] = {
            "n_labeled": int(len(labels)),
            "n_positive": int(labels.eq(1).sum()),
            "n_negative": int(labels.eq(0).sum()),
        }
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired 6h/24h, 8-12 uM cohorts and separate X_centered "
            "feature tables without modifying existing publication outputs."
        )
    )
    parser.add_argument("--adata-path", type=Path, default=AGGREGATE_H5AD)
    parser.add_argument(
        "--stage1-split-path", type=Path, default=STAGE1_SPLIT_PATH
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing timepoint input root: "
            f"{output_root}"
        )
    for path in (
        args.adata_path,
        COMPOUNDINFO_PATH,
        TOX21_SMILES_PATH,
        CEVICHE_CTRP_PATH,
        args.stage1_split_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    cohort_dir = output_root / "cohort"
    feature_dirs = {
        time: output_root / f"gene_features_{int(time)}h"
        for time in TIMEPOINTS_HOURS
    }
    cohort_dir.mkdir(parents=True, exist_ok=False)
    for directory in feature_dirs.values():
        directory.mkdir(parents=False, exist_ok=False)

    compound_info = pd.read_csv(COMPOUNDINFO_PATH, sep="\t")
    pertid_to_ik = dict(
        zip(
            compound_info["pert_id"].astype(str),
            compound_info["inchi_key"],
        )
    )
    tox21 = pd.read_csv(TOX21_SMILES_PATH)
    tox21["ik_skeleton"] = tox21["inchikey"].map(ik_skeleton)
    tox21_indexed = (
        tox21.dropna(subset=["ik_skeleton"])
        .drop_duplicates("ik_skeleton")
        .set_index("ik_skeleton")
    )
    tox21_iks = set(tox21_indexed.index)

    adata = ad.read_h5ad(args.adata_path, backed="r")
    if "X_centered" not in adata.obsm:
        raise KeyError("AnnData must contain obsm['X_centered']")
    obs_by_time = {
        time: filtered_obs(adata.obs, time) for time in TIMEPOINTS_HOURS
    }
    landmark_mask, gene_symbols = load_landmark_definition(adata)
    # CEViChE weighting is deprecated; unadjusted-only below.
    # ceviche_weights = load_ceviche_weights(gene_symbols)
    stage1_scaffolds = load_stage1_scaffolds(args.stage1_split_path)

    manifest_cells: dict[str, object] = {}
    for cell in CELL_IDS:
        identifiers_by_time = {
            time: tox21_iks_for_cell(
                obs_by_time[time],
                cell,
                pertid_to_ik,
                tox21_iks,
            )
            for time in TIMEPOINTS_HOURS
        }
        paired_iks = set.intersection(*identifiers_by_time.values())
        cohort = build_paired_cohort(tox21_indexed, paired_iks)
        paired_iks = set(cohort["ik"])
        if not paired_iks:
            raise RuntimeError(f"{cell}: paired cohort is empty")

        paired_scaffolds = {
            generate_scaffold(smiles, include_chirality=True)
            for smiles in cohort["smiles_canon"]
        }
        paired_scaffolds.discard(None)
        overlap = paired_scaffolds & stage1_scaffolds
        if overlap:
            raise RuntimeError(
                f"{cell}: {len(overlap)} paired scaffolds overlap the frozen "
                "Stage-1 development split"
            )

        split_path = write_paired_split(cell, cohort, cohort_dir)
        feature_checksums: dict[str, dict[str, str]] = {}
        for time in TIMEPOINTS_HOURS:
            raw = build_features(
                adata=adata,
                obs_time=obs_by_time[time],
                cell=cell,
                cohort_iks=paired_iks,
                pertid_to_ik=pertid_to_ik,
                landmark_mask=landmark_mask,
                gene_symbols=gene_symbols,
            )
            # residualized, score = residualize(
            #     raw, gene_symbols, ceviche_weights
            # )
            output_dir = feature_dirs[time]
            raw_path = output_dir / f"{cell}_gene_features_raw.csv"
            # residualized_path = (
            #     output_dir / f"{cell}_gene_features_residualized.csv"
            # )
            # score_path = output_dir / f"{cell}_ceviche_score.csv"
            raw.to_csv(raw_path, index=False)
            # residualized.to_csv(residualized_path, index=False)
            # score.to_csv(score_path, index=False)
            feature_checksums[f"{int(time)}h"] = {
                raw_path.name: file_sha256(raw_path),
                # residualized_path.name: file_sha256(residualized_path),
                # score_path.name: file_sha256(score_path),
            }

        manifest_cells[cell] = {
            "n_6h": len(identifiers_by_time[6.0]),
            "n_24h": len(identifiers_by_time[24.0]),
            "n_paired": len(paired_iks),
            "n_6h_only": len(
                identifiers_by_time[6.0] - identifiers_by_time[24.0]
            ),
            "n_24h_only": len(
                identifiers_by_time[24.0] - identifiers_by_time[6.0]
            ),
            "paired_stage1_scaffold_overlap": 0,
            "cohort_file": str(split_path),
            "cohort_file_sha256": file_sha256(split_path),
            "task_counts": task_counts(cohort),
            "feature_files_sha256": feature_checksums,
        }
        print(
            f"[paired-time] {cell}: 6h={len(identifiers_by_time[6.0])}, "
            f"24h={len(identifiers_by_time[24.0])}, paired={len(paired_iks)}"
        )

    common_feature_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "paired_timepoint_sensitivity",
        "adata_path": str(args.adata_path.resolve()),
        "adata_sha256": file_sha256(args.adata_path.resolve()),
        "obsm_key": "X_centered",
        "gene_scope": "L1000 landmark genes, duplicate symbols removed",
        "n_gene_features": int(landmark_mask.sum()),
        "replicate_aggregation": "median within compound/cell/time/8-12uM",
        "dose_window_um": list(DOSE_WINDOW_UM),
        "paired_timepoints_hours": list(TIMEPOINTS_HOURS),
        "paired_cohort_definition": (
            "per-cell Tox21 compounds observed at both 6h and 24h"
        ),
        "stage1_split_path": str(args.stage1_split_path.resolve()),
        "stage1_split_sha256": file_sha256(args.stage1_split_path.resolve()),
        "stage1_scaffold_overlap_required": 0,
        "cells": manifest_cells,
    }
    for time, directory in feature_dirs.items():
        time_manifest = {
            **common_feature_manifest,
            "feature_time_hours": time,
            "feature_dir": str(directory),
            "cohort_dir": str(cohort_dir),
        }
        (directory / "FEATURE_MANIFEST.json").write_text(
            json.dumps(time_manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    (cohort_dir / "COHORT_MANIFEST.json").write_text(
        json.dumps(common_feature_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "RUN_COMPLETE.json").write_text(
        json.dumps(
            {
                **common_feature_manifest,
                "output_root": str(output_root),
                "cohort_dir": str(cohort_dir),
                "feature_dirs": {
                    f"{int(time)}h": str(directory)
                    for time, directory in feature_dirs.items()
                },
                "complete": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[paired-time] complete: {output_root}")


if __name__ == "__main__":
    main()
