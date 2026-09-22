#!/usr/bin/env python3
"""Compound- and scaffold-level analysis of significant residual gains.

This script is deliberately downstream-only: it reads archived out-of-fold
predictions and never refits the frozen chemical prior or Stage-2 models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score
from statsmodels.stats.multitest import multipletests


HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
DEFAULT_PUBLICATION_ROOT = HERE / "_run_output" / "publication"
DEFAULT_OUTPUT = DEFAULT_PUBLICATION_ROOT / "compound_scaffold_residual_analysis_v1"
TABLE_S7 = (
    DEFAULT_PUBLICATION_ROOT
    / "tables"
    / "07_complete_table_s7"
    / "table_s7_nominal_residual_gains_complete.csv"
)
OOF_ROOT_24H = (
    HERE
    / "_run_output"
    / "nested_cv_k4_calibrated_paired24h_repeated20x5_innerseed_v3"
)
PAIRED_MCF7_SPLIT = (
    HERE
    / "_run_output"
    / "paired_timepoint_inputs_6h24h_8to12uM_v1"
    / "cohort"
    / "tox21_scaffold_df_split_MCF7_8to12uM_paired6h24h.pkl"
)
TOX21_RAW = Path("/data/kyungan/pretrain-gnns/dataset/tox21/raw/tox21_ik.csv")

PRIMARY_SIMILARITY_CUTOFF = 0.60
SIMILARITY_SENSITIVITY = (0.50, 0.60, 0.70)
N_BOOTSTRAP = 2000
RNG_SEED = 20260806

PALETTE = {
    "stable_rescued": "#b84a4a",
    "consensus_rescued": "#e7a35c",
    "combined_true_positive": "#6b98b8",
    "unresolved_active": "#b6b6b6",
}
GROUP_LABELS = {
    "stable_rescued": "Stable rescued active",
    "consensus_rescued": "Consensus rescued active",
    "combined_true_positive": "Other combined true positive",
    "unresolved_active": "Unresolved active",
}

DESCRIPTORS = {
    "Molecular weight": Descriptors.MolWt,
    "cLogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "H-bond acceptors": Lipinski.NumHAcceptors,
    "H-bond donors": Lipinski.NumHDonors,
    "Rotatable bonds": Lipinski.NumRotatableBonds,
    "Ring count": Lipinski.RingCount,
    "Aromatic rings": Lipinski.NumAromaticRings,
    "Fraction Csp3": Lipinski.FractionCSP3,
    "Formal charge": lambda mol: sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
    "Heavy atoms": Lipinski.HeavyAtomCount,
    "QED": QED.qed,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze compounds and scaffolds underlying significant residual gains."
    )
    parser.add_argument("--table-s7", type=Path, default=TABLE_S7)
    parser.add_argument("--oof-root-24h", type=Path, default=OOF_ROOT_24H)
    parser.add_argument("--paired-mcf7-split", type=Path, default=PAIRED_MCF7_SPLIT)
    parser.add_argument("--tox21-raw", type=Path, default=TOX21_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--stable-low-rate", type=float, default=0.20)
    parser.add_argument("--stable-high-rate", type=float, default=0.80)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def require_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))


def prepare_output(path: Path) -> tuple[Path, Path]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    tables = path / "tables"
    figures = path / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    return tables, figures


def load_split_union(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        split = pickle.load(handle)
    frames = [getattr(split, name) for name in ("train", "valid", "test")]
    cohort = pd.concat(frames, ignore_index=True)
    if cohort["ik"].duplicated().any():
        duplicated = cohort.loc[cohort["ik"].duplicated(), "ik"].tolist()
        raise ValueError(f"Duplicate structures in cohort split: {duplicated[:5]}")
    return cohort


def eligible_contexts(table_s7: Path, q_threshold: float) -> pd.DataFrame:
    table = pd.read_csv(table_s7)
    eligible = table.loc[
        (table["delta_auprc"] > 0) & (table["bh_q_value"] < q_threshold)
    ].copy()
    if eligible.empty:
        raise ValueError("No endpoint-context met the multiplicity-controlled criterion.")
    eligible["analysis_tier"] = "confirmatory_context"
    return eligible.sort_values(["family", "task", "lincs_cell_line", "lincs_exposure_h"])


def oof_path_for_context(row: pd.Series, root_24h: Path) -> Path:
    if int(row["lincs_exposure_h"]) != 24 or row["lincs_cell_line"] != "MCF7":
        raise ValueError(
            "Version 1 has audited inputs only for confirmatory MCF7 24-h contexts; "
            f"received {row['task']} {row['lincs_cell_line']} {row['lincs_exposure_h']} h."
        )
    return root_24h / (
        "repeated_nested_cv_oof_predictions_residualized_"
        f"{row['task']}__{row['lincs_cell_line']}.csv"
    )


def bh_adjust(frame: pd.DataFrame, p_col: str, q_col: str) -> pd.DataFrame:
    frame = frame.copy()
    frame[q_col] = np.nan
    mask = frame[p_col].notna()
    if mask.any():
        frame.loc[mask, q_col] = multipletests(
            frame.loc[mask, p_col].to_numpy(float), method="fdr_bh"
        )[1]
    return frame


def consensus_predictions(
    oof: pd.DataFrame,
    threshold: float,
    stable_low: float,
    stable_high: float,
) -> pd.DataFrame:
    required = {
        "ik",
        "y",
        "repeat_id",
        "p_chem_only",
        "p_chem_bio",
        "outer_fold_id",
        "outer_split_signature",
    }
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF file missing columns: {sorted(missing)}")

    y_counts = oof.groupby("ik")["y"].nunique()
    if (y_counts != 1).any():
        raise ValueError("Outcome labels are inconsistent across repeated OOF predictions.")
    per_repeat_counts = oof.groupby(["ik", "repeat_id"]).size()
    if (per_repeat_counts != 1).any():
        raise ValueError("Each compound must occur exactly once per repeat.")

    oof = oof.copy()
    fold_keys = ["repeat_id", "outer_fold_id"]
    oof["rank_pct_chem"] = oof.groupby(fold_keys)["p_chem_only"].rank(
        method="average", pct=True
    )
    oof["rank_pct_bio"] = oof.groupby(fold_keys)["p_chem_bio"].rank(
        method="average", pct=True
    )
    oof["delta_probability"] = oof["p_chem_bio"] - oof["p_chem_only"]
    oof["delta_rank_percentile"] = oof["rank_pct_bio"] - oof["rank_pct_chem"]
    oof["chem_positive"] = oof["p_chem_only"] >= threshold
    oof["bio_positive"] = oof["p_chem_bio"] >= threshold
    oof["bio_increased"] = oof["delta_probability"] > 0

    for column in ("ap_contribution_chem", "ap_contribution_bio"):
        oof[column] = 0.0
    folds_per_repeat = oof.groupby("repeat_id")["outer_fold_id"].nunique()
    if folds_per_repeat.nunique() != 1:
        raise ValueError("Each repeat must contain the same number of evaluable outer folds.")
    n_outer_folds = int(folds_per_repeat.iloc[0])
    if n_outer_folds < 2:
        raise ValueError("At least two outer folds are required for fold-level AP decomposition.")

    for (repeat_id, outer_fold_id), indices in oof.groupby(fold_keys).groups.items():
        repeat = oof.loc[indices]
        y = repeat["y"].to_numpy(int)
        n_positive = int(y.sum())
        if n_positive == 0:
            raise ValueError(
                f"Repeat {repeat_id}, outer fold {outer_fold_id} has no active compounds."
            )
        for score_column, output_column in (
            ("p_chem_only", "ap_contribution_chem"),
            ("p_chem_bio", "ap_contribution_bio"),
        ):
            scores = repeat[score_column].to_numpy(float)
            order = np.argsort(-scores, kind="mergesort")
            y_sorted = y[order]
            precision = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
            # Scale each fold by K so compound-level contributions reconstruct
            # the reported mean outer-fold AUPRC, not a pooled OOF AUPRC.
            contribution_sorted = np.where(
                y_sorted == 1,
                precision / n_positive / n_outer_folds,
                0.0,
            )
            contribution = np.empty_like(contribution_sorted, dtype=float)
            contribution[order] = contribution_sorted
            oof.loc[indices, output_column] = contribution
            observed = float(contribution.sum())
            expected = float(average_precision_score(y, scores)) / n_outer_folds
            if not np.isclose(observed, expected, atol=1e-12):
                raise AssertionError(
                    "AP decomposition failed for "
                    f"repeat {repeat_id}, outer fold {outer_fold_id}, {score_column}: "
                    f"{observed} != {expected}"
                )
    oof["delta_ap_contribution"] = (
        oof["ap_contribution_bio"] - oof["ap_contribution_chem"]
    )
    oof["ap_contribution_increased"] = oof["delta_ap_contribution"] > 0

    consensus = (
        oof.groupby("ik", as_index=False)
        .agg(
            y=("y", "first"),
            n_repeats=("repeat_id", "nunique"),
            mean_p_chem=("p_chem_only", "mean"),
            sd_p_chem=("p_chem_only", "std"),
            mean_p_bio=("p_chem_bio", "mean"),
            sd_p_bio=("p_chem_bio", "std"),
            mean_delta_probability=("delta_probability", "mean"),
            sd_delta_probability=("delta_probability", "std"),
            mean_rank_pct_chem=("rank_pct_chem", "mean"),
            mean_rank_pct_bio=("rank_pct_bio", "mean"),
            mean_delta_rank_percentile=("delta_rank_percentile", "mean"),
            chem_positive_rate=("chem_positive", "mean"),
            bio_positive_rate=("bio_positive", "mean"),
            bio_increase_rate=("bio_increased", "mean"),
            mean_ap_contribution_chem=("ap_contribution_chem", "mean"),
            mean_ap_contribution_bio=("ap_contribution_bio", "mean"),
            mean_delta_ap_contribution=("delta_ap_contribution", "mean"),
            sd_delta_ap_contribution=("delta_ap_contribution", "std"),
            ap_contribution_increase_rate=("ap_contribution_increased", "mean"),
        )
    )
    expected_repeats = int(oof["repeat_id"].nunique())
    consensus["complete_repeat_coverage"] = consensus["n_repeats"] == expected_repeats
    if not consensus["complete_repeat_coverage"].all():
        raise ValueError("Some compounds lack predictions from one or more repeats.")

    active = consensus["y"] == 1
    consensus["consensus_rescued"] = (
        active
        & (consensus["mean_p_chem"] < threshold)
        & (consensus["mean_p_bio"] >= threshold)
    )
    consensus["stable_rescued"] = (
        active
        & (consensus["chem_positive_rate"] <= stable_low)
        & (consensus["bio_positive_rate"] >= stable_high)
    )
    consensus["combined_true_positive"] = active & (consensus["mean_p_bio"] >= threshold)
    consensus["unresolved_active"] = active & (consensus["mean_p_bio"] < threshold)

    consensus["compound_group"] = "inactive"
    consensus.loc[consensus["unresolved_active"], "compound_group"] = "unresolved_active"
    consensus.loc[
        consensus["combined_true_positive"], "compound_group"
    ] = "combined_true_positive"
    consensus.loc[consensus["consensus_rescued"], "compound_group"] = "consensus_rescued"
    consensus.loc[consensus["stable_rescued"], "compound_group"] = "stable_rescued"
    return consensus


def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def scaffold_strings(mol) -> tuple[str, str, str]:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else ""
    if scaffold.GetNumAtoms():
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        generic_smiles = Chem.MolToSmiles(generic, canonical=True)
        key = scaffold_smiles
    else:
        generic_smiles = ""
        key = "ACYCLIC::" + Chem.MolToSmiles(mol, canonical=True)
    return scaffold_smiles, generic_smiles, key


def calculate_properties(structures: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    records = []
    mols = {}
    for row in structures.itertuples(index=False):
        mol = mol_from_smiles(row.smiles_canon)
        record = {
            "ik": row.ik,
            "mol_id": getattr(row, "mol_id", np.nan),
            "smiles_canon": row.smiles_canon,
            "valid_structure": mol is not None,
        }
        if mol is None:
            records.append(record)
            continue
        mols[row.ik] = mol
        scaffold, generic, scaffold_key = scaffold_strings(mol)
        record.update(
            {
                "bemis_murcko_scaffold": scaffold,
                "generic_murcko_scaffold": generic,
                "scaffold_key": scaffold_key,
            }
        )
        for name, function in DESCRIPTORS.items():
            record[name] = float(function(mol))
        records.append(record)
    return pd.DataFrame(records), mols


def morgan_fingerprints(mols: dict[str, object]) -> dict[str, object]:
    return {
        ik: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        for ik, mol in mols.items()
    }


def butina_assignments(
    ids: list[str], fingerprints: dict[str, object], similarity_cutoff: float
) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame(columns=["ik", "butina_cluster", "cluster_size", "cluster_medoid"])
    fps = [fingerprints[ik] for ik in ids]
    distances = []
    for i in range(1, len(fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend(1.0 - value for value in similarities)
    clusters = Butina.ClusterData(
        distances,
        len(fps),
        1.0 - similarity_cutoff,
        isDistData=True,
        reordering=True,
    )
    rows = []
    for cluster_id, members in enumerate(clusters, start=1):
        medoid = ids[members[0]]
        for member in members:
            rows.append(
                {
                    "ik": ids[member],
                    "butina_cluster": cluster_id,
                    "cluster_size": len(members),
                    "cluster_medoid": medoid,
                    "similarity_cutoff": similarity_cutoff,
                }
            )
    return pd.DataFrame(rows)


def pairwise_similarity_table(
    active: pd.DataFrame, fingerprints: dict[str, object]
) -> pd.DataFrame:
    ids = active["ik"].tolist()
    metadata = active.set_index("ik")
    rows = []
    for i in range(1, len(ids)):
        ik_i = ids[i]
        sims = DataStructs.BulkTanimotoSimilarity(
            fingerprints[ik_i], [fingerprints[ik] for ik in ids[:i]]
        )
        for j, similarity in enumerate(sims):
            ik_j = ids[j]
            rows.append(
                {
                    "ik_1": ik_i,
                    "ik_2": ik_j,
                    "mol_id_1": metadata.loc[ik_i, "mol_id"],
                    "mol_id_2": metadata.loc[ik_j, "mol_id"],
                    "smiles_canon_1": metadata.loc[ik_i, "smiles_canon"],
                    "smiles_canon_2": metadata.loc[ik_j, "smiles_canon"],
                    "scaffold_key_1": metadata.loc[ik_i, "scaffold_key"],
                    "scaffold_key_2": metadata.loc[ik_j, "scaffold_key"],
                    "tanimoto": float(similarity),
                    "rescued_1": bool(metadata.loc[ik_i, "consensus_rescued"]),
                    "rescued_2": bool(metadata.loc[ik_j, "consensus_rescued"]),
                    "stable_rescued_1": bool(metadata.loc[ik_i, "stable_rescued"]),
                    "stable_rescued_2": bool(metadata.loc[ik_j, "stable_rescued"]),
                    "delta_probability_1": float(
                        metadata.loc[ik_i, "mean_delta_probability"]
                    ),
                    "delta_probability_2": float(
                        metadata.loc[ik_j, "mean_delta_probability"]
                    ),
                    "delta_ap_contribution_1": float(
                        metadata.loc[ik_i, "mean_delta_ap_contribution"]
                    ),
                    "delta_ap_contribution_2": float(
                        metadata.loc[ik_j, "mean_delta_ap_contribution"]
                    ),
                    "compound_group_1": metadata.loc[ik_i, "compound_group"],
                    "compound_group_2": metadata.loc[ik_j, "compound_group"],
                }
            )
    return pd.DataFrame(rows)


def nearest_neighbor_features(active: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in active.itertuples(index=False):
        ik = row.ik
        subset = pairs.loc[(pairs["ik_1"] == ik) | (pairs["ik_2"] == ik)].copy()
        subset["other_ik"] = np.where(subset["ik_1"] == ik, subset["ik_2"], subset["ik_1"])
        rescued_map = active.set_index("ik")["consensus_rescued"]
        subset["other_rescued"] = subset["other_ik"].map(rescued_map)
        rows.append(
            {
                "ik": ik,
                "max_similarity_active": subset["tanimoto"].max() if len(subset) else np.nan,
                "max_similarity_rescued": (
                    subset.loc[subset["other_rescued"], "tanimoto"].max()
                    if subset["other_rescued"].any()
                    else np.nan
                ),
                "max_similarity_nonrescued": (
                    subset.loc[~subset["other_rescued"], "tanimoto"].max()
                    if (~subset["other_rescued"]).any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


def scaffold_bootstrap_ci(
    frame: pd.DataFrame,
    value_col: str,
    group_col: str,
    scaffold_col: str,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    frame = frame.loc[frame[scaffold_col].notna(), [scaffold_col, value_col, group_col]].copy()
    scaffolds = frame[scaffold_col].unique()
    values = []
    rescued_values = []
    other_values = []
    for scaffold in scaffolds:
        group = frame.loc[frame[scaffold_col] == scaffold]
        rescued_values.append(group.loc[group[group_col], value_col].to_numpy(float))
        other_values.append(group.loc[~group[group_col], value_col].to_numpy(float))
    for _ in range(n_bootstrap):
        multiplicities = np.bincount(
            rng.integers(0, len(scaffolds), size=len(scaffolds)),
            minlength=len(scaffolds),
        )
        x_parts = [
            np.tile(rescued_values[index], count)
            for index, count in enumerate(multiplicities)
            if count and len(rescued_values[index])
        ]
        y_parts = [
            np.tile(other_values[index], count)
            for index, count in enumerate(multiplicities)
            if count and len(other_values[index])
        ]
        x = np.concatenate(x_parts) if x_parts else np.array([], dtype=float)
        y = np.concatenate(y_parts) if y_parts else np.array([], dtype=float)
        if len(x) and len(y):
            values.append(cliffs_delta(x, y))
    if not values:
        return np.nan, np.nan, 0
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), len(values)


def descriptor_tests(active: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rescued = active["consensus_rescued"]
    for descriptor in DESCRIPTORS:
        x = active.loc[rescued, descriptor].dropna().to_numpy(float)
        y = active.loc[~rescued, descriptor].dropna().to_numpy(float)
        if len(x) and len(y):
            test = mannwhitneyu(x, y, alternative="two-sided")
            delta = cliffs_delta(x, y)
        else:
            test = None
            delta = np.nan
        rho, rho_p = spearmanr(
            active[descriptor], active["mean_delta_probability"], nan_policy="omit"
        )
        rank_rho, rank_p = spearmanr(
            active[descriptor], active["mean_delta_rank_percentile"], nan_policy="omit"
        )
        ap_rho, ap_p = spearmanr(
            active[descriptor], active["mean_delta_ap_contribution"], nan_policy="omit"
        )
        ci_low, ci_high, n_valid_boot = scaffold_bootstrap_ci(
            active,
            descriptor,
            "consensus_rescued",
            "scaffold_key",
        )
        rows.append(
            {
                "descriptor": descriptor,
                "n_rescued": len(x),
                "n_other_active": len(y),
                "rescued_median": float(np.median(x)) if len(x) else np.nan,
                "other_active_median": float(np.median(y)) if len(y) else np.nan,
                "cliffs_delta": delta,
                "scaffold_bootstrap_ci_low": ci_low,
                "scaffold_bootstrap_ci_high": ci_high,
                "valid_scaffold_bootstrap_replicates": n_valid_boot,
                "mannwhitney_p": float(test.pvalue) if test is not None else np.nan,
                "spearman_rho_continuous_gain": float(rho),
                "spearman_p": float(rho_p),
                "spearman_rho_rank_gain": float(rank_rho),
                "spearman_rank_p": float(rank_p),
                "spearman_rho_ap_contribution_gain": float(ap_rho),
                "spearman_ap_contribution_p": float(ap_p),
            }
        )
    result = pd.DataFrame(rows)
    result = bh_adjust(result, "mannwhitney_p", "mannwhitney_q")
    result = bh_adjust(result, "spearman_p", "spearman_q")
    result = bh_adjust(result, "spearman_rank_p", "spearman_rank_q")
    result = bh_adjust(
        result, "spearman_ap_contribution_p", "spearman_ap_contribution_q"
    )
    return result


def fragment_smiles(mol, center: int, radius: int) -> str:
    if radius == 0:
        return Chem.MolFragmentToSmiles(
            mol, atomsToUse=[center], rootedAtAtom=center, canonical=True
        )
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center))
    atoms = {center}
    for bond_idx in bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=sorted(atoms),
        bondsToUse=bonds,
        rootedAtAtom=center,
        canonical=True,
    )


def sparse_morgan_motifs(mols: dict[str, object]) -> dict[str, set[str]]:
    presence = {}
    for ik, mol in mols.items():
        bit_info = {}
        AllChem.GetMorganFingerprint(mol, radius=2, bitInfo=bit_info)
        fragments = set()
        for environments in bit_info.values():
            for center, radius in environments:
                if radius < 1:
                    continue
                bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center))
                atoms = {center}
                for bond_idx in bonds:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atoms.add(bond.GetBeginAtomIdx())
                    atoms.add(bond.GetEndAtomIdx())
                if len(atoms) < 3:
                    continue
                fragment = fragment_smiles(mol, center, radius)
                fragments.add(fragment)
        presence[ik] = fragments
    return presence


def enrichment_tests(
    active: pd.DataFrame,
    item_map: dict[str, set],
    item_name: str,
    item_label_map: dict | None = None,
    min_rescued: int = 3,
    min_total: int = 5,
) -> pd.DataFrame:
    rescued_ids = set(active.loc[active["consensus_rescued"], "ik"])
    other_ids = set(active.loc[~active["consensus_rescued"], "ik"])
    all_items = set().union(*(item_map.get(ik, set()) for ik in active["ik"]))
    rows = []
    for item in all_items:
        rescued_present = sum(item in item_map.get(ik, set()) for ik in rescued_ids)
        other_present = sum(item in item_map.get(ik, set()) for ik in other_ids)
        total_present = rescued_present + other_present
        if rescued_present < min_rescued or total_present < min_total:
            continue
        table = [
            [rescued_present, len(rescued_ids) - rescued_present],
            [other_present, len(other_ids) - other_present],
        ]
        odds_ratio, p_value = fisher_exact(table, alternative="greater")
        rows.append(
            {
                item_name: item,
                f"{item_name}_label": item_label_map.get(item, str(item))
                if item_label_map
                else str(item),
                "rescued_present": rescued_present,
                "rescued_total": len(rescued_ids),
                "other_active_present": other_present,
                "other_active_total": len(other_ids),
                "odds_ratio": float(odds_ratio),
                "fisher_p": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                item_name,
                f"{item_name}_label",
                "rescued_present",
                "rescued_total",
                "other_active_present",
                "other_active_total",
                "odds_ratio",
                "fisher_p",
                "fisher_q",
            ]
        )
    return bh_adjust(result.sort_values("fisher_p"), "fisher_p", "fisher_q")


def scaffold_summary(active: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, group in (
        ("all_active", active),
        ("consensus_rescued", active.loc[active["consensus_rescued"]]),
        ("other_active", active.loc[~active["consensus_rescued"]]),
    ):
        counts = group["scaffold_key"].value_counts()
        n = len(group)
        probabilities = counts.to_numpy(float) / n if n else np.array([])
        entropy = -float(np.sum(probabilities * np.log(probabilities))) if n else np.nan
        normalized_entropy = (
            entropy / math.log(len(counts)) if len(counts) > 1 else 0.0 if len(counts) == 1 else np.nan
        )
        rows.append(
            {
                "group": group_name,
                "n_compounds": n,
                "n_unique_scaffolds": int(len(counts)),
                "unique_scaffold_fraction": len(counts) / n if n else np.nan,
                "singleton_scaffold_fraction": (
                    int((counts == 1).sum()) / len(counts) if len(counts) else np.nan
                ),
                "normalized_shannon_entropy": normalized_entropy,
                "top5_scaffold_compound_fraction": (
                    counts.head(5).sum() / n if n else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def similarity_summary(active: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    categories = {
        "rescued_within": pairs["rescued_1"] & pairs["rescued_2"],
        "other_active_within": (~pairs["rescued_1"]) & (~pairs["rescued_2"]),
        "rescued_to_other": pairs["rescued_1"] ^ pairs["rescued_2"],
    }
    rows = []
    for name, mask in categories.items():
        values = pairs.loc[mask, "tanimoto"].to_numpy(float)
        rows.append(
            {
                "pair_group": name,
                "n_pairs": len(values),
                "mean_tanimoto": float(np.mean(values)) if len(values) else np.nan,
                "median_tanimoto": float(np.median(values)) if len(values) else np.nan,
                "q25_tanimoto": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                "q75_tanimoto": float(np.quantile(values, 0.75)) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_overview(all_active: pd.DataFrame, figures: Path, dpi: int) -> None:
    contexts = list(all_active["context_id"].drop_duplicates())
    fig, axes = plt.subplots(len(contexts), 2, figsize=(8.1, 3.7 * len(contexts)), squeeze=False)
    for row_idx, context in enumerate(contexts):
        frame = all_active.loc[all_active["context_id"] == context].copy()
        frame = frame.sort_values("mean_delta_ap_contribution").reset_index(drop=True)
        colors = frame["compound_group"].map(PALETTE)
        ax = axes[row_idx, 0]
        ax.bar(
            np.arange(len(frame)),
            frame["mean_delta_ap_contribution"],
            color=colors,
            width=1.0,
        )
        ax.axhline(0, color="#666666", linestyle=":", linewidth=1.2)
        ax.set_ylabel(r"Mean per-compound $\Delta$AUPRC contribution", fontsize=11)
        ax.set_xlabel("Active compounds ordered by residual contribution", fontsize=11)
        ax.set_title(context.replace("__", " | "), fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=9)

        ax = axes[row_idx, 1]
        for group, group_frame in frame.groupby("compound_group"):
            ax.scatter(
                group_frame["mean_p_chem"],
                group_frame["mean_p_bio"],
                s=28,
                alpha=0.82,
                color=PALETTE[group],
                edgecolor="black",
                linewidth=0.35,
                label=GROUP_LABELS[group],
            )
        ax.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1.2)
        ax.axvline(0.5, color="#aaaaaa", linewidth=0.8)
        ax.axhline(0.5, color="#aaaaaa", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean Chem.-only probability", fontsize=11)
        ax.set_ylabel("Mean Chem. + transcriptomics probability", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.92,
            fontsize=8,
            loc="lower right",
        )

    sns.despine(fig)
    fig.tight_layout(pad=1.2, w_pad=2.2, h_pad=2.0)
    for axis, label in zip(axes.flat, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        box = axis.get_position()
        fig.text(
            box.x0 + 0.005,
            box.y1 + 0.006,
            label,
            fontsize=15,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"compound_residual_overview.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_descriptor_effects(descriptor_results: pd.DataFrame, figures: Path, dpi: int) -> None:
    contexts = list(descriptor_results["context_id"].drop_duplicates())
    fig, axes = plt.subplots(1, len(contexts), figsize=(8.1, 7.0), sharey=True, squeeze=False)
    order = list(DESCRIPTORS)
    y = np.arange(len(order))
    for idx, context in enumerate(contexts):
        ax = axes[0, idx]
        frame = descriptor_results.loc[
            descriptor_results["context_id"] == context
        ].set_index("descriptor").loc[order]
        x = frame["cliffs_delta"].to_numpy(float)
        low = frame["scaffold_bootstrap_ci_low"].to_numpy(float)
        high = frame["scaffold_bootstrap_ci_high"].to_numpy(float)
        xerr = np.vstack([x - low, high - x])
        significant = frame["mannwhitney_q"].to_numpy(float) < 0.05
        colors = np.where(significant, "#b84a4a", "#7f99ad")
        ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#333333", elinewidth=1.3, capsize=3)
        ax.scatter(x, y, c=colors, s=35, edgecolor="black", linewidth=0.5, zorder=3)
        ax.axvline(0, color="#777777", linestyle=":", linewidth=1.1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_yticks(y, labels=order, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Cliff's delta\n(rescued vs other actives)", fontsize=11)
        ax.set_title(context.replace("__", " | "), fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.tick_params(axis="x", labelsize=9)
    axes[0, 0].set_ylabel("Physicochemical property", fontsize=11)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#b84a4a",
            markeredgecolor="black",
            markersize=7,
            label=r"BH $q<0.05$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#7f99ad",
            markeredgecolor="black",
            markersize=7,
            label=r"BH $q\geq0.05$",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        title="Rescued-versus-other contrast",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    sns.despine(fig)
    fig.tight_layout(pad=1.3, w_pad=2.0, rect=(0, 0.07, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"physicochemical_rescue_effects.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_structural_summary(
    all_active: pd.DataFrame,
    all_pairs: pd.DataFrame,
    scaffold_summaries: pd.DataFrame,
    figures: Path,
    dpi: int,
) -> None:
    contexts = list(all_active["context_id"].drop_duplicates())
    fig, axes = plt.subplots(len(contexts), 2, figsize=(8.1, 3.6 * len(contexts)), squeeze=False)
    for row_idx, context in enumerate(contexts):
        pair_frame = all_pairs.loc[all_pairs["context_id"] == context].copy()
        pair_frame["pair_group"] = np.select(
            [
                pair_frame["rescued_1"] & pair_frame["rescued_2"],
                pair_frame["rescued_1"] ^ pair_frame["rescued_2"],
            ],
            ["Rescued-rescued", "Rescued-other"],
            default="Other-other",
        )
        ax = axes[row_idx, 0]
        sns.boxplot(
            data=pair_frame,
            x="pair_group",
            y="tanimoto",
            order=["Rescued-rescued", "Rescued-other", "Other-other"],
            color="#cad7df",
            fliersize=0,
            linewidth=1.1,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Active-compound pair", fontsize=11)
        ax.set_ylabel("Morgan Tanimoto similarity", fontsize=11)
        ax.set_title(context.replace("__", " | "), fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelrotation=15, labelsize=8.5)
        ax.tick_params(axis="y", labelsize=9)

        scaffold = scaffold_summaries.loc[
            scaffold_summaries["context_id"] == context
        ].copy()
        scaffold = scaffold.loc[scaffold["group"].isin(["consensus_rescued", "other_active"])]
        scaffold["group_label"] = scaffold["group"].map(
            {"consensus_rescued": "Rescued", "other_active": "Other active"}
        )
        long = scaffold.melt(
            id_vars="group_label",
            value_vars=[
                "unique_scaffold_fraction",
                "singleton_scaffold_fraction",
                "normalized_shannon_entropy",
            ],
            var_name="metric",
            value_name="value",
        )
        long["metric"] = long["metric"].map(
            {
                "unique_scaffold_fraction": "Unique scaffold fraction",
                "singleton_scaffold_fraction": "Singleton scaffold fraction",
                "normalized_shannon_entropy": "Normalized scaffold entropy",
            }
        )
        ax = axes[row_idx, 1]
        sns.barplot(
            data=long,
            y="metric",
            x="value",
            hue="group_label",
            palette={"Rescued": "#b84a4a", "Other active": "#aeb7bd"},
            alpha=0.72,
            ax=ax,
        )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Scaffold diversity measure", fontsize=11)
        ax.set_ylabel("")
        ax.tick_params(labelsize=9)
        ax.legend(
            title="Active group",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.92,
            fontsize=9,
            title_fontsize=9,
            loc="lower right",
        )
    sns.despine(fig)
    fig.tight_layout(pad=1.2, w_pad=2.4, h_pad=2.0)
    for axis, label in zip(axes.flat, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        box = axis.get_position()
        fig.text(
            box.x0 - 0.045,
            box.y1 + 0.006,
            label,
            fontsize=15,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"structural_neighborhood_summary.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_rescued_atlas(
    active: pd.DataFrame,
    mols: dict[str, object],
    figures: Path,
    context_id: str,
) -> None:
    rescued = active.loc[active["consensus_rescued"]].copy()
    rescued = rescued.sort_values(
        ["stable_rescued", "mean_delta_ap_contribution", "mean_delta_rank_percentile"],
        ascending=[False, False, False],
    ).head(12)
    if rescued.empty:
        return
    legends = []
    molecules = []
    for row in rescued.itertuples(index=False):
        molecules.append(mols[row.ik])
        identifier = row.mol_id if isinstance(row.mol_id, str) else row.ik
        legends.append(
            f"{identifier}\nChem {row.mean_p_chem:.2f} -> Bio {row.mean_p_bio:.2f}\n"
            f"Delta AP contrib {row.mean_delta_ap_contribution:+.4f}; "
            f"rank {row.mean_delta_rank_percentile:+.2f}"
        )
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=3,
        subImgSize=(420, 310),
        legends=legends,
        useSVG=False,
    )
    safe_context = context_id.replace("__", "_")
    image.save(figures / f"rescued_compound_atlas_{safe_context}.png")


def write_markdown_summary(
    output: Path,
    eligible: pd.DataFrame,
    compound_summary: pd.DataFrame,
    descriptor_results: pd.DataFrame,
    motif_results: pd.DataFrame,
    scaffold_results: pd.DataFrame,
) -> None:
    lines = [
        "# Compound- and scaffold-level residual analysis",
        "",
        "## Scope",
        "",
        "The analysis was restricted to endpoint-context combinations with positive Delta AUPRC and BH-adjusted Nadeau-Bengio q < 0.05.",
        "Repeated OOF predictions were averaged within compound; repeats were used as stability measurements rather than independent observations.",
        "",
        "## Confirmatory contexts",
        "",
    ]
    for row in eligible.itertuples(index=False):
        lines.append(
            f"- {row.task} x {row.lincs_cell_line} x {int(row.lincs_exposure_h)} h: "
            f"Delta AUPRC = {row.delta_auprc:+.3f}, q = {row.bh_q_value:.4g}."
        )
    lines.extend(["", "## Compound-level findings", ""])
    for context, frame in compound_summary.groupby("context_id", sort=False):
        active = frame.loc[frame["y"] == 1]
        lines.append(
            f"- {context.replace('__', ' x ')}: {len(frame)} labeled compounds, "
            f"{len(active)} actives, {int(active['consensus_rescued'].sum())} consensus rescued, "
            f"and {int(active['stable_rescued'].sum())} stable rescued."
        )
    lines.extend(["", "## Multiplicity-controlled exploratory screens", ""])
    for context, frame in descriptor_results.groupby("context_id", sort=False):
        n_mw = int((frame["mannwhitney_q"] < 0.05).sum())
        n_sp = int((frame["spearman_q"] < 0.05).sum())
        n_ap = int((frame["spearman_ap_contribution_q"] < 0.05).sum())
        lines.append(
            f"- {context.replace('__', ' x ')}: {n_mw} descriptor rescue-group contrasts and "
            f"{n_sp} probability-shift associations had within-context q < 0.05; "
            f"{n_ap} associations with per-compound AUPRC contribution gain had q < 0.05."
        )
    for context, frame in motif_results.groupby("context_id", sort=False):
        lines.append(
            f"- {context.replace('__', ' x ')}: {int((frame['fisher_q'] < 0.05).sum())} "
            "local Morgan environments passed within-context FDR correction."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A rescued compound is an active compound moved from below to at least 0.5 by the residual model. This is a descriptive operating-point definition; continuous residual probability and rank shifts are the primary compound-level quantities.",
            "",
            "The property, scaffold, and motif analyses are hypothesis-generating because they are conditioned on endpoint-contexts selected for model-level performance. They identify chemical neighborhoods in which the evaluated frozen chemical ensemble was complemented by transcriptomics; they do not establish universal toxicophores or information unavailable from structure in principle.",
            "",
            "## Files",
            "",
            "See `tables/` for complete compound-level and statistical outputs and `figures/` for publication-resolution PDF/PNG figures.",
        ]
    )
    (output / "RESULTS_AND_INTERPRETATION.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    RDLogger.DisableLog("rdApp.warning")
    require_inputs([args.table_s7, args.paired_mcf7_split, args.tox21_raw])
    tables, figures = prepare_output(args.output_dir)

    eligible = eligible_contexts(args.table_s7, args.q_threshold)
    eligible.to_csv(tables / "eligible_confirmatory_contexts.csv", index=False)
    cohort = load_split_union(args.paired_mcf7_split)
    raw = pd.read_csv(args.tox21_raw, sep="\t")
    identifiers = raw[["ik", "mol_id"]].drop_duplicates("ik")
    structures = cohort[["ik", "smiles_canon"]].merge(identifiers, on="ik", how="left")
    properties, all_mols = calculate_properties(structures)
    properties.to_csv(tables / "molecular_properties_and_scaffolds_all_mcf7_paired.csv", index=False)

    all_compounds = []
    all_active = []
    all_pairs = []
    all_descriptors = []
    all_motifs = []
    all_scaffold_enrichment = []
    all_scaffold_summary = []
    all_similarity_summary = []
    all_clusters = []
    all_cliffs = []
    all_cutoff_sensitivity = []

    for context_row in eligible.itertuples(index=False):
        row = pd.Series(context_row._asdict())
        context_id = f"{row['task']}__{row['lincs_cell_line']}__{int(row['lincs_exposure_h'])}h"
        oof_path = oof_path_for_context(row, args.oof_root_24h)
        require_inputs([oof_path])
        oof = pd.read_csv(oof_path)
        consensus = consensus_predictions(
            oof,
            args.probability_threshold,
            args.stable_low_rate,
            args.stable_high_rate,
        )
        consensus = consensus.merge(properties, on="ik", how="left", validate="one_to_one")
        consensus["context_id"] = context_id
        consensus["task"] = row["task"]
        consensus["cell_line"] = row["lincs_cell_line"]
        consensus["exposure_h"] = int(row["lincs_exposure_h"])
        consensus["variant"] = "viability_axis_residualized"
        all_compounds.append(consensus)

        active = consensus.loc[(consensus["y"] == 1) & consensus["valid_structure"]].copy()
        active_ids = active["ik"].tolist()
        active_mols = {ik: all_mols[ik] for ik in active_ids}
        fingerprints = morgan_fingerprints(active_mols)

        primary_clusters = None
        for cutoff in SIMILARITY_SENSITIVITY:
            assignments = butina_assignments(active_ids, fingerprints, cutoff)
            assignments["context_id"] = context_id
            n_clusters = assignments["butina_cluster"].nunique()
            all_cutoff_sensitivity.append(
                {
                    "context_id": context_id,
                    "similarity_cutoff": cutoff,
                    "n_active_compounds": len(active),
                    "n_clusters": int(n_clusters),
                    "n_singleton_clusters": int(
                        assignments.loc[assignments["cluster_size"] == 1, "butina_cluster"].nunique()
                    ),
                    "largest_cluster_size": int(assignments["cluster_size"].max()),
                }
            )
            if math.isclose(cutoff, PRIMARY_SIMILARITY_CUTOFF):
                primary_clusters = assignments
        active = active.merge(
            primary_clusters.drop(columns=["context_id"]), on="ik", how="left", validate="one_to_one"
        )
        primary_clusters = primary_clusters.merge(
            active[["ik", "compound_group", "consensus_rescued", "stable_rescued"]],
            on="ik",
            how="left",
        )
        all_clusters.append(primary_clusters)

        pairs = pairwise_similarity_table(active, fingerprints)
        pairs["context_id"] = context_id
        all_pairs.append(pairs)
        nearest = nearest_neighbor_features(active, pairs)
        active = active.merge(nearest, on="ik", how="left", validate="one_to_one")

        descriptor = descriptor_tests(active)
        descriptor["context_id"] = context_id
        all_descriptors.append(descriptor)

        motif_presence = sparse_morgan_motifs(active_mols)
        motifs = enrichment_tests(
            active,
            motif_presence,
            "morgan_fragment_smiles",
            None,
            min_rescued=3,
            min_total=5,
        )
        motifs["context_id"] = context_id
        all_motifs.append(motifs)

        scaffold_presence = {
            item.ik: {item.scaffold_key} for item in active[["ik", "scaffold_key"]].itertuples(index=False)
        }
        scaffold_labels = active.set_index("scaffold_key")["bemis_murcko_scaffold"].to_dict()
        scaffold_enrichment = enrichment_tests(
            active,
            scaffold_presence,
            "scaffold_key",
            scaffold_labels,
            min_rescued=2,
            min_total=3,
        )
        scaffold_enrichment["context_id"] = context_id
        all_scaffold_enrichment.append(scaffold_enrichment)

        scaffold_stats = scaffold_summary(active)
        scaffold_stats["context_id"] = context_id
        all_scaffold_summary.append(scaffold_stats)
        sim_stats = similarity_summary(active, pairs)
        sim_stats["context_id"] = context_id
        all_similarity_summary.append(sim_stats)

        cliffs = pairs.loc[
            (pairs["tanimoto"] >= 0.70)
            & (
                (pairs["delta_probability_1"] - pairs["delta_probability_2"]).abs()
                >= 0.15
            )
        ].copy()
        cliffs["absolute_delta_gain_difference"] = (
            cliffs["delta_probability_1"] - cliffs["delta_probability_2"]
        ).abs()
        cliffs["context_id"] = context_id
        all_cliffs.append(cliffs.sort_values("absolute_delta_gain_difference", ascending=False))

        active["context_id"] = context_id
        active["task"] = row["task"]
        active["cell_line"] = row["lincs_cell_line"]
        active["exposure_h"] = int(row["lincs_exposure_h"])
        all_active.append(active)
        plot_rescued_atlas(active, active_mols, figures, context_id)

    compound_summary = pd.concat(all_compounds, ignore_index=True)
    active_summary = pd.concat(all_active, ignore_index=True)
    pairs_summary = pd.concat(all_pairs, ignore_index=True)
    descriptor_results = pd.concat(all_descriptors, ignore_index=True)
    motif_results = pd.concat(all_motifs, ignore_index=True)
    scaffold_enrichment = pd.concat(all_scaffold_enrichment, ignore_index=True)
    scaffold_summaries = pd.concat(all_scaffold_summary, ignore_index=True)
    similarity_summaries = pd.concat(all_similarity_summary, ignore_index=True)
    cluster_assignments = pd.concat(all_clusters, ignore_index=True)
    residual_cliffs = pd.concat(all_cliffs, ignore_index=True) if all_cliffs else pd.DataFrame()
    cutoff_sensitivity = pd.DataFrame(all_cutoff_sensitivity)

    compound_summary.to_csv(tables / "compound_consensus_predictions_all_labeled.csv", index=False)
    active_summary.to_csv(tables / "active_compound_properties_predictions_clusters.csv", index=False)
    pairs_summary.to_csv(tables / "active_compound_pairwise_tanimoto.csv", index=False)
    descriptor_results.to_csv(tables / "physicochemical_descriptor_tests.csv", index=False)
    motif_results.to_csv(tables / "morgan_environment_enrichment.csv", index=False)
    scaffold_enrichment.to_csv(tables / "bemis_murcko_scaffold_enrichment.csv", index=False)
    scaffold_summaries.to_csv(tables / "scaffold_diversity_summary.csv", index=False)
    similarity_summaries.to_csv(tables / "tanimoto_similarity_summary.csv", index=False)
    cluster_assignments.to_csv(tables / "butina_cluster_assignments_primary_cutoff.csv", index=False)
    cutoff_sensitivity.to_csv(tables / "butina_cutoff_sensitivity.csv", index=False)
    residual_cliffs.to_csv(tables / "high_similarity_residual_response_cliffs.csv", index=False)

    top_rescued = active_summary.loc[active_summary["consensus_rescued"]].sort_values(
        ["context_id", "stable_rescued", "mean_delta_rank_percentile", "mean_delta_probability"],
        ascending=[True, False, False, False],
    )
    top_rescued.to_csv(tables / "rescued_active_compounds_ranked.csv", index=False)
    combined_correct = active_summary.loc[
        active_summary["combined_true_positive"]
    ].sort_values(
        ["context_id", "mean_delta_ap_contribution", "mean_p_bio"],
        ascending=[True, False, False],
    )
    combined_correct.to_csv(
        tables / "combined_correct_active_compounds_ranked.csv", index=False
    )
    top_ap_contributors = active_summary.loc[
        active_summary["mean_delta_ap_contribution"] > 0
    ].sort_values(
        ["context_id", "mean_delta_ap_contribution", "ap_contribution_increase_rate"],
        ascending=[True, False, False],
    )
    top_ap_contributors.to_csv(
        tables / "positive_auprc_contribution_compounds_ranked.csv", index=False
    )

    plot_overview(active_summary, figures, args.dpi)
    plot_descriptor_effects(descriptor_results, figures, args.dpi)
    plot_structural_summary(
        active_summary, pairs_summary, scaffold_summaries, figures, args.dpi
    )
    write_markdown_summary(
        args.output_dir,
        eligible,
        compound_summary,
        descriptor_results,
        motif_results,
        scaffold_summaries,
    )

    manifest = {
        "analysis": "compound_scaffold_residual_analysis_v1",
        "created_from_archived_oof_predictions": True,
        "models_refit": False,
        "eligibility_rule": f"delta_auprc > 0 and bh_q_value < {args.q_threshold}",
        "eligible_contexts": eligible[
            ["task", "lincs_cell_line", "lincs_exposure_h", "delta_auprc", "bh_q_value"]
        ].to_dict(orient="records"),
        "probability_threshold": args.probability_threshold,
        "stable_rescue_rates": {
            "chem_positive_rate_max": args.stable_low_rate,
            "bio_positive_rate_min": args.stable_high_rate,
        },
        "fingerprint": "Morgan radius 2, 2048-bit",
        "similarity": "Tanimoto",
        "butina_primary_similarity_cutoff": PRIMARY_SIMILARITY_CUTOFF,
        "butina_sensitivity_cutoffs": list(SIMILARITY_SENSITIVITY),
        "descriptor_bootstrap": {
            "method": "Bemis-Murcko-scaffold cluster bootstrap",
            "replicates": N_BOOTSTRAP,
            "seed": RNG_SEED,
        },
        "inputs": {
            "table_s7": str(args.table_s7),
            "oof_root_24h": str(args.oof_root_24h),
            "paired_mcf7_split": str(args.paired_mcf7_split),
            "tox21_raw": str(args.tox21_raw),
        },
        "counts": {
            "compound_context_rows": int(len(compound_summary)),
            "active_compound_context_rows": int(len(active_summary)),
            "rescued_active_context_rows": int(active_summary["consensus_rescued"].sum()),
            "stable_rescued_active_context_rows": int(active_summary["stable_rescued"].sum()),
        },
    }
    (args.output_dir / "RUN_COMPLETE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote compound/scaffold analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
