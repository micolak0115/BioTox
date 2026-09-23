#!/usr/bin/env python3
"""Characterize chemistry associated with biological residual logit scores.

This downstream analysis uses compound-level summaries of finalized repeated
outer-fold predictions. The primary compound-level outcome is the fitted
biological residual score beta^T z, reconstructed after removing the frozen
Stage-1 chemical logit and the combined-model training-fold intercept from the
combined-model logit. Associations condition on the frozen endpoint-specific
K=4 chemical ensemble logit and probability dispersion among its four
constituents. No toxicity predictor is refit.

The analysis is descriptive-to-discovery oriented. It does not attribute the
chemical ensemble to molecular fragments and does not establish causal
toxicophores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

try:
    from .analyze_residual_compound_scaffolds_v1 import (
        calculate_properties,
        consensus_predictions,
        load_split_union,
    )
except ImportError:  # Direct script execution from the publication directory.
    from analyze_residual_compound_scaffolds_v1 import (
        calculate_properties,
        consensus_predictions,
        load_split_union,
    )


HERE = Path(__file__).resolve().parent
RUN_OUTPUT = HERE / "_run_output"
PUBLICATION_ROOT = RUN_OUTPUT / "publication"
TABLE_S7 = (
    PUBLICATION_ROOT
    / "tables"
    / "07_complete_table_s7"
    / "table_s7_nominal_residual_gains_complete.csv"
)
CONSTITUENT_AUDIT = (
    PUBLICATION_ROOT
    / "stage1_k4_constituent_audit_v1"
    / "tables"
    / "stage2_cohort_constituent_summary_by_compound.csv"
)
PRIMARY_OOF_ROOT = RUN_OUTPUT / "nested_cv_k4_calibrated_lbfgs_repeated20x5_combined_v1"
PAIRED_6H_OOF_ROOT = RUN_OUTPUT / "nested_cv_k4_calibrated_paired6h_repeated20x5_innerseed_v3"
PAIRED_24H_OOF_ROOT = RUN_OUTPUT / "nested_cv_k4_calibrated_paired24h_repeated20x5_innerseed_v3"
PRIMARY_SPLIT_ROOT = RUN_OUTPUT / "matched_split_rebuilt"
PAIRED_SPLIT_ROOT = RUN_OUTPUT / "paired_timepoint_inputs_6h24h_8to12uM_v1" / "cohort"
DEFAULT_OUTPUT = (
    PUBLICATION_ROOT
    / "transcriptomic_rescue_chemical_properties_k4_residual_logit_v3"
)

RNG_SEED = 20260807
MODEL_RECONSTRUCTION_ATOL = 1e-12
MIN_ACTIVE_COMPOUNDS = 20
MIN_SCAFFOLDS = 10
CONTEXT_ORDER = [
    "NR-Aromatase__MCF7__24h__paired",
    "SR-ARE__HEPG2__6h__primary",
    "SR-MMP__HA1E__6h__primary",
    "SR-MMP__HEPG2__6h__primary",
    "SR-MMP__HT29__6h__primary",
    "SR-MMP__MCF7__6h__primary",
    "SR-MMP__MCF7__6h__paired",
    "SR-MMP__HA1E__24h__paired",
    "SR-MMP__MCF7__24h__paired",
    "SR-p53__MCF7__24h__paired",
]

GENERAL_CONTINUOUS_PROPERTIES = [
    "Molecular weight",
    "cLogP",
    "TPSA",
    "H-bond acceptors",
    "H-bond donors",
    "Rotatable bonds",
    "Ring count",
    "Aromatic rings",
    "Fraction Csp3",
    "Formal charge",
    "Heavy atoms",
    "Bertz complexity",
]

PROPERTY_LABELS = {
    "Molecular weight": "Molecular weight",
    "cLogP": "cLogP",
    "TPSA": "TPSA",
    "H-bond acceptors": "H-bond acceptors",
    "H-bond donors": "H-bond donors",
    "Rotatable bonds": "Rotatable bonds",
    "Ring count": "Ring count",
    "Aromatic rings": "Aromatic rings",
    "Fraction Csp3": "Fraction Csp3",
    "Formal charge": "Formal charge",
    "Heavy atoms": "Heavy atoms",
    "Bertz complexity": "Bertz complexity",
    "aromatic_nitrogen_count": "Aromatic nitrogen count",
    "azole_like_ring": "Azole-like aromatic ring",
    "nitrile": "Nitrile",
    "steroid_like_fused_polycycle": "Steroid-like fused-polycycle proxy",
    "rigid_polycycle": "Rigid polycycle",
    "direct_electrophile": "Direct electrophile alert",
    "redox_diphenol": "Oxidizable diphenol",
    "aromatic_nitro": "Aromatic nitro/redox alert",
    "oxidative_electrophile_any": "Any oxidative/electrophilic alert",
    "cationic_amphiphile": "Formal-cation amphiphile proxy",
    "high_lipophilicity_low_polarity": "High lipophilicity/low polarity",
    "polyaromatic_lipophile": "Polyaromatic lipophile",
    "nitrophenol_protonophore": "Nitrophenol-like protonophore",
    "mitochondrial_liability_any": "Any mitochondrial-liability proxy",
    "dna_reactive_electrophile": "DNA-reactive electrophile alert",
    "aromatic_amine": "Aromatic amine alert",
    "hydrazine": "Hydrazine alert",
    "aromatic_azo": "Aromatic azo alert",
    "high_aromatic_burden": "High aromatic-ring burden",
    "genotoxicity_alert_any": "Any genotoxicity alert",
}

ENDPOINT_PROPERTIES = {
    "NR-Aromatase": [
        "aromatic_nitrogen_count",
        "azole_like_ring",
        "nitrile",
        "steroid_like_fused_polycycle",
        "rigid_polycycle",
    ],
    "SR-ARE": [
        "direct_electrophile",
        "redox_diphenol",
        "aromatic_nitro",
        "oxidative_electrophile_any",
    ],
    "SR-MMP": [
        "cationic_amphiphile",
        "high_lipophilicity_low_polarity",
        "polyaromatic_lipophile",
        "nitrophenol_protonophore",
        "mitochondrial_liability_any",
    ],
    "SR-p53": [
        "dna_reactive_electrophile",
        "aromatic_nitro",
        "aromatic_amine",
        "hydrazine",
        "aromatic_azo",
        "high_aromatic_burden",
        "genotoxicity_alert_any",
    ],
}

SMARTS = {
    "michael_acceptor": "[C,c]=[C,c]-[C,S](=[O,S])-[#6,#7,#8]",
    "catechol": "c1c([OH])c([OH])ccc1",
    "hydroquinone": "c1c([OH])cc([OH])cc1",
    "epoxide_or_aziridine": "[O,N;r3]1[C;r3][C;r3]1",
    "aldehyde": "[CX3H1](=O)[#6]",
    "aromatic_nitro": "[c]-[N+](=O)[O-]",
    "aromatic_amine": "[c]-[N;H1,H2;v3]",
    "alkyl_halide": "[C;X4]-[Cl,Br,I]",
    "hydrazine": "[N;X3]-[N;X3]",
    "aromatic_azo": "[c]-[N]=[N]-[c]",
    "nitrogen_mustard": "[N;X3]([CH2][CH2][Cl,Br,I])([CH2][CH2][Cl,Br,I])",
    "phenol": "[c]-[OH]",
    "nitrile": "[C,c]#[N]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-s7", type=Path, default=TABLE_S7)
    parser.add_argument("--constituent-audit", type=Path, default=CONSTITUENT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-cluster-bootstrap", type=int, default=2000)
    parser.add_argument("--n-wild-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path) -> tuple[Path, Path]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    tables = path / "tables"
    figures = path / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    return tables, figures


def cohort_code(value: str) -> str:
    return "primary" if str(value).startswith("Primary") else "paired"


def make_context_id(row: pd.Series) -> str:
    return (
        f"{row['task']}__{row['lincs_cell_line']}__"
        f"{int(row['lincs_exposure_h'])}h__{cohort_code(row['analysis_cohort'])}"
    )


def select_contexts(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    selected = table.loc[
        table["delta_auprc"].gt(0) & table["nominal_p_value"].le(0.05)
    ].copy()
    selected["cohort_code"] = selected["analysis_cohort"].map(cohort_code)
    selected["context_id"] = selected.apply(make_context_id, axis=1)
    if set(selected["context_id"]) != set(CONTEXT_ORDER):
        raise ValueError(
            "Nominal-positive context set differs from the prespecified audit set: "
            f"observed={sorted(selected['context_id'])}"
        )
    selected["context_order"] = selected["context_id"].map(
        {value: index for index, value in enumerate(CONTEXT_ORDER)}
    )
    return selected.sort_values("context_order").reset_index(drop=True)


def source_paths(row: pd.Series) -> tuple[Path, Path]:
    task = row["task"]
    cell = row["lincs_cell_line"]
    exposure = int(row["lincs_exposure_h"])
    if row["cohort_code"] == "primary":
        oof_root = PRIMARY_OOF_ROOT
        split = PRIMARY_SPLIT_ROOT / f"tox21_scaffold_df_split_{cell}_8to12uM_6h.pkl"
    else:
        oof_root = PAIRED_6H_OOF_ROOT if exposure == 6 else PAIRED_24H_OOF_ROOT
        split = PAIRED_SPLIT_ROOT / f"tox21_scaffold_df_split_{cell}_8to12uM_paired6h24h.pkl"
    oof = oof_root / f"repeated_nested_cv_oof_predictions_residualized_{task}__{cell}.csv"
    return oof, split


def compile_queries() -> dict[str, Chem.Mol]:
    queries = {}
    for name, smarts in SMARTS.items():
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f"Invalid SMARTS for {name}: {smarts}")
        queries[name] = query
    return queries


def ring_features(mol: Chem.Mol) -> dict[str, float | bool]:
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    fused = sum(
        any(index != other and len(ring.intersection(candidate)) >= 2 for other, candidate in enumerate(rings))
        for index, ring in enumerate(rings)
    )
    azole_like = False
    for ring in rings:
        if len(ring) != 5:
            continue
        atoms = [mol.GetAtomWithIdx(index) for index in ring]
        if all(atom.GetIsAromatic() for atom in atoms) and sum(
            atom.GetAtomicNum() == 7 for atom in atoms
        ) >= 2:
            azole_like = True
            break
    aromatic_nitrogen_count = sum(
        atom.GetAtomicNum() == 7 and atom.GetIsAromatic() for atom in mol.GetAtoms()
    )
    return {
        "fused_ring_count": float(fused),
        "azole_like_ring": bool(azole_like),
        "aromatic_nitrogen_count": float(aromatic_nitrogen_count),
    }


def annotate_properties(properties: pd.DataFrame, mols: dict[str, Chem.Mol]) -> pd.DataFrame:
    queries = compile_queries()
    annotated = properties.copy()
    annotated["Bertz complexity"] = [Descriptors.BertzCT(mols[ik]) for ik in annotated["ik"]]
    for name, query in queries.items():
        annotated[name] = [mols[ik].HasSubstructMatch(query) for ik in annotated["ik"]]
    ring_records = pd.DataFrame(
        [{"ik": ik, **ring_features(mol)} for ik, mol in mols.items()]
    )
    annotated = annotated.merge(ring_records, on="ik", how="left", validate="one_to_one")

    annotated["redox_diphenol"] = annotated["catechol"] | annotated["hydroquinone"]
    annotated["direct_electrophile"] = (
        annotated["michael_acceptor"]
        | annotated["epoxide_or_aziridine"]
        | annotated["aldehyde"]
    )
    annotated["oxidative_electrophile_any"] = (
        annotated["direct_electrophile"]
        | annotated["redox_diphenol"]
        | annotated["aromatic_nitro"]
    )
    annotated["cationic_amphiphile"] = (
        annotated["Formal charge"].gt(0)
        & annotated["cLogP"].ge(2)
        & annotated["Aromatic rings"].ge(1)
    )
    annotated["high_lipophilicity_low_polarity"] = (
        annotated["cLogP"].ge(3) & annotated["TPSA"].le(75)
    )
    annotated["polyaromatic_lipophile"] = (
        annotated["Aromatic rings"].ge(3) & annotated["cLogP"].ge(3)
    )
    annotated["nitrophenol_protonophore"] = annotated["phenol"] & annotated["aromatic_nitro"]
    annotated["mitochondrial_liability_any"] = (
        annotated["cationic_amphiphile"]
        | annotated["high_lipophilicity_low_polarity"]
        | annotated["polyaromatic_lipophile"]
        | annotated["nitrophenol_protonophore"]
    )
    annotated["dna_reactive_electrophile"] = (
        annotated["direct_electrophile"]
        | annotated["alkyl_halide"]
        | annotated["nitrogen_mustard"]
    )
    annotated["high_aromatic_burden"] = annotated["Aromatic rings"].ge(3)
    annotated["genotoxicity_alert_any"] = (
        annotated["dna_reactive_electrophile"]
        | annotated["aromatic_nitro"]
        | annotated["aromatic_amine"]
        | annotated["hydrazine"]
        | annotated["aromatic_azo"]
    )
    annotated["steroid_like_fused_polycycle"] = (
        annotated["Ring count"].ge(4)
        & annotated["fused_ring_count"].ge(3)
        & annotated["Aromatic rings"].le(1)
        & annotated["Fraction Csp3"].ge(0.25)
    )
    annotated["rigid_polycycle"] = (
        annotated["Ring count"].ge(3) & annotated["Rotatable bonds"].le(3)
    )
    return annotated


def zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    scale = float(values.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(values), mean, scale
    return (values - mean) / scale, mean, scale


def logit_probability(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def attach_biological_residual_scores(oof: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    required = {
        "p_chem_stage1_raw",
        "p_chem_only",
        "p_chem_bio",
        "null_intercept",
        "bio_intercept",
    }
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF file missing residual-score columns: {sorted(missing)}")
    frame = oof.copy()
    eta_stage1 = logit_probability(frame["p_chem_stage1_raw"].to_numpy(float))
    eta_chem = logit_probability(frame["p_chem_only"].to_numpy(float))
    eta_bio = logit_probability(frame["p_chem_bio"].to_numpy(float))
    frame["delta_logit_total"] = eta_bio - eta_chem
    frame["intercept_shift"] = (
        frame["bio_intercept"].to_numpy(float)
        - frame["null_intercept"].to_numpy(float)
    )
    frame["biological_residual_score"] = (
        eta_bio
        - eta_stage1
        - frame["bio_intercept"].to_numpy(float)
    )
    equivalent = frame["delta_logit_total"] - frame["intercept_shift"]
    max_error = float(
        np.max(np.abs(frame["biological_residual_score"].to_numpy(float) - equivalent))
    )
    if max_error > 1e-9:
        raise ValueError(
            f"Biological residual-score identities disagree by {max_error:.3e}"
        )
    return frame, max_error


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        raise np.linalg.LinAlgError("rank-deficient design")
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    return beta, fitted, rank


def conditional_design(
    frame: pd.DataFrame,
    property_name: str,
    property_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    y = frame["mean_biological_residual_score"].to_numpy(dtype=float)
    logit, logit_mean, logit_sd = zscore(frame["ensemble_logit"].to_numpy(float))
    dispersion, dispersion_mean, dispersion_sd = zscore(
        frame["constituent_sd_population"].to_numpy(float)
    )
    raw_property = frame[property_name].to_numpy(dtype=float)
    if property_type == "continuous":
        property_term, property_mean, property_sd = zscore(raw_property)
        effect_unit = "per_1sd_property"
    else:
        property_term = raw_property
        property_mean = float(raw_property.mean())
        property_sd = float(raw_property.std(ddof=0))
        effect_unit = "present_vs_absent"
    covariates = np.column_stack([logit, dispersion])
    full = np.column_stack([np.ones(len(frame)), property_term, covariates])
    null = np.column_stack([np.ones(len(frame)), covariates])
    metadata = {
        "property_mean": property_mean,
        "property_sd": property_sd,
        "effect_unit": effect_unit,
        "ensemble_logit_mean": logit_mean,
        "ensemble_logit_sd": logit_sd,
        "constituent_sd_mean": dispersion_mean,
        "constituent_sd_sd": dispersion_sd,
    }
    return y, full, null, metadata


def cluster_indices(labels: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique = pd.unique(labels)
    indices = [np.flatnonzero(labels == label) for label in unique]
    return unique, indices


def conditional_property_test(
    frame: pd.DataFrame,
    property_name: str,
    property_type: str,
    *,
    n_cluster_bootstrap: int,
    n_wild_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    y, full, null, metadata = conditional_design(frame, property_name, property_type)
    labels = frame["scaffold_key"].astype(str).to_numpy()
    unique_scaffolds, groups = cluster_indices(labels)
    n_scaffolds = len(unique_scaffolds)
    if metadata["property_sd"] <= 1e-12:
        return {**metadata, "estimable": False, "reason": "constant_property"}
    try:
        beta, fitted, rank = fit_ols(full, y)
        beta_null, fitted_null, _ = fit_ols(null, y)
    except np.linalg.LinAlgError:
        return {**metadata, "estimable": False, "reason": "rank_deficient"}

    residual = y - fitted
    residual_null = y - fitted_null
    sse_full = float(np.sum(residual**2))
    sse_null = float(np.sum(residual_null**2))
    observed = float(beta[1])
    partial_r2 = (sse_null - sse_full) / sse_null if sse_null > 0 else np.nan

    bootstrap_estimates = []
    for _ in range(n_cluster_bootstrap):
        sampled = rng.integers(0, n_scaffolds, size=n_scaffolds)
        sampled_indices = np.concatenate([groups[index] for index in sampled])
        try:
            estimate, _, _ = fit_ols(full[sampled_indices], y[sampled_indices])
        except np.linalg.LinAlgError:
            continue
        bootstrap_estimates.append(float(estimate[1]))
    bootstrap_estimates = np.asarray(bootstrap_estimates, dtype=float)
    if len(bootstrap_estimates) < max(100, int(0.8 * n_cluster_bootstrap)):
        return {**metadata, "estimable": False, "reason": "unstable_cluster_bootstrap"}
    ci_low, ci_high = np.quantile(bootstrap_estimates, [0.025, 0.975])

    wild_estimates = []
    for _ in range(n_wild_bootstrap):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_scaffolds)
        sign_by_row = np.empty(len(frame), dtype=float)
        for index, group in enumerate(groups):
            sign_by_row[group] = signs[index]
        y_star = fitted_null + residual_null * sign_by_row
        try:
            estimate, _, _ = fit_ols(full, y_star)
        except np.linalg.LinAlgError:
            continue
        wild_estimates.append(float(estimate[1]))
    wild_estimates = np.asarray(wild_estimates, dtype=float)
    if len(wild_estimates) < max(100, int(0.8 * n_wild_bootstrap)):
        return {**metadata, "estimable": False, "reason": "unstable_wild_bootstrap"}
    wild_p = float(
        (1 + np.sum(np.abs(wild_estimates) >= abs(observed)))
        / (len(wild_estimates) + 1)
    )

    xtx_inverse = np.linalg.pinv(full.T @ full)
    leverage = np.einsum("ij,jk,ik->i", full, xtx_inverse, full)
    degrees_freedom = max(1, len(y) - rank)
    mse = sse_full / degrees_freedom
    cooks = (
        residual**2
        / max(mse * full.shape[1], np.finfo(float).eps)
        * leverage
        / np.maximum((1.0 - leverage) ** 2, np.finfo(float).eps)
    )
    return {
        **metadata,
        "estimable": True,
        "reason": "",
        "conditional_effect": observed,
        "cluster_bootstrap_ci_low": float(ci_low),
        "cluster_bootstrap_ci_high": float(ci_high),
        "wild_cluster_bootstrap_p": wild_p,
        "partial_r2": float(partial_r2),
        "full_model_r2": float(1.0 - sse_full / np.sum((y - y.mean()) ** 2))
        if np.sum((y - y.mean()) ** 2) > 0
        else np.nan,
        "n_cluster_bootstrap_valid": int(len(bootstrap_estimates)),
        "n_wild_bootstrap_valid": int(len(wild_estimates)),
        "max_leverage": float(np.max(leverage)),
        "max_cooks_distance": float(np.max(cooks)),
        "design_condition_number": float(np.linalg.cond(full)),
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan
    return float(
        (np.greater(x[:, None], y).sum() - np.less(x[:, None], y).sum())
        / (len(x) * len(y))
    )


def binary_rescue_sensitivity(
    active: pd.DataFrame,
    property_name: str,
    property_type: str,
) -> dict:
    rescued = active["consensus_rescued"].astype(bool)
    values = active[property_name]
    if rescued.sum() == 0 or (~rescued).sum() == 0:
        return {"binary_estimable": False, "binary_reason": "single_rescue_class"}
    if property_type == "continuous":
        x = values.loc[rescued].to_numpy(float)
        y = values.loc[~rescued].to_numpy(float)
        result = mannwhitneyu(x, y, alternative="two-sided")
        return {
            "binary_estimable": True,
            "binary_reason": "",
            "rescued_mean": float(np.mean(x)),
            "other_active_mean": float(np.mean(y)),
            "cliffs_delta": cliffs_delta(x, y),
            "binary_unadjusted_p": float(result.pvalue),
            "fisher_odds_ratio": np.nan,
        }
    present = values.astype(bool)
    table = np.array(
        [
            [(rescued & present).sum(), (rescued & ~present).sum()],
            [(~rescued & present).sum(), (~rescued & ~present).sum()],
        ],
        dtype=int,
    )
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    return {
        "binary_estimable": True,
        "binary_reason": "",
        "rescued_mean": float(present.loc[rescued].mean()),
        "other_active_mean": float(present.loc[~rescued].mean()),
        "cliffs_delta": np.nan,
        "binary_unadjusted_p": float(p_value),
        "fisher_odds_ratio": float(odds_ratio),
    }


def bh_adjust(frame: pd.DataFrame, p_column: str, group_column: str | None) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    groups = [(None, frame.index)] if group_column is None else frame.groupby(group_column).groups.items()
    for _, indices in groups:
        indices = list(indices)
        valid = frame.loc[indices, p_column].notna()
        valid_indices = frame.loc[indices].index[valid]
        if len(valid_indices):
            result.loc[valid_indices] = multipletests(
                frame.loc[valid_indices, p_column], method="fdr_bh"
            )[1]
    return result


def analyze_context(
    row: pd.Series,
    constituent_audit: pd.DataFrame,
    *,
    n_cluster_bootstrap: int,
    n_wild_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Chem.Mol]]:
    oof_path, split_path = source_paths(row)
    if not oof_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"Missing OOF or split input: {oof_path}, {split_path}")
    oof = pd.read_csv(oof_path)
    if oof["repeat_id"].nunique() != 20 or oof["outer_fold_id"].nunique() != 5:
        raise ValueError(f"{row['context_id']}: expected 20 x 5 repeated OOF predictions")
    oof, residual_identity_error = attach_biological_residual_scores(oof)
    consensus = consensus_predictions(oof, threshold=0.5, stable_low=0.20, stable_high=0.80)
    residual_summary = (
        oof.groupby("ik", as_index=False)
        .agg(
            mean_biological_residual_score=("biological_residual_score", "mean"),
            sd_biological_residual_score=("biological_residual_score", "std"),
            mean_delta_logit_total=("delta_logit_total", "mean"),
            mean_intercept_shift=("intercept_shift", "mean"),
        )
    )
    consensus = consensus.merge(
        residual_summary,
        on="ik",
        how="left",
        validate="one_to_one",
    )

    cohort = load_split_union(split_path)
    structures = cohort[["ik", "smiles_canon"]].drop_duplicates("ik")
    properties, mols = calculate_properties(structures)
    if not properties["valid_structure"].all():
        raise ValueError(f"{row['context_id']}: invalid chemical structures present")
    properties = annotate_properties(properties, mols)

    task_audit = constituent_audit.loc[
        constituent_audit["task"].eq(row["task"]),
        [
            "ik",
            "ensemble_probability",
            "ensemble_logit",
            "constituent_sd_population",
            "constituent_range",
        ],
    ].copy()
    consensus = consensus.merge(properties, on="ik", how="left", validate="one_to_one")
    consensus = consensus.merge(task_audit, on="ik", how="left", validate="one_to_one")
    if consensus[["ensemble_probability", "constituent_sd_population"]].isna().any().any():
        raise ValueError(f"{row['context_id']}: missing K=4 constituent audit values")

    raw_by_ik = oof.groupby("ik")["p_chem_stage1_raw"].agg(["min", "max", "first"])
    if (raw_by_ik["max"] - raw_by_ik["min"]).abs().max() > MODEL_RECONSTRUCTION_ATOL:
        raise ValueError(f"{row['context_id']}: frozen Stage-1 probability varies across repeats")
    comparison = consensus[["ik", "ensemble_probability"]].merge(
        raw_by_ik[["first"]], left_on="ik", right_index=True, validate="one_to_one"
    )
    max_probability_error = float(
        np.max(np.abs(comparison["ensemble_probability"] - comparison["first"]))
    )
    if max_probability_error > MODEL_RECONSTRUCTION_ATOL:
        raise ValueError(
            f"{row['context_id']}: K=4 audit differs from Stage-2 raw prior by "
            f"{max_probability_error:.3e}"
        )

    for column, value in {
        "context_id": row["context_id"],
        "family": row["family"],
        "task": row["task"],
        "cell_line": row["lincs_cell_line"],
        "exposure_h": int(row["lincs_exposure_h"]),
        "analysis_cohort": row["analysis_cohort"],
        "cohort_code": row["cohort_code"],
        "nominal_p_value": float(row["nominal_p_value"]),
        "bh_q_value": float(row["bh_q_value"]),
    }.items():
        consensus[column] = value
    consensus["residual_score_identity_max_abs_error"] = residual_identity_error

    active = consensus.loc[consensus["y"].eq(1)].copy().reset_index(drop=True)
    if len(active) < MIN_ACTIVE_COMPOUNDS:
        raise ValueError(f"{row['context_id']}: fewer than {MIN_ACTIVE_COMPOUNDS} active compounds")
    if active["scaffold_key"].nunique() < MIN_SCAFFOLDS:
        raise ValueError(f"{row['context_id']}: fewer than {MIN_SCAFFOLDS} active scaffolds")

    property_names = GENERAL_CONTINUOUS_PROPERTIES + ENDPOINT_PROPERTIES[row["task"]]
    property_types = {
        name: "continuous" for name in GENERAL_CONTINUOUS_PROPERTIES
    }
    property_types["aromatic_nitrogen_count"] = "continuous"
    for name in ENDPOINT_PROPERTIES[row["task"]]:
        property_types.setdefault(name, "binary")

    test_rows = []
    for index, property_name in enumerate(property_names):
        rng = np.random.default_rng(seed + 1009 * int(row["context_order"]) + index)
        conditional = conditional_property_test(
            active,
            property_name,
            property_types[property_name],
            n_cluster_bootstrap=n_cluster_bootstrap,
            n_wild_bootstrap=n_wild_bootstrap,
            rng=rng,
        )
        sensitivity = binary_rescue_sensitivity(
            active, property_name, property_types[property_name]
        )
        test_rows.append(
            {
                "context_id": row["context_id"],
                "family": row["family"],
                "task": row["task"],
                "cell_line": row["lincs_cell_line"],
                "exposure_h": int(row["lincs_exposure_h"]),
                "analysis_cohort": row["analysis_cohort"],
                "property": property_name,
                "property_label": PROPERTY_LABELS[property_name],
                "property_type": property_types[property_name],
                "property_family": "general_descriptor"
                if property_name in GENERAL_CONTINUOUS_PROPERTIES
                else "endpoint_aligned_proxy",
                "n_active_compounds": len(active),
                "n_active_scaffolds": active["scaffold_key"].nunique(),
                "largest_active_scaffold_size": int(active["scaffold_key"].value_counts().max()),
                "n_consensus_rescued": int(active["consensus_rescued"].sum()),
                **conditional,
                **sensitivity,
            }
        )
    tests = pd.DataFrame(test_rows)

    summary = {
        "context_id": row["context_id"],
        "family": row["family"],
        "task": row["task"],
        "cell_line": row["lincs_cell_line"],
        "exposure_h": int(row["lincs_exposure_h"]),
        "analysis_cohort": row["analysis_cohort"],
        "n_labeled_compounds": len(consensus),
        "n_active_compounds": len(active),
        "n_active_scaffolds": active["scaffold_key"].nunique(),
        "n_consensus_rescued": int(active["consensus_rescued"].sum()),
        "mean_delta_auprc_reconstructed": float(active["mean_delta_ap_contribution"].sum()),
        "table_delta_auprc": float(row["delta_auprc"]),
        "k4_probability_max_abs_reconstruction_error": max_probability_error,
        "residual_score_identity_max_abs_error": residual_identity_error,
    }
    if not np.isclose(
        summary["mean_delta_auprc_reconstructed"], summary["table_delta_auprc"], atol=1e-11
    ):
        raise ValueError(f"{row['context_id']}: compound AP contributions do not reconstruct delta AUPRC")
    return consensus, active, tests, mols


def match_rescued_compounds(active_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    for context_id, context in active_all.groupby("context_id", sort=False):
        rescued = context.loc[context["consensus_rescued"]].copy()
        controls = context.loc[~context["consensus_rescued"]].copy()
        if rescued.empty or controls.empty:
            continue
        mols = {row.ik: Chem.MolFromSmiles(row.smiles_canon) for row in context.itertuples()}
        fingerprints = {ik: generator.GetFingerprint(mol) for ik, mol in mols.items()}
        z_logit, _, _ = zscore(context["ensemble_logit"].to_numpy(float))
        z_sd, _, _ = zscore(context["constituent_sd_population"].to_numpy(float))
        context = context.copy()
        context["z_ensemble_logit"] = z_logit
        context["z_constituent_sd"] = z_sd
        lookup = context.set_index("ik")
        controls = context.loc[~context["consensus_rescued"]].copy()
        for rescued_row in rescued.itertuples():
            candidates = controls.loc[controls["scaffold_key"].eq(rescued_row.scaffold_key)]
            same_scaffold_available = not candidates.empty
            if candidates.empty:
                candidates = controls
            best = None
            for control in candidates.itertuples():
                similarity = float(
                    DataStructs.TanimotoSimilarity(
                        fingerprints[rescued_row.ik], fingerprints[control.ik]
                    )
                )
                score = (
                    0.40 * (1.0 - similarity)
                    + 0.30
                    * abs(
                        lookup.loc[rescued_row.ik, "z_ensemble_logit"]
                        - lookup.loc[control.ik, "z_ensemble_logit"]
                    )
                    + 0.30
                    * abs(
                        lookup.loc[rescued_row.ik, "z_constituent_sd"]
                        - lookup.loc[control.ik, "z_constituent_sd"]
                    )
                )
                if best is None or score < best[0]:
                    best = (score, similarity, control)
            if best is None:
                continue
            score, similarity, control = best
            rows.append(
                {
                    "context_id": context_id,
                    "task": rescued_row.task,
                    "cell_line": rescued_row.cell_line,
                    "exposure_h": rescued_row.exposure_h,
                    "rescued_ik": rescued_row.ik,
                    "rescued_smiles": rescued_row.smiles_canon,
                    "control_ik": control.ik,
                    "control_smiles": control.smiles_canon,
                    "same_scaffold": bool(same_scaffold_available),
                    "tanimoto_similarity": similarity,
                    "matching_score": float(score),
                    "rescued_delta_probability": rescued_row.mean_delta_probability,
                    "control_delta_probability": control.mean_delta_probability,
                    "rescued_biological_residual_score": rescued_row.mean_biological_residual_score,
                    "control_biological_residual_score": control.mean_biological_residual_score,
                    "rescued_ensemble_probability": rescued_row.ensemble_probability,
                    "control_ensemble_probability": control.ensemble_probability,
                    "rescued_constituent_sd": rescued_row.constituent_sd_population,
                    "control_constituent_sd": control.constituent_sd_population,
                }
            )
    return pd.DataFrame(rows)


def context_label(context_id: str) -> str:
    task, cell, exposure, cohort = context_id.split("__")
    cohort_label = "primary" if cohort == "primary" else "paired"
    return f"{task} | {cell} | {exposure} | {cohort_label}"


def plot_prior_plane(active_all: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(8.27, 11.69), sharex=False, sharey=False)
    values = active_all["mean_biological_residual_score"].to_numpy(float)
    bound = max(abs(np.quantile(values, 0.02)), abs(np.quantile(values, 0.98)), 1e-6)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    last = None
    for axis, context_id in zip(axes.flat, CONTEXT_ORDER):
        context = active_all.loc[active_all["context_id"].eq(context_id)]
        last = axis.scatter(
            context["ensemble_probability"],
            context["constituent_sd_population"],
            c=context["mean_biological_residual_score"],
            cmap="RdBu_r",
            norm=norm,
            s=23,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.82,
        )
        rescued = context["consensus_rescued"].astype(bool)
        axis.scatter(
            context.loc[rescued, "ensemble_probability"],
            context.loc[rescued, "constituent_sd_population"],
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            s=52,
        )
        axis.set_title(context_label(context_id), fontsize=9.5, fontweight="bold")
        axis.grid(linestyle=":", color="#c3c3c3", alpha=0.65)
        axis.tick_params(labelsize=8.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.supxlabel("Frozen K=4 ensemble probability", fontsize=11.5, y=0.045)
    fig.supylabel("Between-constituent probability SD", fontsize=11.5, x=0.025)
    if last is not None:
        colorbar = fig.colorbar(
            last,
            ax=axes,
            orientation="vertical",
            fraction=0.022,
            pad=0.025,
            shrink=0.58,
        )
        colorbar.set_label(
            "Mean biological residual score (logit units)",
            fontsize=10.5,
        )
        colorbar.ax.tick_params(labelsize=8.5)
    fig.subplots_adjust(left=0.12, right=0.91, top=0.97, bottom=0.09, hspace=0.43, wspace=0.27)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_conditional_forest(tests: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(10, 1, figsize=(8.27, 16.5))
    for axis, context_id in zip(axes, CONTEXT_ORDER):
        context = tests.loc[tests["context_id"].eq(context_id) & tests["estimable"]].copy()
        context["abs_effect"] = context["conditional_effect"].abs()
        endpoint = context.loc[context["property_family"].eq("endpoint_aligned_proxy")]
        general = context.loc[context["property_family"].eq("general_descriptor")]
        selected = pd.concat(
            [
                endpoint.sort_values("abs_effect", ascending=False).head(3),
                general.sort_values("abs_effect", ascending=False).head(3),
            ]
        ).drop_duplicates("property")
        selected = selected.sort_values("conditional_effect").reset_index(drop=True)
        y = np.arange(len(selected))
        significant = selected["q_within_context"].le(0.05)
        colors = np.where(significant, "#b94a48", "#9a9a9a")
        for index, row in selected.iterrows():
            axis.errorbar(
                row["conditional_effect"],
                index,
                xerr=np.array(
                    [
                        [
                            (row["conditional_effect"] - row["cluster_bootstrap_ci_low"])
                            * 1.0
                        ],
                        [
                            (row["cluster_bootstrap_ci_high"] - row["conditional_effect"])
                            * 1.0
                        ],
                    ]
                ),
                fmt="o",
                color=colors[index],
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
                markersize=5.2,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
        axis.axvline(0.0, color="#666666", linestyle=":", linewidth=1.0)
        axis.set_yticks(y)
        axis.set_yticklabels(selected["property_label"], fontsize=8.8)
        axis.set_title(context_label(context_id), fontsize=10.0, fontweight="bold")
        axis.grid(axis="x", linestyle=":", color="#c3c3c3", alpha=0.65)
        axis.tick_params(axis="x", labelsize=8.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.supxlabel(
        "Adjusted association with biological residual score (logit units)",
        fontsize=11.5,
        y=0.018,
    )
    fig.subplots_adjust(left=0.32, right=0.97, top=0.985, bottom=0.045, hspace=0.65)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_representative_pairs(
    pairs: pd.DataFrame,
    path: Path,
    dpi: int,
) -> pd.DataFrame:
    selected_rows = []
    for task in ["NR-Aromatase", "SR-ARE", "SR-MMP", "SR-p53"]:
        candidates = pairs.loc[pairs["task"].eq(task)].copy()
        if candidates.empty:
            continue
        candidates["priority"] = (
            candidates["rescued_delta_probability"]
            - candidates["control_delta_probability"]
            + 0.02 * candidates["tanimoto_similarity"]
        )
        selected_rows.append(candidates.sort_values("priority", ascending=False).iloc[0])
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    if selected.empty:
        return selected

    fig, axes = plt.subplots(len(selected), 2, figsize=(8.27, 2.55 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_index, row in selected.iterrows():
        for column, role in enumerate(["rescued", "control"]):
            smiles = row[f"{role}_smiles"]
            mol = Chem.MolFromSmiles(smiles)
            image = Draw.MolToImage(mol, size=(700, 420), kekulize=True)
            axes[row_index, column].imshow(image)
            axes[row_index, column].axis("off")
            if role == "rescued":
                title = (
                    f"{row['task']} | {row['cell_line']} | {int(row['exposure_h'])} h | "
                    f"Tanimoto={row['tanimoto_similarity']:.2f}\n"
                    "Transcriptomically rescued active"
                )
            else:
                title = "Matched other active"
            probability = row[f"{role}_ensemble_probability"]
            dispersion = row[f"{role}_constituent_sd"]
            axes[row_index, column].set_title(
                f"{title}\nK=4 mean={probability:.3f}; SD={dispersion:.3f}",
                fontsize=9.6,
                fontweight="bold" if role == "rescued" else "normal",
            )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.985, bottom=0.02, hspace=0.54, wspace=0.04)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return selected.drop(columns=["priority"])


def write_report(output_dir: Path, contexts: pd.DataFrame, tests: pd.DataFrame) -> None:
    significant = tests.loc[
        tests["estimable"] & tests["q_within_context"].le(0.05)
    ].copy()
    lines = [
        "# Chemical properties associated with transcriptomic rescue",
        "",
        "## Methods",
        "",
        "The inferential unit was the compound. For each outer-test prediction, the biological residual score was reconstructed as logit[P(Chemistry + transcriptomics)] minus the frozen Stage-1 chemical logit and the combined-model training-fold intercept. Under the fitted model this quantity equals beta^T z. Scores were averaged across 20 repeated scaffold assignments; the 100 correlated fold rows were not treated as independent observations. Analyses were restricted to active compounds in endpoint-cell-time contexts with positive delta AUPRC and nominal Nadeau-Bengio-corrected P <= 0.05.",
        "",
        "For each prespecified property, an unpenalized conditional linear model related the compound-level mean biological residual score to the property while adjusting linearly for the frozen K=4 ensemble logit and between-constituent probability SD. Unlike delta probability, this logit-scale outcome does not contain sigmoid-scale compression; the chemical-score covariate was retained to characterize property effects conditional on molecular-prior position. Continuous properties were standardized within context; binary mechanistic proxies were coded as present versus absent. Confidence intervals used pairs bootstrap resampling of Bemis-Murcko scaffolds. P values used a restricted-model wild cluster bootstrap with one Rademacher weight per scaffold. Benjamini-Hochberg q values were calculated within each context and across the complete property-context family.",
        "",
        "Consensus rescue and delta probability were retained as reader-facing descriptive and sensitivity analyses. The primary chemical-property outcome was the continuous biological residual score. Subtracting the combined-model intercept prevents cohort recalibration from being mislabeled as biological correction. Three-dimensional conformer descriptors were not included because their values depend on conformer-generation and protonation assumptions not represented in the frozen chemical prior.",
        "",
        "## Results",
        "",
        f"The analysis included {len(contexts)} contexts and {int(contexts['n_active_compounds'].sum())} context-specific active-compound records. Each context contained at least {int(contexts['n_active_scaffolds'].min())} distinct active Bemis-Murcko scaffolds. The archived K=4 ensemble probability was recovered from the Stage-2 raw prior with a maximum absolute error of {contexts['k4_probability_max_abs_reconstruction_error'].max():.2e}.",
        "",
    ]
    if significant.empty:
        lines.extend(
            [
                "No molecular property remained significant at within-context BH q <= 0.05 after conditioning on the frozen ensemble logit and constituent dispersion. This indicates that the transcriptomic rescue was not reducible to a single tested physicochemical descriptor or mechanistic proxy under this post hoc analysis.",
                "",
            ]
        )
    else:
        lines.append(
            f"{len(significant)} property-context associations met within-context BH q <= 0.05. These are conditional associations and should not be interpreted as causal toxicophores or as features used by the chemical ensemble."
        )
        lines.append("")
        for row in significant.sort_values(["context_id", "q_within_context"]).itertuples():
            lines.append(
                f"- {context_label(row.context_id)}: {row.property_label}, adjusted residual-score effect={row.conditional_effect:+.3f} logit units, 95% scaffold-bootstrap CI [{row.cluster_bootstrap_ci_low:+.3f}, {row.cluster_bootstrap_ci_high:+.3f}], wild-bootstrap P={row.wild_cluster_bootstrap_p:.4g}, q={row.q_within_context:.4g}."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "The analysis identifies chemical regimes associated with larger or smaller fitted biological residual corrections beyond a fixed heterogeneous structural ensemble. It does not show that the ensemble used these properties, that the properties caused toxicity, or that adding the properties to a new predictor would improve generalization. SMARTS and physicochemical thresholds are mechanistic proxies; they require external experimental confirmation.",
        ]
    )
    (output_dir / "METHODS_RESULTS_DISCUSSION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    required = [args.table_s7, args.constituent_audit]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required input(s): {missing}")
    tables_dir, figures_dir = prepare_output(args.output_dir)

    contexts = select_contexts(args.table_s7)
    constituent_audit = pd.read_csv(args.constituent_audit)
    all_consensus = []
    all_active = []
    all_tests = []
    context_rows = []
    for row in contexts.to_dict("records"):
        row_series = pd.Series(row)
        consensus, active, tests, _ = analyze_context(
            row_series,
            constituent_audit,
            n_cluster_bootstrap=args.n_cluster_bootstrap,
            n_wild_bootstrap=args.n_wild_bootstrap,
            seed=args.seed,
        )
        all_consensus.append(consensus)
        all_active.append(active)
        all_tests.append(tests)
        context_rows.append(
            {
                "context_id": row["context_id"],
                "family": row["family"],
                "task": row["task"],
                "cell_line": row["lincs_cell_line"],
                "exposure_h": int(row["lincs_exposure_h"]),
                "analysis_cohort": row["analysis_cohort"],
                "n_labeled_compounds": len(consensus),
                "n_active_compounds": len(active),
                "n_active_scaffolds": active["scaffold_key"].nunique(),
                "largest_active_scaffold_size": int(active["scaffold_key"].value_counts().max()),
                "n_consensus_rescued": int(active["consensus_rescued"].sum()),
                "k4_probability_max_abs_reconstruction_error": float(
                    np.max(
                        np.abs(
                            consensus["ensemble_probability"]
                            - consensus["ik"].map(
                                pd.read_csv(source_paths(row_series)[0])
                                .groupby("ik")["p_chem_stage1_raw"]
                                .first()
                            )
                        )
                    )
                ),
                "residual_score_identity_max_abs_error": float(
                    consensus["residual_score_identity_max_abs_error"].max()
                ),
                "delta_auprc": float(row["delta_auprc"]),
                "nominal_p_value": float(row["nominal_p_value"]),
                "bh_q_value": float(row["bh_q_value"]),
            }
        )

    consensus_all = pd.concat(all_consensus, ignore_index=True)
    active_all = pd.concat(all_active, ignore_index=True)
    tests = pd.concat(all_tests, ignore_index=True)
    context_summary = pd.DataFrame(context_rows)

    for column in ["wild_cluster_bootstrap_p", "binary_unadjusted_p"]:
        if column not in tests:
            tests[column] = np.nan
    tests["q_within_context"] = bh_adjust(
        tests, "wild_cluster_bootstrap_p", "context_id"
    )
    tests["q_global"] = bh_adjust(tests, "wild_cluster_bootstrap_p", None)
    tests["binary_q_within_context"] = bh_adjust(
        tests, "binary_unadjusted_p", "context_id"
    )
    tests["binary_q_global"] = bh_adjust(tests, "binary_unadjusted_p", None)

    matched_pairs = match_rescued_compounds(active_all)

    consensus_all.to_csv(tables_dir / "compound_level_all_labeled.csv", index=False)
    active_all.to_csv(tables_dir / "active_compound_property_analysis_data.csv", index=False)
    tests.to_csv(tables_dir / "conditional_property_associations.csv", index=False)
    context_summary.to_csv(tables_dir / "context_analysis_summary.csv", index=False)
    matched_pairs.to_csv(tables_dir / "rescued_active_matched_pairs.csv", index=False)

    plot_prior_plane(active_all, figures_dir / "ensemble_conditioned_rescue_plane", args.dpi)
    plot_conditional_forest(tests, figures_dir / "conditional_property_association_forest", args.dpi)
    selected_pairs = plot_representative_pairs(
        matched_pairs,
        figures_dir / "representative_rescued_active_matched_pairs",
        args.dpi,
    )
    selected_pairs.to_csv(tables_dir / "representative_matched_pairs.csv", index=False)
    write_report(args.output_dir, context_summary, tests)

    manifest = {
        "analysis": "transcriptomic_rescue_chemical_properties_k4_residual_logit_v3",
        "selection_rule": "positive delta AUPRC and nominal NB-corrected P <= 0.05",
        "inferential_unit": "compound",
        "primary_outcome": "mean out-of-fold biological residual score beta^T z across 20 repeats",
        "conditioning": [
            "frozen K=4 ensemble logit",
            "between-constituent probability SD",
        ],
        "inference": {
            "confidence_interval": "pairs bootstrap over Bemis-Murcko scaffolds",
            "p_value": "restricted-model Rademacher wild cluster bootstrap by scaffold",
            "n_cluster_bootstrap": int(args.n_cluster_bootstrap),
            "n_wild_bootstrap": int(args.n_wild_bootstrap),
            "multiple_testing": "BH within context and globally",
        },
        "n_contexts": len(context_summary),
        "context_ids": context_summary["context_id"].tolist(),
        "n_context_specific_active_records": len(active_all),
        "n_property_tests": len(tests),
        "n_estimable_property_tests": int(tests["estimable"].sum()),
        "n_within_context_q_le_0_05": int(
            (tests["estimable"] & tests["q_within_context"].le(0.05)).sum()
        ),
        "n_global_q_le_0_05": int(
            (tests["estimable"] & tests["q_global"].le(0.05)).sum()
        ),
        "max_k4_probability_reconstruction_error": float(
            context_summary["k4_probability_max_abs_reconstruction_error"].max()
        ),
        "max_residual_score_identity_error": float(
            context_summary["residual_score_identity_max_abs_error"].max()
        ),
        "input_sha256": {
            "table_s7": sha256(args.table_s7),
            "constituent_audit": sha256(args.constituent_audit),
        },
        "validation": {
            "all_contexts_use_20_repeats_and_5_outer_folds": True,
            "compound_is_inferential_unit": True,
            "stage1_probability_matches_archived_k4_constituent_audit": True,
            "delta_auprc_reconstructed_from_compound_ap_contributions": True,
            "biological_residual_score_equals_delta_logit_minus_intercept_shift": True,
            "combined_model_intercept_removed_from_biological_residual_score": True,
            "toxicity_predictor_refit": False,
            "ensemble_fragment_attribution_performed": False,
            "causal_toxicophore_claim_supported": False,
        },
    }
    (args.output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    args.table_s7 = args.table_s7.resolve()
    args.constituent_audit = args.constituent_audit.resolve()
    args.output_dir = args.output_dir.resolve()
    run(args)


if __name__ == "__main__":
    main()
