#!/usr/bin/env python3
"""Build an A4 atlas of structural liabilities and pathway enrichment.

The analysis is restricted to endpoint-active compounds in four endpoint-context
combinations with nominal Stage-2 AUPRC improvement. Compound-level pathway
responses are quantified by weighted preranked enrichment of externally defined
gene sets against each 978-gene control-referenced residualized expression
profile. Gene-set permutations provide nominal P values, and BH adjustment is
applied across the four displayed compounds within each endpoint.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import ConnectionPatch, Patch
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem, rdBase
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import fisher_exact, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests


HERE = Path(__file__).resolve().parent
PUBLICATION = HERE / "_run_output" / "publication"
SOURCE_ROOT = PUBLICATION / "chemical_liability_phenotypic_realization_active_source_v1"
SOURCE = SOURCE_ROOT / "tables" / "compound_level_liability_pathway_data.csv"
OUTPUT = PUBLICATION / "publication_structural_pathway_atlas_final_v43"
HALLMARK_GMT = HERE / "resources" / "MSigDB.Hallmark.2020.gmt"
GO_GMT = HERE / "resources" / "Enrichr.GO_Biological_Process_2025.gmt"
PRIMARY_EXPRESSION_ROOT = HERE / "_run_output" / "gene_features_consistent_v1"
PAIRED_EXPRESSION_ROOT = (
    HERE / "_run_output" / "paired_timepoint_inputs_6h24h_8to12uM_v1"
)
GSEA_PERMUTATIONS = 10_000
GSEA_WEIGHT = 1.0
GSEA_SEED = 20260810

CONTEXTS = {
    "NR-Aromatase": "NR-Aromatase__MCF7__24h__paired",
    "SR-ARE": "SR-ARE__HEPG2__6h__primary",
    "SR-MMP": "SR-MMP__MCF7__24h__paired",
    "SR-p53": "SR-p53__MCF7__24h__paired",
}
TASK_ORDER = list(CONTEXTS)
CONTEXT_TITLES = {
    "NR-Aromatase": "NR-Aromatase (MCF7 x 24 h)",
    "SR-ARE": "SR-ARE (HepG2 x 6 h)",
    "SR-MMP": "SR-MMP (MCF7 x 24 h)",
    "SR-p53": "SR-p53 (MCF7 x 24 h)",
}
HEATMAP_CONTEXT_LABELS = {
    "NR-Aromatase": "NR-Arom. | MCF7 24 h",
    "SR-ARE": "SR-ARE | HepG2 6 h",
    "SR-MMP": "SR-MMP | MCF7 24 h",
    "SR-p53": "SR-p53 | MCF7 24 h",
}
ATLAS_COMPOUNDS = {
    "NR-Aromatase": [
        {"ik": "QXHHHPZILQDDPS", "name": "Tioconazole", "pubchem_cid": 5482, "highlight": "azole_like_ring", "highlight_label": "Azole-like ring"},
        {"ik": "BYBLEWFAAKGYCD", "name": "Miconazole", "pubchem_cid": 4189, "highlight": "azole_like_ring", "highlight_label": "Azole-like ring"},
        {"ik": "AFNXATANNDIXLG", "name": "Sulconazole", "pubchem_cid": 5318, "highlight": "azole_like_ring", "highlight_label": "Azole-like ring"},
        {"ik": "UGTJLJZQQFGTJD", "name": "CCCP", "pubchem_cid": 2603, "highlight": "nitrile", "highlight_label": "Nitrile groups"},
    ],
    "SR-ARE": [
        {"ik": "VFLDPWHFBUODDF", "name": "Curcumin", "pubchem_cid": 969516, "highlight": "michael_acceptor", "highlight_label": "Michael acceptor"},
        {"ik": "OMZCMEYTWSXEPZ", "name": "Canertinib", "pubchem_cid": 156414, "highlight": "michael_acceptor", "highlight_label": "Michael acceptor"},
        {"ik": "MJVAVZPDRWSRRC", "name": "Menadione", "pubchem_cid": 4055, "highlight": "michael_acceptor", "highlight_label": "Michael acceptor"},
        {"ik": "REFJWTPEDVJJIY", "name": "Quercetin", "pubchem_cid": 5280343, "highlight": "catechol", "highlight_label": "Catechol-like diphenol"},
    ],
    "SR-MMP": [
        {"ik": "UGTJLJZQQFGTJD", "name": "CCCP", "pubchem_cid": 2603, "highlight": "cccp_protonophore", "highlight_label": "Protonophore-associated core"},
        {"ik": "RJMUSRYZPJIFPJ", "name": "Niclosamide", "pubchem_cid": 4477, "highlight": "nitrophenol_like", "highlight_label": "Nitrophenol-like motif"},
        {"ik": "SIYLLGKDQZGJHK", "name": "Benzethonium", "pubchem_cid": 2335, "highlight": "cationic_amphiphile", "highlight_label": "Cationic amphiphile regions"},
        {"ik": "MDLAAYDRRZXJIF", "name": "Penfluridol", "pubchem_cid": 33630, "highlight": "lipophilic_regime", "highlight_label": "Lipophilic aromatic region"},
    ],
    "SR-p53": [
        {"ik": "MFZWMTSUNYWVBU", "name": "Hycanthone", "pubchem_cid": 3634, "highlight": "aromatic_amine", "highlight_label": "Aromatic amine"},
        {"ik": "RJMUSRYZPJIFPJ", "name": "Niclosamide", "pubchem_cid": 4477, "highlight": "aromatic_nitro", "highlight_label": "Aromatic nitro"},
        {"ik": "NRUKOCRGYNPUPR", "name": "Teniposide", "pubchem_cid": 452548, "highlight": "high_aromatic_burden", "highlight_label": "Aromatic ring system"},
        {"ik": "UCFGDBYHRUNTLO", "name": "Topotecan", "pubchem_cid": 60700, "highlight": "high_aromatic_burden", "highlight_label": "Aromatic ring system"},
    ],
}
for records in ATLAS_COMPOUNDS.values():
    for record in records:
        record["name_source"] = "https://pubchem.ncbi.nlm.nih.gov/compound/{}".format(
            record["pubchem_cid"]
        )

SMARTS = {
    "nitrile": {
        "label": "Nitrile",
        "query": "[C,c]#[N]",
        "definition": "Carbon-nitrogen triple bond.",
    },
    "michael_acceptor": {
        "label": "Michael acceptor",
        "query": "[C,c]=[C,c]-[C,S](=[O,S])-[#6,#7,#8]",
        "definition": "Conjugated carbon-carbon bond adjacent to a carbonyl or thiocarbonyl-like center.",
    },
    "catechol": {
        "label": "Catechol",
        "query": "c1c([OH])c([OH])ccc1",
        "definition": "Adjacent aromatic hydroxyl groups on one six-membered aromatic ring.",
    },
    "hydroquinone": {
        "label": "Hydroquinone",
        "query": "c1c([OH])cc([OH])cc1",
        "definition": "Separated aromatic hydroxyl groups in a hydroquinone-like arrangement.",
    },
    "epoxide_or_aziridine": {
        "label": "Epoxide/aziridine",
        "query": "[O,N;r3]1[C;r3][C;r3]1",
        "definition": "Three-membered oxygen- or nitrogen-containing ring.",
    },
    "aldehyde": {
        "label": "Aldehyde",
        "query": "[CX3H1](=O)[#6]",
        "definition": "Carbonyl carbon bearing one hydrogen and one carbon substituent.",
    },
    "aromatic_nitro": {
        "label": "Aromatic nitro",
        "query": "[c]-[N+](=O)[O-]",
        "definition": "Nitro group directly attached to an aromatic carbon.",
    },
    "phenol": {
        "label": "Phenol",
        "query": "[c]-[OH]",
        "definition": "Hydroxyl group directly attached to an aromatic carbon.",
    },
    "nitrophenol_like": {
        "label": "Nitrophenol-like",
        "query": "([c]-[OH]) AND ([c]-[N+](=O)[O-])",
        "definition": "Compound contains both the phenol and aromatic-nitro SMARTS queries.",
    },
    "aromatic_amine": {
        "label": "Aromatic amine",
        "query": "[c]-[N;H1,H2;v3]",
        "definition": "Primary or secondary amine directly attached to an aromatic carbon.",
    },
    "alkyl_halide": {
        "label": "Alkyl halide",
        "query": "[C;X4]-[Cl,Br,I]",
        "definition": "Tetrahedral carbon bonded to chloride, bromide or iodide.",
    },
    "hydrazine": {
        "label": "Hydrazine",
        "query": "[N;X3]-[N;X3]",
        "definition": "Single bond between two trivalent nitrogen atoms.",
    },
    "aromatic_azo": {
        "label": "Aromatic azo",
        "query": "[c]-[N]=[N]-[c]",
        "definition": "Azo linkage connecting two aromatic carbons.",
    },
    "nitrogen_mustard": {
        "label": "Nitrogen mustard",
        "query": "[N;X3]([CH2][CH2][Cl,Br,I])([CH2][CH2][Cl,Br,I])",
        "definition": "Trivalent nitrogen bearing two beta-haloethyl arms.",
    },
}

CATEGORY_DEFINITIONS = {
    "steroid_like_fused_polycycle": {
        "label": "Steroid-like\npolycycle",
        "evidence_class": "endpoint-associated chemotype proxy",
        "rule": "rings >= 4; fused rings >= 3; aromatic rings <= 1; Fraction Csp3 >= 0.25",
        "source": "Miller et al., 2008; Furet et al., 1993",
        "doi": "10.1634/theoncologist.2008-0055;10.1021/jm00062a012",
        "caution": "Topology proxy, not a formal steroid scaffold matcher.",
    },
    "rigid_polycycle": {
        "label": "Rigid\npolycycle",
        "evidence_class": "endpoint-associated chemotype proxy",
        "rule": "rings >= 3 and rotatable bonds <= 3",
        "source": "Miller et al., 2008",
        "doi": "10.1634/theoncologist.2008-0055",
        "caution": "Broad shape proxy; not aromatase-specific.",
    },
    "azole_like_ring": {
        "label": "Azole-like\nring",
        "evidence_class": "endpoint-associated chemotype",
        "rule": "five-membered aromatic ring containing at least two nitrogen atoms",
        "source": "Furet et al., 1993",
        "doi": "10.1021/jm00062a012",
        "caution": "Azoles can inhibit multiple CYP enzymes; not CYP19A1-specific.",
    },
    "aromatic_nitrogen_any": {
        "label": "Aromatic-N\nproxy",
        "evidence_class": "broad heme-binding proxy",
        "rule": "at least one aromatic nitrogen atom",
        "source": "Furet et al., 1993",
        "doi": "10.1021/jm00062a012",
        "caution": "Broad proxy with low target specificity.",
    },
    "nitrile": {
        "label": "Nitrile",
        "evidence_class": "descriptive structural feature",
        "rule": SMARTS["nitrile"]["query"],
        "source": "Furet et al., 1993",
        "doi": "10.1021/jm00062a012",
        "caution": "Nitrile alone is not a validated aromatase alert.",
    },
    "direct_electrophile": {
        "label": "Direct\nelectrophile",
        "evidence_class": "structural-liability alert",
        "rule": "Michael acceptor OR epoxide/aziridine OR aldehyde",
        "source": "Schultz et al., 2007; Ma and He, 2012",
        "doi": "10.1021/tx700212u;10.1124/pr.110.004333",
        "caution": "Reactivity screen; not a deterministic ARE toxicophore.",
    },
    "redox_diphenol": {
        "label": "Oxidizable\ndiphenol",
        "evidence_class": "redox-liability proxy",
        "rule": "catechol OR hydroquinone-like diphenol",
        "source": "Satoh et al., 2014; Sumi et al., 2009",
        "doi": "10.1016/j.freeradbiomed.2013.11.002;10.2131/jts.34.627",
        "caution": "Potential pro-electrophile/redox proxy; context dependent.",
    },
    "aromatic_nitro": {
        "label": "Aromatic\nnitro",
        "evidence_class": "published structural-alert class",
        "rule": SMARTS["aromatic_nitro"]["query"],
        "source": "Kazius et al., 2005; Benigni and Bossa, 2006",
        "doi": "10.1021/jm040835a;10.2174/157340906777441663",
        "caution": "Metabolism and substitution pattern influence activity.",
    },
    "oxidative_electrophile_any": {
        "label": "Any oxidative/\nelectrophilic",
        "evidence_class": "composite liability proxy",
        "rule": "direct electrophile OR oxidizable diphenol OR aromatic nitro",
        "source": "Schultz et al., 2007; Ma and He, 2012",
        "doi": "10.1021/tx700212u;10.1124/pr.110.004333",
        "caution": "Union of heterogeneous oxidative/electrophilic routes.",
    },
    "cationic_amphiphile": {
        "label": "Cationic\namphiphile",
        "evidence_class": "physicochemical liability proxy",
        "rule": "formal charge > 0; cLogP >= 2; aromatic rings >= 1",
        "source": "Murphy and Smith, 2007; Halliwell, 1997",
        "doi": "10.1146/annurev.pharmtox.47.120505.105110;10.1177/019262339702500111",
        "caution": "May reflect accumulation or phospholipidosis rather than direct depolarization.",
    },
    "high_lipophilicity_low_polarity": {
        "label": "Lipophilic/\nlow polarity",
        "evidence_class": "physicochemical liability proxy",
        "rule": "cLogP >= 3 and TPSA <= 75 square angstrom",
        "source": "Murphy and Smith, 2007",
        "doi": "10.1146/annurev.pharmtox.47.120505.105110",
        "caution": "Broad exposure/accumulation regime, not a toxicophore.",
    },
    "polyaromatic_lipophile": {
        "label": "Polyaromatic\nlipophile",
        "evidence_class": "physicochemical liability proxy",
        "rule": "aromatic rings >= 3 and cLogP >= 3",
        "source": "Murphy and Smith, 2007",
        "doi": "10.1146/annurev.pharmtox.47.120505.105110",
        "caution": "Broad chemical regime, not a direct mitochondrial mechanism.",
    },
    "nitrophenol_protonophore": {
        "label": "Nitrophenol-like\nprotonophore",
        "evidence_class": "mechanistic proxy",
        "rule": "phenol AND aromatic nitro",
        "source": "Kessler et al., 1976; Kotova and Antonenko, 2022",
        "doi": "10.1073/pnas.73.9.3141;10.32607/actanaturae.11610",
        "caution": "Substructure proxy; protonophore activity requires experimental confirmation.",
    },
    "mitochondrial_liability_any": {
        "label": "Any mitochondrial-\nliability proxy",
        "evidence_class": "composite liability proxy",
        "rule": "union of the four displayed mitochondrial proxy rules",
        "source": "Murphy and Smith, 2007; Kotova and Antonenko, 2022",
        "doi": "10.1146/annurev.pharmtox.47.120505.105110;10.32607/actanaturae.11610",
        "caution": "Union of mechanistically heterogeneous liabilities.",
    },
    "dna_reactive_electrophile": {
        "label": "DNA-reactive\nelectrophile",
        "evidence_class": "structural-liability alert",
        "rule": "direct electrophile OR alkyl halide OR nitrogen mustard",
        "source": "Benigni and Bossa, 2006; Enoch and Cronin, 2010",
        "doi": "10.2174/157340906777441663;10.3109/10408444.2010.494175",
        "caution": "Potential DNA-reactivity alert, not direct evidence of adduct formation.",
    },
    "aromatic_amine": {
        "label": "Aromatic\namine",
        "evidence_class": "published genotoxicity-alert class",
        "rule": SMARTS["aromatic_amine"]["query"],
        "source": "Kazius et al., 2005; Purohit and Basu, 2000",
        "doi": "10.1021/jm040835a;10.1021/tx000002x",
        "caution": "Metabolic activation and substitution pattern are important.",
    },
    "hydrazine": {
        "label": "Hydrazine",
        "evidence_class": "published genotoxicity-alert class",
        "rule": SMARTS["hydrazine"]["query"],
        "source": "Benigni and Bossa, 2006",
        "doi": "10.2174/157340906777441663",
        "caution": "Screening alert requiring contextual confirmation.",
    },
    "aromatic_azo": {
        "label": "Aromatic\nazo",
        "evidence_class": "published genotoxicity-alert class",
        "rule": SMARTS["aromatic_azo"]["query"],
        "source": "Chung and Cerniglia, 1992; Benigni and Bossa, 2006",
        "doi": "10.1016/0165-1110(92)90044-A;10.2174/157340906777441663",
        "caution": "Metabolic reduction and substituent context influence activity.",
    },
    "high_aromatic_burden": {
        "label": "High aromatic-\nring burden",
        "evidence_class": "descriptive chemical regime",
        "rule": "aromatic rings >= 3",
        "source": "Kazius et al., 2005",
        "doi": "10.1021/jm040835a",
        "caution": "Not a genotoxicity alert by itself.",
    },
    "genotoxicity_alert_any": {
        "label": "Any genotoxicity\nalert",
        "evidence_class": "composite structural-alert rule",
        "rule": "DNA-reactive electrophile OR aromatic nitro OR aromatic amine OR hydrazine OR aromatic azo",
        "source": "Benigni and Bossa, 2006; Kazius et al., 2005",
        "doi": "10.2174/157340906777441663;10.1021/jm040835a",
        "caution": "Composite screening rule, not a p53-specific toxicophore.",
    },
}

ENDPOINT_CATEGORIES = {
    "NR-Aromatase": ["steroid_like_fused_polycycle", "rigid_polycycle", "azole_like_ring", "aromatic_nitrogen_any", "nitrile"],
    "SR-ARE": ["direct_electrophile", "redox_diphenol", "aromatic_nitro"],
    "SR-MMP": ["cationic_amphiphile", "high_lipophilicity_low_polarity", "polyaromatic_lipophile", "nitrophenol_protonophore"],
    "SR-p53": ["dna_reactive_electrophile", "aromatic_nitro", "aromatic_amine", "hydrazine", "aromatic_azo", "high_aromatic_burden"],
}

INK = "#222222"
GRID = "#D8D8D8"
OTHER = "#B9B9B9"
CONSENSUS = "#E7A45D"
STABLE = "#B64E4B"
REPRESENTATIVE = "#A62320"
MOTIF_COLOR = "#C62828"

PATHWAY_DEFINITIONS = {
    "NR-Aromatase": {
        "gmt": HALLMARK_GMT,
        "gene_set": "Estrogen Response Early",
        "pathway_label": "Hallmark early estrogen response",
    },
    "SR-ARE": {
        "gmt": HALLMARK_GMT,
        "gene_set": "Reactive Oxygen Species Pathway",
        "pathway_label": "Hallmark reactive oxygen species pathway",
    },
    "SR-MMP": {
        "gmt": HALLMARK_GMT,
        "gene_set": "Unfolded Protein Response",
        "pathway_label": "Hallmark unfolded protein response",
    },
    "SR-p53": {
        "gmt": HALLMARK_GMT,
        "gene_set": "p53 Pathway",
        "pathway_label": "Hallmark p53 pathway",
    },
}

POSITIVE_COLORS = {
    "bh": "#B2182B",
    "none": "#FDD1C6",
}
NEGATIVE_COLORS = {
    "bh": "#2166AC",
    "none": "#D1E5F0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output() -> tuple[Path, Path]:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    tables = OUTPUT / "tables"
    figures = OUTPUT / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    return tables, figures


def panel_letter(axis: plt.Axes, letter: str, x: float = -0.08, y: float = 1.08) -> None:
    axis.text(
        x,
        y,
        letter,
        transform=axis.transAxes,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def benjamini_hochberg(values: pd.Series) -> np.ndarray:
    result = np.full(len(values), np.nan)
    valid = values.notna().to_numpy()
    if valid.any():
        result[valid] = multipletests(values.to_numpy(float)[valid], method="fdr_bh")[1]
    return result


def active_context_data(compounds: pd.DataFrame) -> pd.DataFrame:
    selected = compounds.loc[compounds["context_id"].isin(CONTEXTS.values())].copy()
    if set(selected["context_id"]) != set(CONTEXTS.values()):
        missing = set(CONTEXTS.values()) - set(selected["context_id"])
        raise ValueError(f"Missing representative contexts: {sorted(missing)}")
    selected["improvement_group"] = "No consensus rescue"
    selected.loc[selected["consensus_rescued"].eq(True), "improvement_group"] = "Consensus rescue"
    selected.loc[selected["stable_rescued"].eq(True), "improvement_group"] = "Stable rescue"
    return selected


def load_gmt(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3:
            result[fields[0]] = [gene for gene in fields[2:] if gene]
    return result


def expression_path(task: str, frame: pd.DataFrame) -> Path:
    cell = str(frame["cell_line"].iloc[0])
    exposure = int(frame["exposure_h"].iloc[0])
    cohort = str(frame["cohort_code"].iloc[0])
    if cohort == "primary":
        if exposure != 6:
            raise ValueError(f"{task}: primary expression source is defined only for 6 h")
        return PRIMARY_EXPRESSION_ROOT / f"{cell}_gene_features_residualized.csv"
    return (
        PAIRED_EXPRESSION_ROOT
        / f"gene_features_{exposure}h"
        / f"{cell}_gene_features_residualized.csv"
    )


def stable_seed(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return (GSEA_SEED + int(digest[:8], 16)) % (2**32 - 1)


def enrichment_score(
    ranked_metric: np.ndarray,
    hit_mask: np.ndarray,
) -> tuple[float, int, np.ndarray]:
    ranked_metric = np.asarray(ranked_metric, dtype=float)
    hit_mask = np.asarray(hit_mask, dtype=bool)
    n_genes = ranked_metric.size
    n_hits = int(hit_mask.sum())
    if n_hits == 0 or n_hits == n_genes:
        raise ValueError(f"Invalid gene-set size {n_hits} for ranked list of {n_genes}")
    hit_weights = np.power(np.abs(ranked_metric), GSEA_WEIGHT)
    normalizer = float(hit_weights[hit_mask].sum())
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("Pathway genes have zero or invalid total rank weight")
    increments = np.where(
        hit_mask,
        hit_weights / normalizer,
        -1.0 / float(n_genes - n_hits),
    )
    running = np.cumsum(increments)
    positive_index = int(np.argmax(running))
    negative_index = int(np.argmin(running))
    if running[positive_index] >= -running[negative_index]:
        return float(running[positive_index]), positive_index, running
    return float(running[negative_index]), negative_index, running


def preranked_gene_set_enrichment(
    expression: pd.Series,
    pathway_genes: set[str],
    seed: int,
) -> dict[str, object]:
    values = expression.to_numpy(dtype=float)
    genes = expression.index.to_numpy(dtype=str)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite expression value in preranked profile")
    order = np.argsort(-values, kind="stable")
    ranked_metric = values[order]
    ranked_genes = genes[order]
    hit_mask = np.isin(ranked_genes, list(pathway_genes))
    observed_es, extremum_index, _ = enrichment_score(ranked_metric, hit_mask)

    n_genes = ranked_metric.size
    n_hits = int(hit_mask.sum())
    rng = np.random.default_rng(seed)
    null_es = np.empty(GSEA_PERMUTATIONS, dtype=float)
    absolute_metric = np.power(np.abs(ranked_metric), GSEA_WEIGHT)
    miss_penalty = -1.0 / float(n_genes - n_hits)
    chunk_size = 500
    for start in range(0, GSEA_PERMUTATIONS, chunk_size):
        stop = min(start + chunk_size, GSEA_PERMUTATIONS)
        size = stop - start
        random_hits = np.zeros((size, n_genes), dtype=bool)
        for row_index in range(size):
            selected = rng.choice(n_genes, size=n_hits, replace=False)
            random_hits[row_index, selected] = True
        denominators = (random_hits * absolute_metric[None, :]).sum(axis=1)
        increments = np.where(
            random_hits,
            absolute_metric[None, :] / denominators[:, None],
            miss_penalty,
        )
        running = np.cumsum(increments, axis=1)
        maxima = running.max(axis=1)
        minima = running.min(axis=1)
        null_es[start:stop] = np.where(maxima >= -minima, maxima, minima)

    if observed_es >= 0:
        same_sign = null_es >= 0
        extreme = null_es >= observed_es
        scale = float(null_es[same_sign].mean())
        leading_mask = hit_mask & (np.arange(n_genes) <= extremum_index)
    else:
        same_sign = null_es < 0
        extreme = null_es <= observed_es
        scale = float(abs(null_es[same_sign].mean()))
        leading_mask = hit_mask & (np.arange(n_genes) >= extremum_index)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid same-sign null ES normalization factor")
    nominal_p = float((1 + np.sum(extreme & same_sign)) / (1 + np.sum(same_sign)))
    leading_edge = ranked_genes[leading_mask].tolist()
    return {
        "es": observed_es,
        "nes": observed_es / scale,
        "nominal_p": nominal_p,
        "leading_edge_genes": ";".join(leading_edge),
        "leading_edge_size": len(leading_edge),
        "gene_set_size_measured": n_hits,
        "ranked_gene_count": n_genes,
        "n_gene_set_permutations": GSEA_PERMUTATIONS,
        "permutation_seed": seed,
        "null_positive_n": int((null_es >= 0).sum()),
        "null_negative_n": int((null_es < 0).sum()),
    }


def build_preranked_statistics(compounds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gmt_paths = {definition["gmt"] for definition in PATHWAY_DEFINITIONS.values()}
    gmt_cache = {path: load_gmt(path) for path in gmt_paths}
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for task in TASK_ORDER:
        context = compounds.loc[compounds["context_id"].eq(CONTEXTS[task])].copy()
        path = expression_path(task, context)
        if not path.is_file():
            raise FileNotFoundError(path)
        expression = pd.read_csv(path)
        if expression["ik"].duplicated().any():
            raise ValueError(f"Duplicate expression compound IDs: {path}")
        expression = expression.set_index("ik")
        definition = PATHWAY_DEFINITIONS[task]
        requested = gmt_cache[definition["gmt"]][definition["gene_set"]]
        available = set(requested).intersection(expression.columns)
        if len(available) < 5:
            raise ValueError(f"{task}: fewer than five measured pathway genes")
        audit_rows.append(
            {
                "task": task,
                "context_id": CONTEXTS[task],
                "expression_path": str(path),
                "expression_sha256": sha256(path),
                "gmt_path": str(definition["gmt"]),
                "gmt_sha256": sha256(definition["gmt"]),
                "gene_set_name": definition["gene_set"],
                "pathway_label": definition["pathway_label"],
                "gene_set_size_source": len(requested),
                "gene_set_size_measured": len(available),
                "ranked_gene_count": expression.shape[1],
            }
        )
        for compound in ATLAS_COMPOUNDS[task]:
            ik = compound["ik"]
            if ik not in expression.index:
                raise ValueError(f"{task}: expression missing for {ik}")
            result = preranked_gene_set_enrichment(
                expression.loc[ik],
                available,
                stable_seed(task, ik, definition["gene_set"]),
            )
            rows.append(
                {
                    "task": task,
                    "context_id": CONTEXTS[task],
                    "ik": ik,
                    "drug_name": compound["name"],
                    "pathway_label": definition["pathway_label"],
                    "gene_set_name": definition["gene_set"],
                    **result,
                }
            )
    result = pd.DataFrame(rows)
    result["bh_q"] = result.groupby("task", sort=False)["nominal_p"].transform(
        lambda values: benjamini_hochberg(values)
    )
    return result, pd.DataFrame(audit_rows)


def enrichment_color(nes: float, nominal_p: float, bh_q: float) -> str:
    significance = "bh" if bh_q <= 0.05 else "none"
    palette = POSITIVE_COLORS if nes >= 0 else NEGATIVE_COLORS
    return palette[significance]


def q_significance_marker(q_value: float) -> str:
    if q_value <= 0.001:
        return "***"
    if q_value <= 0.01:
        return "**"
    if q_value <= 0.05:
        return "*"
    return ""


def plot_probability_scatter(fig: plt.Figure, spec, compounds: pd.DataFrame) -> None:
    inner = spec.subgridspec(2, 4, height_ratios=[0.58, 1.0], hspace=0.48, wspace=0.32)
    heading = fig.add_subplot(inner[0, :])
    heading.axis("off")
    panel_letter(heading, "A", x=-0.10, y=1.00)
    heading.text(
        0,
        1.0,
        "Active-compound probability rescue",
        transform=heading.transAxes,
        fontsize=12.2,
        fontweight="bold",
        ha="left",
        va="top",
    )
    axes = []
    for index, task in enumerate(TASK_ORDER):
        axis = fig.add_subplot(inner[1, index])
        axes.append(axis)
        frame = compounds.loc[compounds["context_id"].eq(CONTEXTS[task])].copy()
        other = frame.loc[~frame["consensus_rescued"].eq(True)]
        consensus = frame.loc[frame["consensus_rescued"].eq(True) & ~frame["stable_rescued"].eq(True)]
        stable = frame.loc[frame["stable_rescued"].eq(True)]
        axis.scatter(
            other["mean_p_chem"],
            other["mean_p_bio"],
            s=18,
            facecolor="#D0D0D0",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.76,
            zorder=2,
        )
        axis.scatter(
            consensus["mean_p_chem"],
            consensus["mean_p_bio"],
            s=31,
            facecolor=CONSENSUS,
            edgecolor=INK,
            linewidth=0.55,
            alpha=0.92,
            zorder=3,
        )
        axis.scatter(
            stable["mean_p_chem"],
            stable["mean_p_bio"],
            s=38,
            facecolor=STABLE,
            edgecolor=INK,
            linewidth=0.65,
            alpha=0.95,
            zorder=4,
        )
        displayed = frame.loc[frame["ik"].isin([item["ik"] for item in ATLAS_COMPOUNDS[task]])]
        if len(displayed) != len(ATLAS_COMPOUNDS[task]):
            raise ValueError(f"Missing displayed compounds for {task}")
        axis.scatter(
            displayed["mean_p_chem"],
            displayed["mean_p_bio"],
            marker="*",
            s=78,
            facecolor="#FFF4F1",
            edgecolor=REPRESENTATIVE,
            linewidth=0.95,
            zorder=6,
        )
        axis.plot([0, 1], [0, 1], color="#777777", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
        axis.axvline(0.5, color=GRID, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
        axis.axhline(0.5, color=GRID, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([0, 0.5, 1.0])
        axis.set_yticks([0, 0.5, 1.0])
        axis.tick_params(labelsize=8.0, width=0.8, length=3)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(
            CONTEXT_TITLES[task]
            + f"\nrescued {int(frame['consensus_rescued'].sum())}/{len(frame)}",
            fontsize=8.6,
            fontweight="bold",
            pad=5,
        )
        axis.set_xlabel("Chem.-only probability", fontsize=8.2, fontweight="bold", labelpad=2)
        if index == 0:
            axis.set_ylabel(
                "Chem. + transcriptomics probability",
                fontsize=8.2,
                fontweight="bold",
                labelpad=0,
            )
        else:
            axis.set_yticklabels([])
    heading.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markerfacecolor="#D0D0D0", markeredgecolor="white", markersize=6, label="Other active"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor=CONSENSUS, markeredgecolor=INK, markersize=6, label="Consensus rescue"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor=STABLE, markeredgecolor=INK, markersize=6, label="Stable rescue"),
            Line2D([], [], marker="*", linestyle="", markerfacecolor="#FFF4F1", markeredgecolor=REPRESENTATIVE, markersize=8, label="Displayed compounds"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.02),
        ncol=2,
        frameon=False,
        fontsize=6.8,
        columnspacing=0.8,
        handletextpad=0.35,
    )


def smart_alert_enrichment(compounds: pd.DataFrame) -> pd.DataFrame:
    records = []
    for task in TASK_ORDER:
        frame = compounds.loc[compounds["context_id"].eq(CONTEXTS[task])].copy()
        improved = frame["consensus_rescued"].eq(True)
        for alert_id in ENDPOINT_CATEGORIES[task]:
            if alert_id not in frame.columns:
                raise ValueError(f"Missing source category {alert_id}")
            present = frame[alert_id].eq(True)
            a = int((improved & present).sum())
            b = int((improved & ~present).sum())
            c = int((~improved & present).sum())
            d = int((~improved & ~present).sum())
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            corrected_or = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
            records.append(
                {
                    "task": task,
                    "context_id": CONTEXTS[task],
                    "alert_id": alert_id,
                    "alert_label": CATEGORY_DEFINITIONS[alert_id]["label"].replace("\n", " "),
                    "exact_rule": CATEGORY_DEFINITIONS[alert_id]["rule"],
                    "evidence_class": CATEGORY_DEFINITIONS[alert_id]["evidence_class"],
                    "n_improved": int(improved.sum()),
                    "n_no_improvement": int((~improved).sum()),
                    "improved_alert_positive": a,
                    "improved_alert_negative": b,
                    "no_improvement_alert_positive": c,
                    "no_improvement_alert_negative": d,
                    "fisher_odds_ratio": float(odds_ratio),
                    "half_cell_corrected_odds_ratio": float(corrected_or),
                    "log2_improved_vs_no_improvement": float(np.log2(corrected_or)),
                    "nominal_p": float(p_value),
                    "estimable": bool(a + c > 0),
                }
            )
    result = pd.DataFrame(records)
    result["bh_q_within_endpoint"] = np.nan
    for _, indices in result.groupby("task").groups.items():
        result.loc[indices, "bh_q_within_endpoint"] = benjamini_hochberg(result.loc[indices, "nominal_p"])
    result["bh_q_global"] = benjamini_hochberg(result["nominal_p"])
    return result


def plot_alert_heatmap(fig: plt.Figure, spec, enrichment: pd.DataFrame) -> None:
    inner = spec.subgridspec(5, 1, height_ratios=[0.92, 1, 1, 1, 1], hspace=0.62)
    heading = fig.add_subplot(inner[0, 0])
    heading.axis("off")
    panel_letter(heading, "B", x=-0.10, y=1.02)
    heading.text(
        0,
        1.0,
        "Chemical-liability enrichment in rescued actives",
        transform=heading.transAxes,
        fontsize=12.2,
        fontweight="bold",
        ha="left",
        va="top",
    )
    all_values = enrichment.loc[enrichment["estimable"], "log2_improved_vs_no_improvement"].to_numpy(float)
    limit = max(3.0, float(np.nanquantile(np.abs(all_values), 0.95)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    images = []
    axes = []
    for index, task in enumerate(TASK_ORDER):
        axis = fig.add_subplot(inner[index + 1, 0])
        axes.append(axis)
        category_ids = ENDPOINT_CATEGORIES[task]
        subset = enrichment.loc[enrichment["task"].eq(task)].set_index("alert_id")
        values = np.array([[subset.loc[item, "log2_improved_vs_no_improvement"] for item in category_ids]])
        image = axis.imshow(values, aspect="auto", cmap="RdBu_r", norm=norm)
        images.append(image)
        axis.set_xticks(range(len(category_ids)), [CATEGORY_DEFINITIONS[item]["label"] for item in category_ids], fontsize=7.8)
        axis.set_yticks([0], [HEATMAP_CONTEXT_LABELS[task]], fontsize=7.9)
        axis.tick_params(axis="x", length=0, pad=2)
        axis.tick_params(axis="y", length=0, pad=5)
        for j, category_id in enumerate(category_ids):
            row = subset.loc[category_id]
            symbol = "†" if row["bh_q_within_endpoint"] <= 0.05 else ("*" if row["nominal_p"] <= 0.05 else "")
            color = "white" if abs(row["log2_improved_vs_no_improvement"]) > 0.56 * limit else INK
            axis.text(
                j,
                0,
                f"{row['log2_improved_vs_no_improvement']:+.1f}{symbol}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=color,
                fontweight="bold",
            )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color(INK)
    cax = heading.inset_axes([0.72, 0.02, 0.27, 0.14])
    colorbar = fig.colorbar(images[-1], cax=cax, orientation="horizontal")
    colorbar.set_label(r"$\log_2$ odds ratio", fontsize=7.2, fontweight="bold", labelpad=1)
    colorbar.ax.tick_params(labelsize=6.8, pad=1, length=2)
    heading.text(
        0.69,
        0.03,
        "* nominal Fisher P <= 0.05; † within-endpoint BH q <= 0.05",
        transform=heading.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#4B4B4B",
    )


def effect_value(x: np.ndarray, y: np.ndarray) -> float:
    if np.unique(x).size < 3 or np.unique(y).size < 3:
        return np.nan
    return float(spearmanr(x, y).statistic)


def scaffold_spearman_test(
    frame: pd.DataFrame,
    n_bootstrap: int,
    n_wild: int,
    seed: int,
) -> dict[str, float | int]:
    data = frame[["anchor_gene_expression", "mean_biological_residual_score", "scaffold_key"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    x = data["anchor_gene_expression"].to_numpy(float)
    y = data["mean_biological_residual_score"].to_numpy(float)
    scaffolds = data["scaffold_key"].astype(str).to_numpy()
    groups = [np.flatnonzero(scaffolds == value) for value in pd.unique(scaffolds)]
    observed = effect_value(x, y)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        value = effect_value(x[indices], y[indices])
        if np.isfinite(value):
            bootstrap.append(value)
    if len(bootstrap) < max(200, int(0.8 * n_bootstrap)):
        raise RuntimeError("Insufficient valid scaffold-bootstrap replicates")
    y_rank = rankdata(y, method="average")
    fitted_null = np.repeat(y_rank.mean(), len(y_rank))
    residual = y_rank - fitted_null
    wild = []
    for _ in range(n_wild):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(groups))
        y_star = fitted_null.copy()
        for sign, indices in zip(signs, groups):
            y_star[indices] += sign * residual[indices]
        value = effect_value(x, y_star)
        if np.isfinite(value):
            wild.append(value)
    wild_array = np.asarray(wild, dtype=float)
    p_value = float((1 + np.sum(np.abs(wild_array) >= abs(observed))) / (len(wild_array) + 1))
    ci_low, ci_high = np.quantile(np.asarray(bootstrap), [0.025, 0.975])
    return {
        "n_compounds": len(data),
        "n_scaffolds": len(groups),
        "spearman_rho": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "nominal_p": p_value,
        "n_bootstrap_valid": len(bootstrap),
        "n_wild_valid": len(wild_array),
    }


def anchor_gene_statistics(compounds: pd.DataFrame) -> pd.DataFrame:
    records = []
    for index, task in enumerate(TASK_ORDER):
        frame = compounds.loc[compounds["context_id"].eq(CONTEXTS[task])]
        result = scaffold_spearman_test(frame, n_bootstrap=2000, n_wild=5000, seed=20260810 + 1009 * index)
        result.update(
            {
                "task": task,
                "context_id": CONTEXTS[task],
                "cell_line": frame["cell_line"].iloc[0],
                "exposure_h": int(frame["exposure_h"].iloc[0]),
                "anchor_gene": frame["anchor_gene"].iloc[0],
                "x": "control-referenced viability-adjusted anchor-gene expression",
                "y": "mean fitted transcriptomic residual logit",
                "ci_method": "pairs bootstrap of Bemis-Murcko scaffolds",
                "p_method": "restricted-null Rademacher wild bootstrap by Bemis-Murcko scaffold",
            }
        )
        records.append(result)
    result = pd.DataFrame(records)
    result["bh_q"] = benjamini_hochberg(result["nominal_p"])
    return result


def azole_atoms(mol: Chem.Mol) -> list[int]:
    for ring in mol.GetRingInfo().AtomRings():
        atoms = [mol.GetAtomWithIdx(index) for index in ring]
        if len(ring) == 5 and all(atom.GetIsAromatic() for atom in atoms) and sum(atom.GetAtomicNum() == 7 for atom in atoms) >= 2:
            return list(ring)
    return []


def matched_atoms(mol: Chem.Mol, smarts_id: str) -> set[int]:
    query_text = SMARTS[smarts_id]["query"]
    if " AND " in query_text:
        atoms: set[int] = set()
        for component in ("phenol", "aromatic_nitro"):
            atoms.update(matched_atoms(mol, component))
        return atoms
    query = Chem.MolFromSmarts(query_text)
    if query is None:
        raise ValueError(f"Invalid SMARTS for {smarts_id}: {query_text}")
    return {atom for match in mol.GetSubstructMatches(query) for atom in match}


def atlas_highlight_atoms(mol: Chem.Mol, highlight_id: str) -> list[int]:
    if highlight_id == "azole_like_ring":
        return azole_atoms(mol)
    if highlight_id in SMARTS:
        return sorted(matched_atoms(mol, highlight_id))
    if highlight_id == "cccp_protonophore":
        core = matched_atoms(mol, "nitrile") | matched_atoms(mol, "hydrazine")
        expanded = set(core)
        for atom_index in core:
            expanded.update(neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors())
        return sorted(expanded)
    if highlight_id == "cationic_amphiphile":
        charged = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0}
        aromatic = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()}
        return sorted(charged | aromatic)
    if highlight_id in {"lipophilic_regime", "high_aromatic_burden"}:
        return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    raise ValueError(f"Unsupported highlight mode: {highlight_id}")


def molecule_image(
    smiles: str,
    highlight_atoms: list[int],
    highlight_color: str = REPRESENTATIVE,
) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.rdDepictor.Compute2DCoords(mol)
    highlighted = set(highlight_atoms)
    bonds = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in highlighted and bond.GetEndAtomIdx() in highlighted
    ]
    red, green, blue = to_rgb(highlight_color)
    atom_colors = {index: (red, green, blue, 0.28) for index in highlighted}
    bond_colors = {index: (red, green, blue, 0.28) for index in bonds}
    drawer = rdMolDraw2D.MolDraw2DCairo(2800, 1500)
    options = drawer.drawOptions()
    options.padding = 0.025
    options.bondLineWidth = 4.2
    options.highlightBondWidthMultiplier = 10
    options.scalingFactor = 32.0
    options.fixedFontSize = 144
    options.minFontSize = 24
    options.maxFontSize = 144
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlighted),
        highlightBonds=bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    pixels = np.asarray(image)
    mask = np.any(pixels < 245, axis=2)
    if mask.any():
        y, x = np.where(mask)
        padding = 40
        box = (
            max(0, int(x.min()) - padding),
            max(0, int(y.min()) - padding),
            min(image.width, int(x.max()) + padding),
            min(image.height, int(y.max()) + padding),
        )
        image = image.crop(box)
    return image


def format_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def significance_marker(p_value: float, q_value: float) -> str:
    if q_value <= 0.001:
        return "***"
    if q_value <= 0.01:
        return "**"
    if q_value <= 0.05:
        return "*"
    if p_value <= 0.05:
        return "†"
    return ""


def plot_structure_atlas_panel(
    fig: plt.Figure,
    spec,
    task: str,
    letter: str,
    compounds: pd.DataFrame,
    statistics: pd.DataFrame,
) -> list[pd.Series]:
    outer = spec.subgridspec(2, 4, height_ratios=[0.34, 1.0], hspace=0.02, wspace=0.07)
    heading = fig.add_subplot(outer[0, 0])
    heading.remove()
    heading = fig.add_subplot(outer[0, :])
    heading.axis("off")
    panel_letter(heading, letter, x=-0.10, y=1.02)
    frame = compounds.loc[compounds["context_id"].eq(CONTEXTS[task])].copy()
    stat = statistics.loc[statistics["task"].eq(task)].iloc[0]
    marker = significance_marker(stat["nominal_p"], stat["bh_q"])
    heading.text(
        0.0,
        1.0,
        CONTEXT_TITLES[task].replace("\n", " | "),
        transform=heading.transAxes,
        fontsize=10.2,
        fontweight="bold",
        ha="left",
        va="top",
    )
    heading.text(
        0.0,
        0.28,
        f"{stat['anchor_gene']}-residual association: rho={stat['spearman_rho']:+.2f}{marker}; "
        f"95% CI [{stat['ci_low']:+.2f}, {stat['ci_high']:+.2f}]; P={format_p(stat['nominal_p'])}; q={format_p(stat['bh_q'])}",
        transform=heading.transAxes,
        fontsize=7.4,
        ha="left",
        va="bottom",
        color="#4A4A4A",
    )
    rows = []
    for index, compound in enumerate(ATLAS_COMPOUNDS[task]):
        selected = frame.loc[frame["ik"].eq(compound["ik"])]
        if len(selected) != 1:
            raise ValueError(f"Expected one compound for {task}/{compound['ik']}; found {len(selected)}")
        row = selected.iloc[0].copy()
        mol = Chem.MolFromSmiles(row["smiles_canon"])
        if mol is None:
            raise ValueError(f"Invalid SMILES for {compound['name']}")
        atoms = atlas_highlight_atoms(mol, compound["highlight"])
        if not atoms:
            raise ValueError(f"No highlight atoms for {compound['name']} ({compound['highlight']})")
        cell = outer[1, index].subgridspec(2, 1, height_ratios=[0.70, 0.30], hspace=0.01)
        image_axis = fig.add_subplot(cell[0, 0])
        image_axis.imshow(molecule_image(row["smiles_canon"], atoms))
        image_axis.axis("off")
        text_axis = fig.add_subplot(cell[1, 0])
        text_axis.axis("off")
        text_axis.text(0.5, 0.98, compound["name"], transform=text_axis.transAxes, ha="center", va="top", fontsize=8.2, fontweight="bold")
        text_axis.text(0.5, 0.68, compound["highlight_label"], transform=text_axis.transAxes, ha="center", va="top", fontsize=7.0, fontweight="bold", color=REPRESENTATIVE)
        text_axis.text(
            0.5,
            0.34,
            f"p {row['mean_p_chem']:.2f} -> {row['mean_p_bio']:.2f} | Delta p {row['mean_delta_probability']:+.2f}\n{row['anchor_gene']} {row['anchor_gene_expression']:+.2f}",
            transform=text_axis.transAxes,
            ha="center",
            va="top",
            fontsize=6.2,
            color="#444444",
            linespacing=1.05,
        )
        row["drug_name"] = compound["name"]
        row["pubchem_cid"] = compound["pubchem_cid"]
        row["drug_name_source"] = compound["name_source"]
        row["highlight_id"] = compound["highlight"]
        row["highlight_label"] = compound["highlight_label"]
        row["highlight_atom_indices"] = ";".join(str(value) for value in atoms)
        rows.append(row)
    return rows


def pathway_axis_label() -> str:
    return "Pathway\nenrichment"


def _plot_endpoint_ranked_panel_v29(
    fig: plt.Figure,
    spec,
    task: str,
    letter: str,
    compounds: pd.DataFrame,
    gsea_statistics: pd.DataFrame,
) -> list[pd.Series]:
    outer = spec.subgridspec(
        2,
        4,
        height_ratios=[0.12, 1.0],
        width_ratios=[0.80, 0.68, 0.80, 2.10],
        hspace=0.02,
        wspace=0.055,
    )
    heading = fig.add_subplot(outer[0, :])
    heading.axis("off")
    panel_letter(heading, letter, x=-0.115, y=1.05)
    frame = compounds.loc[compounds["context_id"].eq(CONTEXTS[task])].copy()
    heading.text(
        0.5,
        1.0,
        CONTEXT_TITLES[task],
        transform=heading.transAxes,
        fontsize=9.2,
        fontweight="bold",
        ha="center",
        va="top",
    )

    scatter_axis = fig.add_subplot(outer[1, 0])
    other = frame.loc[~frame["consensus_rescued"].eq(True)]
    consensus = frame.loc[frame["consensus_rescued"].eq(True) & ~frame["stable_rescued"].eq(True)]
    stable = frame.loc[frame["stable_rescued"].eq(True)]
    scatter_axis.scatter(other["mean_p_chem"], other["mean_p_bio"], s=4, color="#D0D0D0", alpha=0.50, linewidth=0, zorder=2)
    scatter_axis.scatter(consensus["mean_p_chem"], consensus["mean_p_bio"], s=8, facecolor=CONSENSUS, edgecolor=INK, linewidth=0.30, alpha=0.90, zorder=3)
    scatter_axis.scatter(stable["mean_p_chem"], stable["mean_p_bio"], s=10, facecolor=STABLE, edgecolor=INK, linewidth=0.35, alpha=0.94, zorder=4)
    displayed = frame.loc[frame["ik"].isin([item["ik"] for item in ATLAS_COMPOUNDS[task]])]
    displayed_consensus = displayed.loc[~displayed["stable_rescued"].eq(True)]
    displayed_stable = displayed.loc[displayed["stable_rescued"].eq(True)]
    scatter_axis.scatter(
        displayed_consensus["mean_p_chem"],
        displayed_consensus["mean_p_bio"],
        marker="o",
        s=34,
        facecolor=CONSENSUS,
        edgecolor=INK,
        linewidth=1.05,
        zorder=20,
    )
    scatter_axis.scatter(
        displayed_stable["mean_p_chem"],
        displayed_stable["mean_p_bio"],
        marker="o",
        s=34,
        facecolor=STABLE,
        edgecolor=INK,
        linewidth=1.05,
        zorder=20,
    )
    scatter_axis.plot([0, 1], [0, 1], color="#777777", linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    scatter_axis.axvline(0.5, color=GRID, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    scatter_axis.axhline(0.5, color=GRID, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    scatter_axis.set_xlim(0, 1)
    scatter_axis.set_ylim(0, 1)
    scatter_axis.set_aspect("equal", adjustable="box")
    scatter_axis.set_xticks([0, 0.5, 1.0])
    scatter_axis.set_yticks([0, 0.5, 1.0])
    scatter_axis.tick_params(labelsize=6.7, width=0.7, length=2.5, pad=1.5)
    scatter_axis.spines[["top", "right"]].set_visible(False)
    scatter_axis.set_xlabel("Chem.-only probability", fontsize=7.0, fontweight="bold", labelpad=1.5)
    scatter_axis.set_ylabel("Chem. + transcriptomics\nprobability", fontsize=7.0, fontweight="bold", labelpad=1.5)
    scatter_axis.text(
        0.97,
        0.97,
        f"rescued {int(frame['consensus_rescued'].sum())}/{len(frame)}",
        transform=scatter_axis.transAxes,
        fontsize=6.2,
        fontweight="bold",
        ha="left",
        va="top",
    )
    scatter_axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markerfacecolor="#D0D0D0", markeredgecolor="none", markersize=3.6, label="Other active"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor=CONSENSUS, markeredgecolor=INK, markersize=3.6, label="Consensus rescue"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor=STABLE, markeredgecolor=INK, markersize=3.6, label="Stable rescue"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor=STABLE, markeredgecolor=INK, markeredgewidth=1.0, markersize=5.0, label="Displayed rescue"),
        ],
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#B8B8B8",
        fontsize=4.25,
        borderpad=0.28,
        labelspacing=0.20,
        handletextpad=0.24,
    )

    metadata = {item["ik"]: item for item in ATLAS_COMPOUNDS[task]}
    selected_rows = []
    for compound in ATLAS_COMPOUNDS[task]:
        selected = frame.loc[frame["ik"].eq(compound["ik"])]
        if len(selected) != 1:
            raise ValueError(f"Expected one compound for {task}/{compound['ik']}; found {len(selected)}")
        row = selected.iloc[0].copy()
        row["drug_name"] = compound["name"]
        row["pubchem_cid"] = compound["pubchem_cid"]
        row["drug_name_source"] = compound["name_source"]
        row["highlight_id"] = compound["highlight"]
        row["highlight_label"] = compound["highlight_label"]
        gsea = gsea_statistics.loc[
            gsea_statistics["task"].eq(task) & gsea_statistics["ik"].eq(compound["ik"])
        ]
        if len(gsea) != 1:
            raise ValueError(f"Expected one GSEA row for {task}/{compound['ik']}; found {len(gsea)}")
        for key, value in gsea.iloc[0].items():
            if key not in {"task", "context_id", "ik", "drug_name"}:
                row[key] = value
        selected_rows.append(row)
    selected_rows.sort(key=lambda value: abs(float(value["nes"])), reverse=True)

    label_grid = outer[1, 1].subgridspec(4, 1, hspace=0.025)
    structure_grid = outer[1, 3].subgridspec(4, 1, hspace=0.015)
    structure_axes = []
    exported = []
    for rank, row in enumerate(selected_rows, start=1):
        compound = metadata[row["ik"]]
        mol = Chem.MolFromSmiles(row["smiles_canon"])
        atoms = atlas_highlight_atoms(mol, compound["highlight"])
        if not atoms:
            raise ValueError(f"No highlight atoms for {compound['name']}")
        image_axis = fig.add_subplot(structure_grid[rank - 1, 0])
        image_axis.imshow(
            molecule_image(row["smiles_canon"], atoms, MOTIF_COLOR),
            interpolation="lanczos",
        )
        image_axis.axis("off")
        structure_axes.append(image_axis)
        label_axis = fig.add_subplot(label_grid[rank - 1, 0])
        label_axis.axis("off")
        label_axis.text(
            0.98,
            0.61,
            f"{rank}. {compound['name']}",
            transform=label_axis.transAxes,
            fontsize=6.5,
            fontweight="bold",
            ha="right",
            va="center",
            color=INK,
        )
        label_axis.text(
            0.98,
            0.34,
            textwrap.fill(compound["highlight_label"], width=16),
            transform=label_axis.transAxes,
            fontsize=5.6,
            fontweight="bold",
            ha="right",
            va="center",
            color=MOTIF_COLOR,
            linespacing=0.90,
        )
        row["pathway_rank"] = rank
        row["highlight_atom_indices"] = ";".join(str(value) for value in atoms)
        exported.append(row)

    bar_axis = fig.add_subplot(outer[1, 2])
    y = np.arange(len(selected_rows))
    scores = np.array([float(row["nes"]) for row in selected_rows])
    p_values = np.array([float(row["nominal_p"]) for row in selected_rows])
    q_values = np.array([float(row["bh_q"]) for row in selected_rows])
    colors = [
        enrichment_color(score, p_value, q_value)
        for score, p_value, q_value in zip(scores, p_values, q_values)
    ]
    bars = bar_axis.barh(
        y,
        scores,
        height=0.50,
        color=colors,
        edgecolor=INK,
        linewidth=0.70,
        alpha=0.92,
        zorder=3,
    )
    bar_axis.invert_yaxis()
    bar_axis.set_yticks([])
    bar_axis.axvline(0, color=INK, linewidth=0.8, zorder=2)
    bar_axis.grid(axis="x", color=GRID, linestyle=(0, (2, 2)), linewidth=0.6, zorder=0)
    bar_axis.spines[["top", "right", "left"]].set_visible(False)
    bar_axis.tick_params(axis="x", labelsize=6.7, width=0.7, length=2.5, pad=1.5)
    bar_axis.set_xlim(-3.1, 3.1)
    bar_axis.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    bar_axis.set_box_aspect(1.0)
    bar_axis.set_xlabel(
        pathway_axis_label(),
        fontsize=7.0,
        fontweight="bold",
        labelpad=2.5,
    )
    for bar, score, q_value in zip(bars, scores, q_values):
        marker = q_significance_marker(q_value)
        if not marker:
            continue
        marker_x = max(score, 0.0) + 0.16
        marker_y = bar.get_y() + bar.get_height() / 2.0
        star = bar_axis.text(
            marker_x,
            marker_y,
            marker,
            fontsize=7.8,
            fontweight="bold",
            ha="center",
            va="center",
            color=INK,
            clip_on=False,
            zorder=8,
        )
        star.set_path_effects([path_effects.withStroke(linewidth=1.1, foreground="white")])
    for row_index, image_axis in enumerate(structure_axes):
        connector = ConnectionPatch(
            xyA=(3.08, float(y[row_index])),
            coordsA=bar_axis.transData,
            xyB=(0.01, 0.50),
            coordsB=image_axis.transAxes,
            color="#A6A6A6",
            linewidth=0.55,
            linestyle="-",
            alpha=0.80,
            zorder=1,
            clip_on=False,
        )
        fig.add_artist(connector)
    positive_direction = bool(np.nanmean(scores) >= 0)
    dark_color = POSITIVE_COLORS["bh"] if positive_direction else NEGATIVE_COLORS["bh"]
    pale_color = POSITIVE_COLORS["none"] if positive_direction else NEGATIVE_COLORS["none"]
    bar_axis.legend(
        handles=[
            Patch(facecolor=dark_color, edgecolor=INK, label="<= 0.05"),
            Patch(facecolor=pale_color, edgecolor=INK, label="> 0.05"),
        ],
        loc="upper left" if positive_direction else "upper right",
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#B8B8B8",
        fontsize=4.0,
        borderpad=0.22,
        labelspacing=0.18,
        handlelength=1.0,
        handletextpad=0.28,
        title="BH q",
        title_fontsize=4.2,
    )
    return exported


def plot_endpoint_ranked_panel(
    fig: plt.Figure,
    spec,
    task: str,
    letter: str,
    compounds: pd.DataFrame,
    gsea_statistics: pd.DataFrame,
) -> list[pd.Series]:
    outer = spec.subgridspec(
        2,
        2,
        height_ratios=[0.11, 1.0],
        width_ratios=[1.0, 3.0],
        hspace=0.01,
        wspace=0.09,
    )
    heading = fig.add_subplot(outer[0, :])
    heading.axis("off")
    panel_letter(heading, letter, x=-0.105, y=0.91)
    frame = compounds.loc[compounds["context_id"].eq(CONTEXTS[task])].copy()
    context = CONTEXT_TITLES[task].split("(", 1)[1].rsplit(")", 1)[0]
    title_box = HPacker(
        children=[
            TextArea(task, textprops={"fontsize": 9.2, "fontweight": "bold", "color": INK}),
            TextArea(f" ({context})", textprops={"fontsize": 6.8, "fontweight": "bold", "color": INK}),
        ],
        align="baseline",
        pad=0,
        sep=1,
    )
    heading.add_artist(
        AnchoredOffsetbox(
            loc="lower center",
            child=title_box,
            frameon=False,
            pad=0,
            borderpad=0,
            bbox_to_anchor=(0.5, 0.20),
            bbox_transform=heading.transAxes,
        )
    )

    scatter_axis = fig.add_subplot(outer[1, 0])
    other = frame.loc[~frame["consensus_rescued"].eq(True)]
    consensus = frame.loc[frame["consensus_rescued"].eq(True) & ~frame["stable_rescued"].eq(True)]
    stable = frame.loc[frame["stable_rescued"].eq(True)]
    scatter_axis.scatter(other["mean_p_chem"], other["mean_p_bio"], s=4, color="#D0D0D0", alpha=0.50, linewidth=0, zorder=2)
    displayed_ids = [item["ik"] for item in ATLAS_COMPOUNDS[task]]
    consensus_background = consensus.loc[~consensus["ik"].isin(displayed_ids)]
    stable_background = stable.loc[~stable["ik"].isin(displayed_ids)]
    scatter_axis.scatter(consensus_background["mean_p_chem"], consensus_background["mean_p_bio"], s=12, facecolor=CONSENSUS, edgecolor=INK, linewidth=0.30, alpha=0.90, zorder=3)
    scatter_axis.scatter(stable_background["mean_p_chem"], stable_background["mean_p_bio"], s=12, facecolor=STABLE, edgecolor=INK, linewidth=0.35, alpha=0.94, zorder=4)
    displayed = frame.loc[frame["ik"].isin(displayed_ids)]
    displayed_consensus = displayed.loc[~displayed["stable_rescued"].eq(True)]
    displayed_stable = displayed.loc[displayed["stable_rescued"].eq(True)]
    scatter_axis.scatter(
        displayed_consensus["mean_p_chem"],
        displayed_consensus["mean_p_bio"],
        marker="*",
        s=27,
        facecolor=CONSENSUS,
        edgecolor=INK,
        linewidth=0.40,
        zorder=20,
    )
    scatter_axis.scatter(
        displayed_stable["mean_p_chem"],
        displayed_stable["mean_p_bio"],
        marker="*",
        s=27,
        facecolor=STABLE,
        edgecolor=INK,
        linewidth=0.40,
        zorder=20,
    )
    scatter_axis.plot([0, 1], [0, 1], color="#777777", linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    scatter_axis.axvline(0.5, color=GRID, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    scatter_axis.axhline(0.5, color=GRID, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    scatter_axis.set_xlim(0, 1)
    scatter_axis.set_ylim(0, 1)
    scatter_axis.set_aspect("equal", adjustable="box")
    scatter_axis.set_xticks([0, 0.5, 1.0])
    scatter_axis.set_yticks([0, 0.5, 1.0])
    scatter_axis.tick_params(labelsize=6.7, width=0.7, length=2.5, pad=1.5)
    scatter_axis.spines[["top", "right"]].set_visible(False)
    scatter_axis.set_xlabel("Chem.-only probability", fontsize=7.0, fontweight="bold", labelpad=1.5)
    scatter_axis.set_ylabel("Chem. + Bio.\nprobability", fontsize=7.0, fontweight="bold", labelpad=1.5)
    mean_delta_probability = float(frame["mean_delta_probability"].mean())
    positive_uplift_n = int(frame["mean_delta_probability"].gt(0).sum())
    positive_uplift_pct = 100.0 * positive_uplift_n / len(frame)
    rescued_n = int(frame["consensus_rescued"].sum())
    scatter_axis.text(
        0.5,
        1.055,
        f"Mean probability uplift = {mean_delta_probability:+.2f}\n"
        f"Positive uplift = {positive_uplift_n}/{len(frame)} ({positive_uplift_pct:.1f}%)\n"
        f"Rescued = {rescued_n}/{len(frame)}",
        transform=scatter_axis.transAxes,
        fontsize=5.4,
        fontweight="bold",
        ha="center",
        va="bottom",
        linespacing=1.15,
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.6},
    )
    scatter_axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markerfacecolor="#D0D0D0", markeredgecolor="none", markersize=3.6, label="Other active"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor=CONSENSUS, markeredgecolor=INK, markeredgewidth=0.25, markersize=3.8, label="Consensus rescue"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor=STABLE, markeredgecolor=INK, markeredgewidth=0.25, markersize=3.8, label="Stable rescue"),
            Line2D([], [], marker="*", linestyle="", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.35, markersize=5.0, label="Displayed rescue"),
        ],
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#B8B8B8",
        fontsize=5.0,
        borderpad=0.28,
        labelspacing=0.20,
        handletextpad=0.28,
    )

    metadata = {item["ik"]: item for item in ATLAS_COMPOUNDS[task]}
    selected_rows = []
    for compound in ATLAS_COMPOUNDS[task]:
        selected = frame.loc[frame["ik"].eq(compound["ik"])]
        if len(selected) != 1:
            raise ValueError(f"Expected one compound for {task}/{compound['ik']}; found {len(selected)}")
        row = selected.iloc[0].copy()
        row["drug_name"] = compound["name"]
        row["pubchem_cid"] = compound["pubchem_cid"]
        row["drug_name_source"] = compound["name_source"]
        row["highlight_id"] = compound["highlight"]
        row["highlight_label"] = compound["highlight_label"]
        gsea = gsea_statistics.loc[
            gsea_statistics["task"].eq(task) & gsea_statistics["ik"].eq(compound["ik"])
        ]
        if len(gsea) != 1:
            raise ValueError(f"Expected one GSEA row for {task}/{compound['ik']}; found {len(gsea)}")
        for key, value in gsea.iloc[0].items():
            if key not in {"task", "context_id", "ik", "drug_name"}:
                row[key] = value
        selected_rows.append(row)
    selected_rows.sort(key=lambda value: abs(float(value["nes"])), reverse=True)

    right = outer[1, 1].subgridspec(
        2,
        1,
        height_ratios=[0.40, 0.60],
        hspace=0.18,
    )
    bar_holder = right[0, 0].subgridspec(
        1,
        3,
        width_ratios=[0.14, 0.72, 0.14],
        wspace=0.0,
    )
    bar_axis = fig.add_subplot(bar_holder[0, 1])
    x = np.arange(len(selected_rows))
    scores = np.array([float(row["nes"]) for row in selected_rows])
    p_values = np.array([float(row["nominal_p"]) for row in selected_rows])
    q_values = np.array([float(row["bh_q"]) for row in selected_rows])
    colors = [POSITIVE_COLORS["bh"] if score >= 0 else NEGATIVE_COLORS["bh"] for score in scores]
    bars = bar_axis.bar(
        x,
        scores,
        width=0.56,
        color=colors,
        edgecolor=INK,
        linewidth=0.70,
        alpha=0.82,
        zorder=3,
    )
    bar_axis.axhline(0, color=INK, linewidth=0.8, zorder=2)
    bar_axis.grid(axis="y", color=GRID, linestyle=(0, (2, 2)), linewidth=0.6, zorder=0)
    bar_axis.spines[["top", "right"]].set_visible(False)
    bar_axis.tick_params(axis="y", labelsize=6.5, width=0.7, length=2.5, pad=1.5)
    bar_axis.tick_params(axis="x", labelsize=6.2, width=0.7, length=2.5, pad=2.0)
    bar_axis.set_ylim(-3.1, 3.1)
    bar_axis.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    bar_axis.set_xticks(x)
    bar_axis.set_xticklabels(
        [str(row["drug_name"]) for row in selected_rows],
        fontsize=6.2,
        fontweight="bold",
    )
    bar_axis.set_ylabel(pathway_axis_label(), fontsize=7.0, fontweight="bold", labelpad=2.5)
    for bar, score, q_value in zip(bars, scores, q_values):
        marker = q_significance_marker(q_value)
        if not marker:
            continue
        marker_y = score + (0.13 if score >= 0 else -0.13)
        star = bar_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            marker_y,
            marker,
            fontsize=7.8,
            fontweight="bold",
            ha="center",
            va="bottom" if score >= 0 else "top",
            color=INK,
            clip_on=False,
            zorder=8,
        )
        star.set_path_effects([path_effects.withStroke(linewidth=1.1, foreground="white")])
    structure_grid = right[1, 0].subgridspec(1, 4, wspace=0.025)
    exported = []
    for column, row in enumerate(selected_rows):
        compound = metadata[row["ik"]]
        mol = Chem.MolFromSmiles(row["smiles_canon"])
        atoms = atlas_highlight_atoms(mol, compound["highlight"])
        if not atoms:
            raise ValueError(f"No highlight atoms for {compound['name']}")
        cell = structure_grid[0, column].subgridspec(
            2,
            1,
            height_ratios=[0.89, 0.11],
            hspace=0.01,
        )
        image_axis = fig.add_subplot(cell[0, 0])
        image_axis.imshow(
            molecule_image(row["smiles_canon"], atoms, MOTIF_COLOR),
            interpolation="lanczos",
        )
        image_axis.axis("off")
        motif_axis = fig.add_subplot(cell[1, 0])
        motif_axis.axis("off")
        motif_axis.text(
            0.5,
            0.92,
            textwrap.fill(compound["highlight_label"], width=18),
            transform=motif_axis.transAxes,
            fontsize=5.8,
            fontweight="bold",
            ha="center",
            va="top",
            color=MOTIF_COLOR,
            linespacing=0.92,
        )
        row["pathway_rank"] = column + 1
        row["highlight_atom_indices"] = ";".join(str(value) for value in atoms)
        exported.append(row)
    return exported


def write_caption() -> None:
    caption = """# Figure caption

**Chemical structures and pathway enrichment among transcriptomically rescued compounds.** Panels **A-D** show NR-Aromatase (MCF7, 24 h), SR-ARE (HepG2, 6 h), SR-MMP (MCF7, 24 h) and SR-p53 (MCF7, 24 h), respectively; each cellular context is printed inline in smaller type. In each panel, the enlarged left plot compares mean repeated out-of-fold recalibrated Chem.-only and Chem. + Bio. probabilities among endpoint-active compounds. The diagonal denotes equal probabilities. Mean probability uplift is the mean signed change in active-class probability and, because every displayed compound has label one, is algebraically identical to the reduction in mean absolute probability error relative to the active label. Positive uplift reports both the number and percentage of active compounds with a positive probability change, and rescued is the consensus-rescue count. Consensus rescue required a mean Chem.-only probability below 0.5 and a mean combined probability at or above 0.5; stable rescue additionally required Chem.-only positivity in no more than 20% and combined positivity in at least 80% of repeated scaffold assignments. Slightly enlarged star markers with reduced edge width identify the four displayed compounds using their rescue-class fill color, while all other point sizes are unchanged. Probability differences include the separately fitted combined-model intercept and are descriptive model changes rather than a pure biological-offset decomposition.

For every displayed compound, all 978 control-referenced, viability-axis-residualized L1000 landmark-gene values were ranked from induced to repressed. Weighted preranked enrichment scores (weight 1) were calculated for the Hallmark early estrogen-response, Hallmark reactive-oxygen-species, Hallmark unfolded-protein-response and Hallmark p53 gene sets in panels A-D, respectively. Bar height is the normalized enrichment score (NES); positive and negative NES indicate concentration of pathway genes toward the induced and repressed ends of the compound-level transcriptional profile, respectively. Enrichment scores were normalized against 10,000 random gene sets of the same measured size. Nominal gene-set permutation P values and within-endpoint Benjamini-Hochberg q values are retained in the accompanying table. Uniform red and blue bars denote enrichment toward the induced and repressed ends of the ranked profile, respectively. Asterisks centered above significant bars denote BH q <= 0.05, 0.01 or 0.001. Compound names are the enrichment-plot x-axis labels, and the corresponding high-resolution structures appear in the same horizontal order below. Prespecified motifs or chemical regions are highlighted in moderately transparent red, and the motif name is printed beneath its structure. Compounds are ordered by decreasing absolute NES.

The enrichment analysis uses gene-set rather than phenotype permutations and therefore quantifies pathway concentration within an individual compound's ranked transcriptional response. The resulting P and q values are exploratory because the displayed compounds were prespecified from rescue behavior. Highlighted chemical features are structural annotations, not deterministic toxicophores. Enrichment does not establish intracellular exposure, causal mediation, motif-driven pathway activation or toxicity.
"""
    (OUTPUT / "FIGURE_CAPTION.md").write_text(caption, encoding="utf-8")


def write_methods_results(ranked_compounds: pd.DataFrame) -> None:
    lines = [
        "# Structural and pathway-response atlas",
        "",
        "## Methods",
        "",
        "Four endpoint-context combinations were prespecified from the nominally positive Stage-2 viability-adjusted analysis: NR-Aromatase-MCF7-24 h, SR-ARE-HepG2-6 h, SR-MMP-MCF7-24 h and SR-p53-MCF7-24 h. Scatter plots were restricted to compounds with active endpoint labels. Mean probabilities were calculated from repeated outer-test predictions. Consensus and stable rescue followed the archived repeated-CV threshold definitions and were not re-estimated from chemical structure.",
        "",
        "To quantify the broader active-compound shift in each scatter plot, mean signed probability uplift was calculated as mean Delta p_active = N_active^-1 sum_{i:y_i=1}(p_i,Chem+Bio - p_i,Chem). Because every plotted compound had active label y = 1, this quantity exactly equals MAE(Chem.-only, y=1) - MAE(Chem.+Bio., y=1), where MAE is calculated on the probability scale. Positive values therefore indicate a reduction in active-label absolute probability error. Positive uplift reports the count and percentage of active compounds for which p_Chem+Bio - p_Chem was greater than zero.",
        "",
        "Four prespecified consensus-rescued compounds were displayed per endpoint. Structures were rendered with RDKit, and the prespecified motif or chemical region was highlighted in transparent red. Compound names were verified against PubChem. The displayed compounds were ordered by the absolute magnitude of their compound-level normalized enrichment score rather than by probability change or chemical similarity.",
        "",
        "For each displayed compound, the complete 978-gene control-referenced viability-axis-residualized profile was ranked in decreasing order. A weighted running-sum enrichment score (weight 1) tested whether the measured members of the endpoint-aligned external gene set accumulated toward the induced or repressed end of this ranked profile. NR-Aromatase used Hallmark Estrogen Response Early, SR-ARE used Hallmark Reactive Oxygen Species Pathway, SR-MMP used Hallmark Unfolded Protein Response, and SR-p53 used Hallmark p53 Pathway. Bar height is the normalized enrichment score (NES); negative and positive values denote enrichment toward the repressed and induced ends of the ranked compound-level response, respectively.",
        "",
        "Each observed enrichment score was normalized by the mean same-sign enrichment score from 10,000 random gene sets sampled without replacement from the same 978-gene universe and matched to the measured pathway size. The sign-specific nominal P value was the plus-one-corrected fraction of same-sign null scores at least as extreme as the observed score. Benjamini-Hochberg correction was applied across the four displayed compounds within each endpoint. This is a compound-level preranked gene-set analysis with gene-set permutations, not phenotype-permutation GSEA.",
        "",
        "## Results",
        "",
    ]
    for task in TASK_ORDER:
        subset = ranked_compounds.loc[ranked_compounds["task"].eq(task)].sort_values("pathway_rank")
        names = ", ".join(
            f"{row.drug_name} (NES {row.nes:+.3f}, P={row.nominal_p:.4g}, q={row.bh_q:.4g})"
            for row in subset.itertuples(index=False)
        )
        lines.append(f"{task}: preranked enrichment order was {names}.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Probability rescue, molecular annotation and pathway enrichment represent distinct descriptive layers. Gene-set permutation significance describes concentration of an external pathway set within one compound's ranked response; it is not inferential evidence that the motif caused the response. The figure does not establish a deterministic toxicophore, intracellular exposure, causal mediation or molecular mechanism. It also does not show that the heterogeneous top-four chemical ensemble used the highlighted feature.",
        ]
    )
    (OUTPUT / "METHODS_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    compounds = active_context_data(pd.read_csv(SOURCE))
    gsea_statistics, gsea_audit = build_preranked_statistics(compounds)
    tables, figures = prepare_output()
    ranked_compounds = []
    fig = plt.figure(figsize=(8.27, 11.69))
    grid = fig.add_gridspec(4, 1, hspace=0.11)
    for index, (task, letter) in enumerate(zip(TASK_ORDER, ["A", "B", "C", "D"])):
        ranked_compounds.extend(
            plot_endpoint_ranked_panel(
                fig,
                grid[index, 0],
                task,
                letter,
                compounds,
                gsea_statistics,
            )
        )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.990, bottom=0.035)
    output_base = figures / "endpoint_structural_pathway_rank_atlas"
    # Matplotlib otherwise embeds imshow-based molecular drawings at the
    # figure's default 100 dpi even when the RDKit source raster is larger.
    fig.savefig(output_base.with_suffix(".pdf"), dpi=600)
    fig.savefig(output_base.with_suffix(".png"), dpi=600)
    plt.close(fig)

    compounds.to_csv(tables / "panels_a_to_d_active_compound_probabilities.csv", index=False)
    uplift_rows = []
    for task in TASK_ORDER:
        context = compounds.loc[compounds["context_id"].eq(CONTEXTS[task])]
        uplift_rows.append(
            {
                "task": task,
                "context_id": CONTEXTS[task],
                "n_active": len(context),
                "mean_delta_probability_active": context["mean_delta_probability"].mean(),
                "mean_active_label_probability_mae_reduction": context["mean_delta_probability"].mean(),
                "positive_uplift_n": int(context["mean_delta_probability"].gt(0).sum()),
                "positive_uplift_percent": 100.0 * context["mean_delta_probability"].gt(0).mean(),
                "consensus_rescued_n": int(context["consensus_rescued"].sum()),
                "stable_rescued_n": int(context["stable_rescued"].sum()),
            }
        )
    pd.DataFrame(uplift_rows).to_csv(
        tables / "panel_active_probability_uplift_statistics.csv",
        index=False,
    )
    gsea_statistics.to_csv(tables / "displayed_compound_preranked_enrichment_statistics.csv", index=False)
    gsea_audit.to_csv(tables / "preranked_enrichment_input_audit.csv", index=False)
    representative_frame = pd.DataFrame(ranked_compounds).reset_index(drop=True)
    representative_frame.to_csv(tables / "panels_a_to_d_ranked_compounds_preranked_enrichment.csv", index=False)
    source_paths = (
        [SOURCE]
        + [Path(path) for path in gsea_audit["gmt_path"].unique()]
        + [Path(path) for path in gsea_audit["expression_path"].unique()]
    )
    pd.DataFrame(
        [
            {"source_path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in source_paths
        ]
    ).drop_duplicates("source_path").to_csv(OUTPUT / "SOURCE_REGISTRY.csv", index=False)
    write_caption()
    write_methods_results(representative_frame)
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure_pdf": str(output_base.with_suffix(".pdf")),
        "figure_png": str(output_base.with_suffix(".png")),
        "source": str(SOURCE),
        "n_active_compound_rows": len(compounds),
        "n_endpoint_contexts": len(TASK_ORDER),
        "rdkit_version": rdBase.rdkitVersion,
        "rescue_definition": "consensus and stable rescue from repeated out-of-fold threshold behavior",
        "ranking": "four prespecified consensus-rescued compounds per endpoint ordered by descending absolute compound-level preranked NES",
        "pathway_score_method": "weighted preranked enrichment score normalized by same-sign random gene-set null",
        "gsea_weight": GSEA_WEIGHT,
        "gene_set_permutations": GSEA_PERMUTATIONS,
        "nominal_p": "plus-one-corrected same-sign gene-set permutation tail probability",
        "multiple_testing": "Benjamini-Hochberg across four displayed compounds within each endpoint",
        "nr_aromatase_gene_set": "Hallmark Estrogen Response Early",
        "sr_mmp_gene_set": "Hallmark Unfolded Protein Response",
        "sr_mmp_pathway_label": "Hallmark unfolded protein response",
        "interpretation": "descriptive probability rescue, structural annotation and compound-level pathway enrichment; not causal toxicophore attribution",
    }
    (OUTPUT / "RUN_COMPLETE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
