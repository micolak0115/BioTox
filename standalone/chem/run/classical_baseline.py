# chem/classical_baseline.py
"""
Classical cheminformatics baselines -- MACCS keys / Morgan fingerprints x
Logistic Regression / XGBoost -- for the chemical-encoder benchmark.

Unlike chem/model.py's `Fingerprint` (a trainable nn.Module + neural
ProbeHead), this module fits one plain sklearn/XGBoost binary classifier per
Tox21 task directly on bit-vector features, bypassing torch entirely. It
follows the same cached/remaining-seed convention as chem/benchmark.py's
other *_benchmark functions (_metrics_cache_path/_load_cached_metrics/
_save_cached_metrics/_load_cached_seeds/_run_seed_jobs, imported from there,
not duplicated) and returns metrics in the exact shape
chem/trainer.py::compute_metrics produces, so results_to_df/summarize_results
work on it unmodified.

Must be pointed at the SAME frozen split linear_publication uses
(412 train / 137 valid / 138 test, plain "scaffold" split) -- NOT the
413/137/137 "label_balanced_scaffold" split chem/eval_matched_137.py and
chem/main.py's configs use (a different partition, zero valid/test ik
overlap with the correct one -- confirmed by direct comparison).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))

from chem.benchmark import (  # noqa: E402
    _load_cached_metrics,
    _load_cached_seeds,
    _metrics_cache_path,
    _run_seed_jobs,
    _save_cached_metrics,
)
from chem.trainer import compute_metrics  # noqa: E402
from publication.utils import load_matched_split  # noqa: E402  (linear/ no longer exists; see publication/METHODS.md)

FP_KINDS = ("maccs", "morgan", "rdkit")
CLF_KINDS = ("logreg", "xgboost")
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
RDKIT_FP_BITS = 2048  # matches Morgan's bit count for consistency

# model_name -> (fingerprint kind, classifier kind)
# RDKit topological fingerprint (Chem.RDKFingerprint) added as a genuine
# third classical fingerprint alongside MACCS/Morgan, per explicit
# publication/ study design decision -- see publication/METHODS.md Phase 3.
CLASSICAL_VARIANTS = {
    "MACCS_LR": ("maccs", "logreg"),
    "MACCS_XGB": ("maccs", "xgboost"),
    "Morgan_LR": ("morgan", "logreg"),
    "Morgan_XGB": ("morgan", "xgboost"),
    "RDKit_LR": ("rdkit", "logreg"),
    "RDKit_XGB": ("rdkit", "xgboost"),
}

DEFAULT_EXPECTED_SPLIT_SIZES = {"train": 412, "valid": 137, "test": 138}


def featurize(smiles_list: list, kind: str) -> np.ndarray:
    """MACCS (167-bit), Morgan (2048-bit, r=2), or RDKit topological
    (2048-bit, Chem.RDKFingerprint -- path-based, distinct from both MACCS'
    fixed substructure keys and Morgan's circular/ECFP-style encoding) bit
    vectors. Null molecules (failed SMILES parse) get an all-zero row,
    matching Fingerprint.encode's fallback in chem/model.py."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys

    if kind == "maccs":
        n_bits = 167
    elif kind == "morgan":
        n_bits = MORGAN_BITS
    elif kind == "rdkit":
        n_bits = RDKIT_FP_BITS
    else:
        raise ValueError(f"unknown fingerprint kind {kind!r}")

    out = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if kind == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif kind == "morgan":
            fp = AllChem.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS).GetFingerprint(mol)
        else:  # rdkit
            fp = Chem.RDKFingerprint(mol, fpSize=RDKIT_FP_BITS)
        out[i] = np.array(fp, dtype=np.float32)
    return out


LOGREG_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
XGB_EARLY_STOPPING_ROUNDS = 20  # matches the GNN/ChemBERTa/Chemprop candidates' patience=20 convention
# max_depth x learning_rate grid (6 combinations, matching LOGREG_C_GRID's size) --
# n_estimators is NOT grid-searched, since early stopping already selects it
# per combination; subsample/colsample_bytree stay fixed to keep the grid small.
XGB_MAX_DEPTH_GRID = (3, 4, 6)
XGB_LEARNING_RATE_GRID = (0.05, 0.1)
FIT_SETTINGS = {}


def configure_settings(settings):
    global FIT_SETTINGS
    FIT_SETTINGS = dict(settings or {})


def _setting(key, default):
    return FIT_SETTINGS.get(key, default)


def _fit_predict_one_task(clf_kind: str, X_fit: np.ndarray, y_fit_col: np.ndarray, X_test: np.ndarray, seed: int,
                           X_valid: np.ndarray = None, y_valid_col: np.ndarray = None) -> np.ndarray:
    """X_valid/y_valid_col are optional (default None -- preserves the
    original train-only behavior for classical_benchmark/_run_classical_seed's
    callers, which fold train+valid together since they historically had no
    stopping criterion to spend valid on). When provided (chem_stage1_per_task_
    offset.py's caller), valid is used strictly within-candidate -- XGBoost
    early stopping via eval_set, LogisticRegression's C selected by valid
    AUPRC -- never for cross-candidate comparison, matching the same
    train=fit/valid=tune-only contract the GNN/ChemBERTa/Chemprop candidates
    already follow."""
    valid_mask = ~np.isnan(y_fit_col)
    X_tr, y_tr = X_fit[valid_mask], y_fit_col[valid_mask]

    has_valid = X_valid is not None and y_valid_col is not None
    if has_valid:
        valid_valid_mask = ~np.isnan(y_valid_col)
        X_va, y_va = X_valid[valid_valid_mask], y_valid_col[valid_valid_mask]
        has_valid = len(np.unique(y_va)) >= 2  # degenerate valid slice: fall back to train-only fit

    if len(np.unique(y_tr)) < 2:
        # degenerate task fold: fall back to the observed prevalence (matches
        # compute_metrics' own random_auprc convention for a task it would skip).
        fallback = float(y_tr.mean()) if len(y_tr) else 0.5
        return np.full(X_test.shape[0], fallback, dtype=np.float64)

    if clf_kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score

        if has_valid:
            best_c, best_score, best_clf = None, -np.inf, None
            for c in _setting("logreg_c_grid", LOGREG_C_GRID):
                cand = LogisticRegression(
                    C=c, class_weight="balanced",
                    max_iter=int(_setting("logreg_max_iter", 2000)), random_state=seed,
                )
                cand.fit(X_tr, y_tr)
                score = average_precision_score(y_va, cand.predict_proba(X_va)[:, 1])
                if score > best_score:
                    best_c, best_score, best_clf = c, score, cand
            clf = best_clf
        else:
            clf = LogisticRegression(
                class_weight="balanced", max_iter=int(_setting("logreg_max_iter", 2000)),
                random_state=seed,
            )
            clf.fit(X_tr, y_tr)
    elif clf_kind == "xgboost":
        from xgboost import XGBClassifier
        from sklearn.metrics import average_precision_score

        neg, pos = float((y_tr == 0).sum()), float((y_tr == 1).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        base_kwargs = dict(
            n_estimators=int(_setting("xgb_n_estimators", 300)),
            subsample=float(_setting("xgb_subsample", 0.8)),
            colsample_bytree=float(_setting("xgb_colsample_bytree", 0.8)),
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="aucpr",  # AUPRC-consistent with this study's primary metric (was logloss)
            n_jobs=1,
        )
        if has_valid:
            # hyperparameter search over (max_depth, learning_rate): each
            # combination gets its own early-stopped fit (aucpr-monitored, same
            # AUPRC selection criterion as the LogisticRegression C search
            # above), scored explicitly via average_precision_score on valid
            # (not XGBoost's internal best_score) so both classifier kinds are
            # selected by the identical metric computation.
            best_params, best_score, best_clf = None, -np.inf, None
            for max_depth in _setting("xgb_max_depth_grid", XGB_MAX_DEPTH_GRID):
                for lr in _setting("xgb_learning_rate_grid", XGB_LEARNING_RATE_GRID):
                    cand = XGBClassifier(
                        max_depth=max_depth, learning_rate=lr,
                        early_stopping_rounds=int(_setting("xgb_early_stopping_rounds", 20)),
                        **base_kwargs,
                    )
                    cand.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                    score = average_precision_score(y_va, cand.predict_proba(X_va)[:, 1])
                    if score > best_score:
                        best_params, best_score, best_clf = (max_depth, lr), score, cand
            clf = best_clf
        else:
            clf = XGBClassifier(
                max_depth=int(_setting("xgb_default_max_depth", 4)),
                learning_rate=float(_setting("xgb_default_learning_rate", 0.05)),
                **base_kwargs,
            )
            clf.fit(X_tr, y_tr)
    else:
        raise ValueError(f"unknown clf_kind {clf_kind!r}")

    return clf.predict_proba(X_test)[:, 1]


def _assert_split_sizes(matched_df: pd.DataFrame, expected: dict) -> None:
    sizes = matched_df["split"].value_counts().to_dict()
    for split_name, n in expected.items():
        actual = sizes.get(split_name, 0)
        if actual != n:
            raise ValueError(
                f"matched split size mismatch: expected {split_name}={n}, got {actual}. "
                "Wrong split file -- classical_benchmark must point at the "
                "linear_publication 412/137/138 scaffold split, not the "
                "413/137/137 label_balanced_scaffold split used elsewhere in chem/."
            )


def _run_classical_seed(model_name: str, fp_kind: str, clf_kind: str, seed: int, config: dict, matched_df: pd.DataFrame,
                         tag: str = "classical", extra_fit_df: Optional[pd.DataFrame] = None) -> tuple:
    cache_path = _metrics_cache_path(config["save_dir"], model_name, tag, seed)
    cached = _load_cached_metrics(cache_path)
    if cached is not None:
        print(f"[Classical][SKIP] {model_name} seed={seed}: cached")
        return seed, cached

    task_cols = config["task_cols"]
    train_df = matched_df[matched_df["split"] == "train"]
    valid_df = matched_df[matched_df["split"] == "valid"]
    test_df = matched_df[matched_df["split"] == "test"]
    # No early-stopping mechanism for classical sklearn/XGBoost models -- fit
    # on train+valid (valid never used for fitting elsewhere in this repo's
    # neural encoders, but here there's no separate stopping criterion to
    # spend it on, so folding it into the fit set uses the data instead of
    # discarding it). Condition-2 (expanded) variant additionally folds in the
    # scaffold-disjoint extra-Tox21 pool (train+valid, same "no early
    # stopping needed" reasoning) -- same 138-compound test either way.
    fit_parts = [train_df, valid_df]
    if extra_fit_df is not None:
        fit_parts.append(extra_fit_df[["smiles_canon", *task_cols]])
    fit_df = pd.concat(fit_parts, axis=0, ignore_index=True)

    try:
        X_fit = featurize(fit_df["smiles_canon"].tolist(), fp_kind)
        X_test = featurize(test_df["smiles_canon"].tolist(), fp_kind)
        y_fit = fit_df[task_cols].to_numpy(dtype=np.float64)
        y_test = test_df[task_cols].to_numpy(dtype=np.float64)

        preds = np.zeros((X_test.shape[0], len(task_cols)), dtype=np.float64)
        for t in range(len(task_cols)):
            preds[:, t] = _fit_predict_one_task(clf_kind, X_fit, y_fit[:, t], X_test, seed)

        metrics = compute_metrics(preds, y_test, "multilabel_classification")
        _save_cached_metrics(cache_path, metrics)
        print(f"[Classical][DONE] {model_name} seed={seed}: auprc={metrics.get('auprc')} auroc={metrics.get('auroc')}")
        return seed, metrics
    except Exception as e:
        print(f"[Classical][FAIL] {model_name} seed={seed}: {e}")
        return seed, {"error": str(e)}


def classical_benchmark(config: dict, matched_split_path: str, seeds: list, models: Optional[list] = None) -> dict:
    """
    Returns {model_name: {seed: metrics}} for MACCS/Morgan x LogReg/XGBoost,
    following the same cached/remaining-seed loop as chem/benchmark.py's
    other *_benchmark functions.
    """
    matched_df = load_matched_split(matched_split_path)
    expected = config.get("expected_split_sizes", DEFAULT_EXPECTED_SPLIT_SIZES)
    _assert_split_sizes(matched_df, expected)

    allowed = set(models or config.get("classical_models") or [])
    variants = {
        name: v for name, v in CLASSICAL_VARIANTS.items()
        if not allowed or name in allowed
    }

    results = {}
    for model_name, (fp_kind, clf_kind) in variants.items():
        cached_pairs = _load_cached_seeds(config["save_dir"], model_name, "classical", seeds)
        remaining = [s for s in seeds if s not in cached_pairs]
        print(f"\n[Classical] {model_name} — {len(cached_pairs)} cached, {len(remaining)} to run")

        new_pairs = {}
        if remaining:
            n_jobs = config.get("n_jobs", 1)
            jobs = [
                (lambda seed=seed: _run_classical_seed(model_name, fp_kind, clf_kind, seed, config, matched_df))
                for seed in remaining
            ]
            new_pairs = dict(_run_seed_jobs(jobs, n_jobs))

        results[model_name] = {**cached_pairs, **new_pairs}

    return results


def classical_expanded_benchmark(config: dict, matched_split_path: str, extra_tox21_path: str, seeds: list, models: Optional[list] = None) -> dict:
    """
    Condition-2 control arm: same MACCS/Morgan x LogReg/XGBoost variants,
    fit on matched_train+valid PLUS the scaffold-disjoint extra-Tox21 pool
    (train+valid), evaluated on the same matched 138-compound test. Isolates
    whether GNN/ChemBERTa's edge over classical FP+ML on the expanded pool
    is really about their external self-supervised pretraining, or just
    about having more Tox21-labeled rows -- classical FP+ML has no
    pretrained-representation concept to give it "for free," but it CAN use
    the same extra rows, so this keeps that axis controlled.
    """
    matched_df = load_matched_split(matched_split_path)
    expected = config.get("expected_split_sizes", DEFAULT_EXPECTED_SPLIT_SIZES)
    _assert_split_sizes(matched_df, expected)

    extra_df = pd.read_pickle(extra_tox21_path)
    overlap = set(extra_df["ik"]) & set(matched_df["ik"])
    if overlap:
        raise AssertionError(f"{len(overlap)} extra-Tox21 compounds overlap the 687 matched compounds by ik")

    allowed = set(models or config.get("classical_models") or [])
    variants = {
        name: v for name, v in CLASSICAL_VARIANTS.items()
        if not allowed or name in allowed
    }

    tag = "classical_expanded"
    results = {}
    for model_name, (fp_kind, clf_kind) in variants.items():
        cached_pairs = _load_cached_seeds(config["save_dir"], model_name, tag, seeds)
        remaining = [s for s in seeds if s not in cached_pairs]
        print(f"\n[Classical-expanded] {model_name} — {len(cached_pairs)} cached, {len(remaining)} to run")

        new_pairs = {}
        if remaining:
            n_jobs = config.get("n_jobs", 1)
            jobs = [
                (lambda seed=seed: _run_classical_seed(
                    model_name, fp_kind, clf_kind, seed, config, matched_df, tag=tag, extra_fit_df=extra_df
                ))
                for seed in remaining
            ]
            new_pairs = dict(_run_seed_jobs(jobs, n_jobs))

        results[model_name] = {**cached_pairs, **new_pairs}

    return results
