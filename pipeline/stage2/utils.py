# publication/utils.py
"""
Shared statistical/GLM machinery for the offset-logistic bio-adds-to-chem
study. Ported from backup/linear_offset_logistic/utils.py +
fit_offset_logistic_pilot.py, consolidated into one module and adapted for
this study's per-cell (not crosscell) design -- see METHODS.md for the full
rationale behind every departure from the backup version, summarized here:

  - CELL_IDS: 4 cells (HA1E/HEPG2/HT29/MCF7), no LOCO reservation of A549/PC3
    -- this study has no held-out-cell arm.
  - eta_chem is a per-(compound, task) quantity, precomputed ONCE by
    chem_stage1_per_task_offset.py (Phase 3: deployed equal-weight mean over
    the selected top-4 candidates; train fits candidate parameters,
    valid is only for within-candidate early stopping/checkpoint selection,
    and the Stage-1 development-selection partition ranks candidates for the
    task-specific K=4 mean; one inference pass on the cohort union) and read
    as a static CSV by Phase 5/6 -- not fold- or cell-specific, and not
    computed inside this module.
    build_task_cell_frame below therefore does NOT merge an offsets table
    -- callers (Phase 5/6) do that themselves after loading
    chem_offset_final.csv. See METHODS.md Phase 3 for the correction
    history behind this being a static per-task file rather than a
    per-fold-refit quantity.
  - build_task_cell_frame no longer filters gene columns by a "{cell}__"
    prefix: Phase 2 emits one gene-features CSV per cell (already
    cell-scoped), not a shared wide table keyed by "{cell}__{gene}" across
    all cells (that only made sense under the old crosscell design, where
    every cell shared the same compound axis to align columns against).
  - inverse_prevalence_weights, _select_alpha, build_task_cell_frame,
    _weighted_binomial_deviance, _cv_fold_losses (previously split across
    fit_offset_logistic_pilot.py) are consolidated here since this study has
    no single-frozen-split "pilot" script -- nested CV is the primary
    evaluation, so every caller (nested CV, permutation test) needs these.
"""
from __future__ import annotations

import functools
import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.splitter import build_scaffold_dict  # noqa: E402
from stage1.utils import ik_first  # noqa: E402

# _select_alpha's grid points are mutually independent (unlike Brent's
# sequential search), so they run concurrently via joblib. Each worker caps
# itself to 1 BLAS thread (threadpoolctl) so N concurrent single-threaded
# fits use N cores cleanly instead of each fit spawning a full-width thread
# pool and contending with the others -- this was the actual cause of the
# 128-threads-on-64-cores oversubscription observed in the live Phase 5 run,
# not the grid change itself. Reserve a couple cores for the OS/other work.
# Conservative standalone default. The Stage-2 driver passes an explicit
# budget when it also parallelizes independent task-cell jobs.
GRID_N_JOBS_DEFAULT = max(1, min(8, (os.cpu_count() or 4) - 2))

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

CELL_IDS = ["HA1E", "HEPG2", "HT29", "MCF7"]

DOSE_WINDOW_UM = (8.0, 12.0)


# ---------------------------------------------------------------------------
# ik-keyed joins / matched-split loading
# ---------------------------------------------------------------------------
def add_ik_first(df: pd.DataFrame, ik_col: str = "ik", out_col: str = "ik") -> pd.DataFrame:
    df = df.copy()
    df[out_col] = df[ik_col].astype(str).map(ik_first)
    return df


def load_matched_split(matched_split_path: str) -> pd.DataFrame:
    """
    Load one of Phase 1's per-cell (or the chem-finetune-pool) DataSplit
    pickles and return one concatenated DataFrame with a `split` column
    ("train"/"valid"/"test").
    """
    import pickle

    with open(matched_split_path, "rb") as f:
        loaded = pickle.load(f)
    data_split = loaded[0] if isinstance(loaded, tuple) else loaded

    frames = []
    for split_name in ("train", "valid", "test"):
        df = getattr(data_split, split_name).copy()
        if len(df) == 0:
            continue
        df["split"] = split_name
        df = add_ik_first(df, ik_col="ik", out_col="ik")
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def build_task_cell_frame(compounds: pd.DataFrame, gene_features: pd.DataFrame, task: str):
    """
    compounds: one cell's cohort (Phase 1 output, via load_matched_split).
    gene_features: that SAME cell's own gene-features table (Phase 2 output --
    plain gene-symbol columns, no "{cell}__" prefix, since Phase 2 emits one
    file per cell already).

    Keeps ALL TOX21_TASKS columns (not just `task`) for downstream
    flexibility (e.g. Phase 9's diagnostics), even though only `task`'s
    column is actually used by Phase 5/6's bio-side GLM. (Earlier versions
    of this function needed this for a stricter reason -- feeding a
    then-multi-task chemical model -- that no longer applies now that
    Phase 3's fixed equal-weight chemical priors are genuinely single-task
    per METHODS.md's Phase 3 design; kept anyway since it's harmless and
    other callers may want it.) Only ROW filtering is done against `task`
    specifically.

    Does NOT attach eta_chem -- callers (nested CV, permutation test) merge
    the static Stage-1 offset table themselves after constructing this
    task/cell frame.
    """
    gene_cols = [c for c in gene_features.columns if c != "ik"]
    keep_cols = ["ik", "split", "smiles_canon"] + TOX21_TASKS
    df = compounds[keep_cols].merge(
        gene_features[["ik"] + gene_cols], on="ik", how="inner"
    )
    df = (
        df[df[task].notna()]
        .sort_values("ik", kind="stable")
        .reset_index(drop=True)
    )
    return df, gene_cols


# ---------------------------------------------------------------------------
# class-imbalance weighting -- sample weighting ONLY, no SMOTE/synthetic
# oversampling (see METHODS.md Phase 4 for why: scaffold-split leakage risk +
# corrupted AUPRC interpretation at this study's small positive counts).
# ---------------------------------------------------------------------------
def inverse_prevalence_weights(y: np.ndarray) -> np.ndarray:
    """sklearn's class_weight='balanced' formula: w_i = n / (2 * n_class(y_i))."""
    y = np.asarray(y)
    n = len(y)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    w = np.empty(n)
    w[y == 1] = n / (2.0 * max(n_pos, 1))
    w[y == 0] = n / (2.0 * max(n_neg, 1))
    return w


# ---------------------------------------------------------------------------
# offset-ridge logistic GLM. `offset=` is a fixed, coefficient-1 term in
# statsmodels by construction -- this MUST stay an `offset=` kwarg, never a
# free-fitted regressor folded into X (see METHODS.md Phase 4 guardrail #1).
# ---------------------------------------------------------------------------
def fit_offset_ridge_glm(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    alpha: float,
    L1_wt: float = 0.0,
    include_intercept: bool = True,
    var_weights: np.ndarray = None,
    maxiter: int = 100,
    cnvrg_tol: float = 1e-7,
):
    """
    Ridge-penalized (L1_wt=0.0 -> pure L2) logistic regression with a FIXED,
    non-fitted offset term:

        eta = offset + X @ beta   (+ intercept if include_intercept)
        y ~ Bernoulli(sigmoid(eta))

    The intercept, when included, is left UNPENALIZED -- statsmodels'
    fit_regularized broadcasts a scalar `alpha` to every design column
    including the constant added by sm.add_constant, which (confirmed
    empirically: a true intercept of 3.0 was crushed to 0.0044 at alpha=100
    and to 0.0000 at alpha=1e6) silently defeats the intercept's role as a
    free "calibration-in-the-large" term at exactly the large-alpha,
    near-saturated regularization values this study's alpha grid routinely
    selects. Passed as a per-column array instead: 0.0 at the intercept's
    position (column 0, since sm.add_constant prepends by default -- matches
    predict_offset_ridge_glm's params[0]/params[1:] split), `alpha`
    everywhere else, so only the gene coefficients are ridge-shrunk.

    Returns the fitted statsmodels GLMResultsWrapper.
    """
    X_design = sm.add_constant(X, has_constant="add") if include_intercept else X
    kwargs = {"var_weights": var_weights} if var_weights is not None else {}
    model = sm.GLM(y, X_design, family=sm.families.Binomial(), offset=offset, **kwargs)
    if include_intercept:
        alpha_vec = np.concatenate([[0.0], np.full(X.shape[1], alpha, dtype=float)])
    else:
        alpha_vec = alpha
    return model.fit_regularized(
        method="elastic_net", alpha=alpha_vec, L1_wt=L1_wt, maxiter=maxiter, cnvrg_tol=cnvrg_tol,
    )


def fit_offset_intercept_glm(
    y: np.ndarray,
    offset: np.ndarray,
    var_weights: np.ndarray = None,
    maxiter: int = 100,
):
    """
    Fit the fair Stage-2 chemical null:

        eta_null = offset + intercept

    The offset remains fixed with coefficient 1, while the single intercept
    is estimated without any penalty. This is calibration-in-the-large only;
    it cannot change sample ranking and therefore leaves AUROC/AUPRC
    unchanged relative to the raw Stage-1 offset.
    """
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    intercept_design = np.ones((len(y), 1), dtype=float)
    kwargs = {"var_weights": var_weights} if var_weights is not None else {}
    model = sm.GLM(
        y,
        intercept_design,
        family=sm.families.Binomial(),
        offset=offset,
        **kwargs,
    )
    return model.fit(maxiter=maxiter)


def predict_offset_intercept_glm(result, offset: np.ndarray) -> np.ndarray:
    """Predict ``sigmoid(offset + intercept)`` from the fitted null model."""
    intercept = float(np.asarray(result.params).reshape(-1)[0])
    return expit(np.asarray(offset, dtype=float) + intercept)


def predict_offset_ridge_glm(
    result,
    X: np.ndarray,
    offset: np.ndarray,
    include_intercept: bool = True,
) -> np.ndarray:
    """sigmoid(offset + X @ beta [+ intercept]) using a fitted GLM result."""
    params = np.asarray(result.params)
    if include_intercept:
        intercept, beta = params[0], params[1:]
    else:
        intercept, beta = 0.0, params
    eta = offset + X @ beta + intercept
    return expit(eta)


# ---------------------------------------------------------------------------
# scaffold-grouped k-fold
# ---------------------------------------------------------------------------
def scaffold_group_kfold(
    smiles_list: List[str],
    k: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Assign compounds to k scaffold-disjoint folds via greedy largest-first
    bin-packing, so no two compounds sharing a Murcko scaffold land in
    different folds. Returns k (train_idx, test_idx) index-array pairs.

    NOTE (Open item from the plan): this is plain group-only k-fold. The
    recommended upgrade is sklearn.model_selection.StratifiedGroupKFold
    (group=scaffold, stratify=y) to also balance positive counts across
    folds within the scaffold-group constraint -- see
    stratified_scaffold_group_kfold below, added alongside this for that
    reason. Both keep the group-disjointness guarantee; the stratified
    version is strictly better when class imbalance matters (which it does
    here -- see METHODS.md Phase 5).
    """
    scaffold_dict = build_scaffold_dict(list(enumerate(smiles_list)))
    groups = list(scaffold_dict.values())

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in order]
    groups = sorted(groups, key=len, reverse=True)

    fold_indices: List[List[int]] = [[] for _ in range(k)]
    for group in groups:
        target = min(range(k), key=lambda i: len(fold_indices[i]))
        fold_indices[target].extend(group)

    all_idx = np.arange(len(smiles_list))
    splits = []
    for i in range(k):
        test_idx = np.array(sorted(fold_indices[i]), dtype=int)
        train_idx = np.array(sorted(set(all_idx.tolist()) - set(test_idx.tolist())), dtype=int)
        splits.append((train_idx, test_idx))
    return splits


def stratified_scaffold_group_kfold(
    smiles_list: List[str],
    y: np.ndarray,
    k: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Scaffold-grouped k-fold that also balances positive-label counts across
    folds, via sklearn's StratifiedGroupKFold (group constraint is a HARD
    guarantee by construction -- no scaffold is ever split across folds;
    stratification is best-effort on top of that, per sklearn's own
    documented contract). Recommended over scaffold_group_kfold given this
    study's thin positive counts (as low as 19 total for HEPG2 x
    NR-PPAR-gamma) -- see METHODS.md Phase 5 / the plan's "Open items".
    """
    from sklearn.model_selection import StratifiedGroupKFold

    scaffold_dict = build_scaffold_dict(list(enumerate(smiles_list)))
    group_id = np.empty(len(smiles_list), dtype=int)
    for gid, indices in enumerate(scaffold_dict.values()):
        for i in indices:
            group_id[i] = gid
    # any SMILES that failed to parse (not in scaffold_dict) gets its own
    # singleton group so it's never silently dropped from the split.
    unassigned = set(range(len(smiles_list))) - {i for indices in scaffold_dict.values() for i in indices}
    next_gid = len(scaffold_dict)
    for i in unassigned:
        group_id[i] = next_gid
        next_gid += 1

    y = np.asarray(y)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(smiles_list, y, groups=group_id)]


# ---------------------------------------------------------------------------
# shared binary-classification metrics. AUPRC is primary throughout (matches
# this study's class-imbalance-heavy Tox21 endpoints far more sensitively
# than AUROC); never compare raw AUPRC across cells or across tasks (see
# METHODS.md Phase 8) -- only within-(task,cell) paired deltas are comparable.
# ---------------------------------------------------------------------------
def compute_binary_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true)
    p = np.asarray(p)
    mask = ~np.isnan(y_true) & np.isfinite(p)
    y_true, p = y_true[mask], p[mask]
    if mask.sum() < 2 or len(np.unique(y_true)) < 2:
        return {"auprc": float("nan"), "auroc": float("nan"), "n": int(mask.sum())}
    return {
        "auprc": float(average_precision_score(y_true, p)),
        "auroc": float(roc_auc_score(y_true, p)),
        "n": int(mask.sum()),
    }


def _weighted_binomial_deviance(y_true: np.ndarray, p: np.ndarray, w: np.ndarray = None) -> float:
    """Mean weighted binomial deviance -- the GLM's OWN loss, used for alpha
    selection (matched-to-loss criterion, not a ranking metric like AUPRC)."""
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)
    if w is None:
        w = np.ones_like(y_true, dtype=float)
    ll = w * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return float(-2.0 * np.sum(ll) / np.sum(w))


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit feature scaling on ``X_train`` and apply it to both partitions."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


def _cv_fold_losses(log10_alpha: float, X, y, offset, fold_data, include_intercept: bool,
                     maxiter: int = 50, cnvrg_tol: float = 1e-4,
                     standardize_features: bool = False) -> dict:
    """Returns {"losses": [...], "n_converged": int, "n_warned": int, "n_failed": int}.

    Three distinct outcomes per fold, not two: a fold either (a) fits and
    statsmodels raises no convergence concern -- converged, loss counted;
    (b) fits without raising an exception but statsmodels emits its
    "GLM ridge optimization may have failed, |grad|=..." UserWarning -- the
    fit technically completed, but its coefficients are not trustworthy, so
    its loss is EXCLUDED from the mean (previously this fold's loss WAS
    silently included, since a warning is not an exception and the old
    `except Exception: continue` never saw it -- that is the actual bug this
    replaces, not an unequal-sample-size issue as originally hypothesized);
    or (c) fitting raises a genuine exception (e.g. a singular design matrix)
    -- excluded, as before. (b) and (c) are counted separately so a caller
    can tell "fit completed but was flagged unreliable" apart from "fit
    crashed outright" -- both currently get folded into `n_failed` for the
    grid-level summary, but are tracked distinctly here for future use.
    """
    alpha = 10.0 ** log10_alpha
    losses = []
    n_converged = n_warned = n_failed = 0
    for train_idx, test_idx, w_tr, w_te in fold_data:
        try:
            X_train = X[train_idx]
            X_test = X[test_idx]
            if standardize_features:
                X_train, X_test = standardize_train_test(X_train, X_test)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", category=UserWarning)
                result = fit_offset_ridge_glm(
                    X_train, y[train_idx], offset[train_idx], alpha,
                    L1_wt=0.0, include_intercept=include_intercept, var_weights=w_tr,
                    maxiter=maxiter, cnvrg_tol=cnvrg_tol,
                )
            convergence_warned = any("ridge optimization may have failed" in str(w.message) for w in caught)
            if convergence_warned:
                n_warned += 1
                continue
            p_te = predict_offset_ridge_glm(result, X_test, offset[test_idx], include_intercept)
            losses.append(_weighted_binomial_deviance(y[test_idx], p_te, w_te))
            n_converged += 1
        except Exception:
            n_failed += 1
            continue
    return {"losses": losses, "n_converged": n_converged, "n_warned": n_warned, "n_failed": n_failed}


def _grid_point_job(
    log10_alpha,
    X,
    y,
    offset,
    fold_data,
    include_intercept,
    maxiter,
    cnvrg_tol,
    standardize_features,
):
    """One grid point's full inner-CV evaluation -- the unit of parallelism
    for _select_alpha's grid. Caps this worker to 1 BLAS thread so running
    many of these concurrently (one per core) doesn't recreate the same
    thread-oversubscription problem at the worker level."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return log10_alpha, _cv_fold_losses(
            log10_alpha,
            X,
            y,
            offset,
            fold_data,
            include_intercept,
            maxiter,
            cnvrg_tol,
            standardize_features,
        )


def _run_grid_jobs(jobs: list, n_jobs: int) -> list:
    if n_jobs == 1:
        return [fn() for fn in jobs]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)() for fn in jobs)


def _select_alpha(X, y, offset, smiles, include_intercept, cv_k, cv_seed, use_sample_weight,
                   log10_min: float = -4.0, log10_max: float = 8.0, points_per_decade: float = 4.0,
                   light: bool = False, stratify_folds: bool = True,
                   maxiter: int = 50, cnvrg_tol: float = 1e-4, verbose_task_cell: str = None,
                   grid_n_jobs: int = GRID_N_JOBS_DEFAULT,
                   standardize_features: bool = False):
    """
    Alpha selection over a fixed, discrete log10(alpha) grid -- replaces an
    earlier Brent's-method (continuous, unbounded) search + separate
    diagnostic-grid + re-bracketing fallback. Switched deliberately, not as
    a simplification for its own sake:

    - The grid IS the truncation defense Brent's unboundedness used to
      provide, as long as it's wide enough: log10_min=-4 (widened from -2)
      is confirmed to correctly resolve cases like NR-AhR/HA1E, where the
      true optimum sits near the grid's upper edge and beta -> 0 (no signal)
      is the right answer, not a boundary artifact.
    - `light` no longer changes what's computed -- both call sites
      (nested_cv_offset_logistic.py, permutation_test_offset_logistic.py)
      always pass light=True and only ever read `alpha_star`/
      `alpha_boundary_hit` from the result, so unifying the two paths onto
      one shared grid removes a source of alpha_star disagreement between
      them for free. `light` is kept as a parameter for call-site
      compatibility; every field below is always computed and returned.
    - Grid points double as the plateau probe (no separate multiplier-probe
      pass needed): `alpha_plateau_low/high` and `is_plateau_10x` fall out
      directly from which grid points clear the 1-SE threshold.
    - `alpha_boundary_hit` is a REAL truncation signal again now that the
      grid is finite (with Brent it was essentially dead -- unbounded
      search rarely lands exactly on log10_min/log10_max). It checks the
      DEPLOYED value (`alpha_star`, after the 1-SE walk), not the argmin --
      `alpha_min_boundary_hit` is reported separately for the argmin's own
      position. A hit does NOT always mean truncation risk: once beta
      saturates to ~0 under strong enough regularization, the loss curve
      goes genuinely flat and the walk has no reason to stop before the
      grid's edge, so `alpha_star` routinely lands at `log10_max` for
      endpoints where biology adds ~nothing beyond an already-strong
      chemical prior -- alpha=1e8 and alpha=1e5 are numerically
      indistinguishable once beta has saturated. Read `alpha_boundary_hit`
      together with `delta_auprc`: near-zero delta + boundary hit is the
      saturation signature (expected, not a problem); a large delta at the
      boundary would be the genuine-truncation signature worth widening
      the grid for.
    - `points_per_decade=4` (~0.25 dex spacing) is the deliberate tuning
      knob, not incidental: too sparse (e.g. 1/decade) makes the 1-SE
      tie-break jump whole decades; too dense is wasted precision given
      `cv_k=3` inner folds already carry substantial loss-estimate noise of
      their own -- picking a finer argmin than the fold noise supports
      isn't resolving signal, it's fitting noise. 4-5/decade is the
      sweet spot: stable tie-breaks, fit count still tractable.

    When ``standardize_features`` is true, every inner fold fits its own
    scaler on the inner-training rows and applies it to the inner-validation
    rows. This prevents the outer-training partition's inner-validation
    moments from leaking into alpha selection.

    Selection uses binomial deviance. It is unweighted for the publication
    model and weighted only when `use_sample_weight=True` is explicitly
    requested as a sensitivity analysis. The inner
    `stratified_scaffold_group_kfold(k=3)` split is unchanged.
    `stratify_folds` uses stratified_scaffold_group_kfold instead of plain
    scaffold_group_kfold for the internal CV folds.

    Convergence handling: `_cv_fold_losses` now distinguishes a fold that
    converged cleanly from one where statsmodels raised its "ridge
    optimization may have failed" warning (previously silently INCLUDED,
    since a warning is not an exception -- the actual bug, not the
    hypothesized "unequal sample truncation" one) or raised an outright
    exception. Both non-clean outcomes are excluded from that alpha's mean
    loss. Per-grid-point convergence counts are aggregated into this
    function's return dict (`n_folds_converged_at_alpha_min`,
    `grid_points_with_failures`, `grid_total_fold_failures`) so failures
    are visible in the per-fold CSV, not just an anonymous stderr warning.
    Pass `verbose_task_cell` (e.g. "SR-p53/HA1E") to also print a one-line
    per-call diagnostic when any grid point had a failure, for cheap
    grep-based post-hoc auditing of the run log.

    Parallelism: unlike Brent's method, grid points have no sequential
    dependency on each other, so all of them run concurrently via joblib
    (`_run_grid_jobs`, `loky` backend), each worker capped to 1 BLAS thread
    (`threadpoolctl`) so N concurrent single-threaded fits use N cores
    cleanly rather than each fit spawning its own full-width thread pool
    and contending with the others -- this, not the grid itself, was the
    actual cause of the 128-threads-on-64-cores oversubscription observed
    in the original (sequential, Brent-based) live run. This is a pure
    scheduling change: same points, same formulas, same result: it does
    not alter the alpha selected, only how fast the selection runs.
    `grid_n_jobs` defaults to `cpu_count() - 2`; pass `grid_n_jobs=1` to
    force sequential execution (e.g. for debugging).
    """
    if stratify_folds:
        folds = stratified_scaffold_group_kfold(smiles, y, k=cv_k, seed=cv_seed)
    else:
        folds = scaffold_group_kfold(smiles, k=cv_k, seed=cv_seed)
    fold_data = []
    for train_idx, test_idx in folds:
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        w_tr = inverse_prevalence_weights(y_tr) if use_sample_weight else None
        w_te = inverse_prevalence_weights(y_te) if use_sample_weight else None
        fold_data.append((train_idx, test_idx, w_tr, w_te))
    if not fold_data:
        return None

    n_grid = max(2, round((log10_max - log10_min) * points_per_decade) + 1)
    grid = np.linspace(log10_min, log10_max, n_grid)
    step = (log10_max - log10_min) / (n_grid - 1)
    min_converged_folds = cv_k

    # A grid point's mean loss is only comparable to other points if it's
    # averaged over the SAME number of folds -- a point where only 1 of
    # cv_k=3 folds converged is not "the loss at this alpha," it's one
    # fold's loss, which is just as likely to be spuriously LOW as high
    # relative to a properly-averaged 3-fold point. Found this happening in
    # practice: both the argmin and the 1-SE walk initially landed on a
    # 1-converged-fold point ~3.75 dex away from the well-supported
    # minimum, purely because that single surviving fold happened to have
    # low deviance -- not because that alpha is actually better. Points
    # below `min_converged_folds` are excluded entirely (loss -> inf) from
    # both the argmin search and the 1-SE walk, rather than silently
    # compared on unequal footing. Default requires ALL cv_k folds to
    # converge; a task/cell where this excludes most of the grid is itself
    # a finding (see METHODS.md) -- not a case to force a number for by
    # loosening this quietly.

    # Grid evaluated in ONE parallel batch, all n_grid points at once --
    # NOT staged into sequential chunks. A staged/early-stopping version was
    # tried (evaluate ascending chunks, stop once the loss curve visibly
    # flattens) specifically to skip the uninformative high-alpha tail once
    # beta saturates to ~0, but measured empirically across several chunk
    # sizes and flatness thresholds, it never beat this single-batch
    # baseline's wall-clock time -- a handful of slow-to-converge individual
    # fits dominate regardless of alpha, and sequential batches can't
    # overlap a slow outlier the way one wide parallel batch can, so the
    # chunking overhead consistently outweighed whatever points it skipped.
    # Reverted to single-batch for that reason. The SATURATION FINDING
    # itself is real and valuable independent of whether it saves compute,
    # so it's still detected below -- just as a post-hoc read of the
    # completed grid rather than a mid-search stopping rule.
    jobs = [
        functools.partial(
            _grid_point_job,
            float(lg),
            X,
            y,
            offset,
            fold_data,
            include_intercept,
            maxiter,
            cnvrg_tol,
            standardize_features,
        )
        for lg in grid
    ]
    n_jobs = min(grid_n_jobs, len(jobs))
    grid_result = dict(_run_grid_jobs(jobs, n_jobs))  # log10_alpha -> _cv_fold_losses() dict
    grid_mean_loss = {
        lg: (float(np.mean(r["losses"])) if r["n_converged"] >= min_converged_folds and r["losses"] else np.inf)
        for lg, r in grid_result.items()
    }

    # Saturation detection: once ridge regularization is strong enough that
    # beta is driven to ~0, further increasing alpha changes nothing
    # measurable -- an offset-only model penalized harder is still an
    # offset-only model. That flatness IS the finding "biology adds no
    # residual signal beyond the chemical prior for this (task, cell)," not
    # incidental -- so it's surfaced explicitly (`beta_saturated_no_bio_signal`)
    # rather than left implicit in wherever alpha_star happens to land.
    # Threshold is 1% of the argmin's own standard error (not a fixed
    # absolute constant) so it scales to this specific problem's noise
    # floor -- tried an absolute 1e-4 first; it triggered far too late for
    # a case whose se_min was ~0.18, since loss can drift by more than 1e-4
    # per step for many steps while still being noise at that scale.
    FLAT_EPSILON_REL = 0.01
    FLAT_EPSILON_ABS_FLOOR = 1e-6  # guards the rare case where se is ~0
    SATURATION_PATIENCE = 3  # consecutive flat points, past the argmin,
    # required before concluding saturation -- one flat pair could be
    # coincidence; three in a row (0.75 dex of flatness at the default
    # 4/decade spacing) is not.
    saturated_at = None
    sorted_lg_all = sorted(grid_mean_loss)
    if sorted_lg_all:
        running_argmin = min(sorted_lg_all, key=lambda lg: grid_mean_loss[lg])
        argmin_losses = grid_result[running_argmin]["losses"]
        argmin_se = (
            float(np.std(argmin_losses, ddof=1) / np.sqrt(len(argmin_losses))) if len(argmin_losses) > 1 else 0.0
        )
        flat_epsilon = max(FLAT_EPSILON_ABS_FLOOR, argmin_se * FLAT_EPSILON_REL)
        past_argmin = [lg for lg in sorted_lg_all if lg > running_argmin]
        for i in range(SATURATION_PATIENCE - 1, len(past_argmin)):
            tail = past_argmin[i - (SATURATION_PATIENCE - 1): i + 1]
            tail_deltas = [abs(grid_mean_loss[tail[j]] - grid_mean_loss[tail[j - 1]]) for j in range(1, len(tail))]
            if all(np.isfinite(grid_mean_loss[lg]) for lg in tail) and all(d < flat_epsilon for d in tail_deltas):
                saturated_at = tail[-1]
                break

    points_with_failures = [lg for lg, r in grid_result.items() if (r["n_warned"] + r["n_failed"]) > 0]
    points_excluded_thin = [lg for lg, r in grid_result.items() if r["n_converged"] < min_converged_folds]
    total_fold_failures = sum(r["n_warned"] + r["n_failed"] for r in grid_result.values())
    if verbose_task_cell and saturated_at is not None:
        print(f"[_select_alpha] {verbose_task_cell}: loss curve saturated at alpha=1e{saturated_at:.2f} -- "
              f"beta driven to ~0, no additional information from further regularization: no bio signal "
              f"beyond the chemical prior detected for this (task, cell).")
    if verbose_task_cell and points_with_failures:
        worst = sorted(points_with_failures, key=lambda lg: grid_result[lg]["n_warned"] + grid_result[lg]["n_failed"], reverse=True)[:3]
        detail = ", ".join(f"alpha=1e{lg:.2f}({grid_result[lg]['n_warned']}w/{grid_result[lg]['n_failed']}f)" for lg in worst)
        print(f"[_select_alpha] {verbose_task_cell}: {total_fold_failures} fold-fit convergence issues "
              f"across {len(points_with_failures)}/{n_grid} grid points ({len(points_excluded_thin)} excluded "
              f"entirely for <{min_converged_folds}/{cv_k} converged folds) -- worst: {detail}")
    if all(np.isinf(loss) for loss in grid_mean_loss.values()):
        return None

    log10_alpha_min = min(grid_mean_loss, key=grid_mean_loss.get)
    mean_loss_min = grid_mean_loss[log10_alpha_min]
    losses_at_min = grid_result[log10_alpha_min]["losses"]
    n_folds_at_min = grid_result[log10_alpha_min]["n_converged"]
    se_min = float(np.std(losses_at_min, ddof=1) / np.sqrt(len(losses_at_min))) if len(losses_at_min) > 1 else 0.0

    threshold = mean_loss_min + se_min

    # 1-SE rule: walk the CONTIGUOUS plateau of grid points extending from
    # the minimum toward higher alpha (more regularization), stopping at
    # the first point whose loss exceeds threshold -- NOT "any grid point
    # anywhere within threshold." The loss-vs-alpha curve is not guaranteed
    # unimodal (fold noise, or a genuinely second low-loss regime at a
    # distant alpha), so an unrestricted "take the max of all qualifying
    # points" can jump to an unrelated, disconnected low-loss island far
    # from the actual minimum -- found exactly this happening during
    # validation (a fold jumping ~3.75 dex to a distant point) before this
    # contiguity walk was added. Deploying that distant point would not be
    # "the simplest model within noise of the best one," which is what the
    # 1-SE rule is actually supposed to select.
    sorted_lg = sorted(grid_mean_loss)
    min_pos = sorted_lg.index(log10_alpha_min)
    star_pos = min_pos
    for pos in range(min_pos + 1, len(sorted_lg)):
        if grid_mean_loss[sorted_lg[pos]] <= threshold:
            star_pos = pos
        else:
            break
    log10_alpha_star = sorted_lg[star_pos]

    # alpha_boundary_hit checks the DEPLOYED value (alpha_star), not the
    # argmin -- the 1-SE walk can (and, once beta saturates toward 0 under
    # strong enough regularization, routinely does) extend well past the
    # argmin's own position out to the grid's hard edge, since the loss
    # curve goes genuinely flat once beta is already ~0 and increasing
    # alpha further changes nothing measurable. Checking only the argmin
    # position (the original version of this line) missed exactly that
    # case -- confirmed in production: NR-AR/HA1E selected alpha_star=1e8
    # (log10=8=grid_log10_max) in all 5 outer folds with the old
    # argmin-only check reporting alpha_boundary_hit=False in every one.
    # A hit here does not necessarily mean truncation error, though: once
    # beta has saturated to ~0, alpha_star=1e8 and alpha_star=1e5 are
    # numerically indistinguishable in their effect on p_final, so this
    # flag should be read alongside delta_auprc (near-zero delta + boundary
    # hit is the saturation signature; a large delta at the boundary would
    # be the genuine-truncation-risk signature worth widening the grid for).
    alpha_min_boundary_hit = bool(log10_alpha_min <= log10_min + step / 2 or log10_alpha_min >= log10_max - step / 2)
    alpha_star_boundary_hit = bool(log10_alpha_star <= log10_min + step / 2 or log10_alpha_star >= log10_max - step / 2)

    # Plateau diagnostics -- grid points themselves are the probe, no
    # separate multiplier pass needed.
    ten_x_window = [lg for lg in grid_mean_loss if log10_alpha_min <= lg <= log10_alpha_min + 1.0]
    is_plateau_10x = bool(ten_x_window) and all(grid_mean_loss[lg] <= threshold for lg in ten_x_window)
    plateau_extends_beyond_10x = any(
        loss <= threshold for lg, loss in grid_mean_loss.items() if lg > log10_alpha_star + 1e-9
    )
    plateau_deltas_10x = [grid_mean_loss[lg] - mean_loss_min for lg in ten_x_window]

    result = {
        "alpha_star": float(10.0 ** log10_alpha_star),
        "alpha_min_loss": float(10.0 ** log10_alpha_min),
        "cv_loss_at_alpha_min": mean_loss_min,
        "cv_loss_se_at_alpha_min": se_min,
        "alpha_boundary_hit": alpha_star_boundary_hit,
        "alpha_min_boundary_hit": alpha_min_boundary_hit,
        # True when the search stopped early because the loss curve went
        # flat (beta saturated to ~0): a direct, explicit finding that
        # biology adds no detectable residual signal beyond the chemical
        # prior for this (task, cell), not merely a compute optimization.
        # False does not imply a real bio signal exists -- only that either
        # a genuine minimum with a bounded plateau was found, or the full
        # grid was exhausted without one (see alpha_boundary_hit for that
        # latter case).
        "beta_saturated_no_bio_signal": saturated_at is not None,
        "n_folds_converged_at_alpha_min": n_folds_at_min,
        "n_folds_total": len(fold_data),
        "grid_points_with_failures": len(points_with_failures),
        "grid_total_fold_failures": total_fold_failures,
        "grid_n_points": n_grid,
        "grid_n_points_evaluated": len(grid_result),
        "grid_log10_min": log10_min,
        "grid_log10_max": log10_max,
        "grid_points_per_decade": points_per_decade,
        "alpha_plateau_low": float(10.0 ** log10_alpha_min),
        "alpha_plateau_high": float(10.0 ** log10_alpha_star),
        "is_plateau_10x": is_plateau_10x,
        "plateau_extends_beyond_10x": bool(plateau_extends_beyond_10x),
        "plateau_max_delta_loss_10x": float(max(plateau_deltas_10x)) if plateau_deltas_10x else 0.0,
        # Kept for backward compatibility with any code inspecting these --
        # both are structurally meaningless now (no Brent, no re-bracketing)
        # but their absence would be a silent schema break for readers that
        # don't guard with .get().
        "re_bracketed": False,
        "brent_converged": True,
    }
    if light:
        return {k: result[k] for k in (
            "alpha_star", "alpha_min_loss", "cv_loss_at_alpha_min", "cv_loss_se_at_alpha_min",
            "alpha_boundary_hit", "alpha_min_boundary_hit", "beta_saturated_no_bio_signal",
            "re_bracketed", "brent_converged",
            "n_folds_converged_at_alpha_min", "n_folds_total",
            "grid_points_with_failures", "grid_total_fold_failures",
            "grid_n_points", "grid_n_points_evaluated",
        )}
    return result


# ---------------------------------------------------------------------------
# stratified bootstrap CI on a paired AUPRC/AUROC delta
# ---------------------------------------------------------------------------
def stratified_bootstrap_delta_ci(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_method: str = "bca",
) -> dict:
    """
    Paired bootstrap CI for AUPRC/AUROC delta (metric(p_b) - metric(p_a)),
    e.g. chem-only (p_a) vs. chem+gene (p_b), on one fixed held-out set.
    Resamples positive- and negative-labeled rows SEPARATELY (each at their
    own observed count) so every resample has exactly n_pos/n_neg -- avoids
    degenerate single-class draws under this study's class imbalance.
    """
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    n = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    def _delta(y_s, pa_s, pb_s):
        ma = compute_binary_metrics(y_s, pa_s)
        mb = compute_binary_metrics(y_s, pb_s)
        return mb["auprc"] - ma["auprc"], mb["auroc"] - ma["auroc"]

    theta_auprc, theta_auroc = _delta(y, p_a, p_b)

    rng = np.random.RandomState(seed)
    deltas_auprc, deltas_auroc = [], []
    for _ in range(n_bootstrap):
        draw_pos = pos_idx[rng.randint(0, n_pos, size=n_pos)] if n_pos > 0 else np.array([], dtype=int)
        draw_neg = neg_idx[rng.randint(0, n_neg, size=n_neg)] if n_neg > 0 else np.array([], dtype=int)
        idx = np.concatenate([draw_pos, draw_neg])
        da, dr = _delta(y[idx], p_a[idx], p_b[idx])
        if np.isnan(da) or np.isnan(dr):
            continue
        deltas_auprc.append(da)
        deltas_auroc.append(dr)

    n_valid = len(deltas_auprc)
    if n_valid == 0:
        return {
            "auprc_ci_low": float("nan"), "auprc_ci_high": float("nan"), "auprc_frac_positive": float("nan"),
            "auroc_ci_low": float("nan"), "auroc_ci_high": float("nan"),
            "n_bootstrap_valid": 0, "n_bootstrap_requested": int(n_bootstrap),
            "n_pos": int(n_pos), "n_neg": int(n_neg), "ci_method_used": ci_method,
        }

    deltas_auprc = np.array(deltas_auprc)
    deltas_auroc = np.array(deltas_auroc)

    def _jackknife_delta(full_delta_fn):
        jack_vals = np.empty(n)
        for i in range(n):
            keep = np.ones(n, dtype=bool)
            keep[i] = False
            jack_vals[i] = full_delta_fn(y[keep], p_a[keep], p_b[keep])
        return jack_vals

    def _ci_for(theta_hat, boot_deltas, full_delta_fn):
        if ci_method == "percentile":
            return float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5)), "percentile"

        prop_less = np.clip(np.mean(boot_deltas < theta_hat), 1e-6, 1 - 1e-6)
        z0 = norm.ppf(prop_less)

        if ci_method == "basic":
            lo = np.percentile(boot_deltas, 2.5)
            hi = np.percentile(boot_deltas, 97.5)
            return float(2 * theta_hat - hi), float(2 * theta_hat - lo), "basic"

        jack_vals = _jackknife_delta(full_delta_fn)
        jack_mean = jack_vals.mean()
        num = np.sum((jack_mean - jack_vals) ** 3)
        den = 6.0 * (np.sum((jack_mean - jack_vals) ** 2) ** 1.5)
        if den == 0 or not np.isfinite(num / den):
            warnings.warn(
                "[stratified_bootstrap_delta_ci] BCa acceleration undefined "
                "(degenerate jackknife) -- falling back to basic bootstrap CI"
            )
            lo = np.percentile(boot_deltas, 2.5)
            hi = np.percentile(boot_deltas, 97.5)
            return float(2 * theta_hat - hi), float(2 * theta_hat - lo), "basic_fallback"

        a = num / den
        z_lo, z_hi = norm.ppf(0.025), norm.ppf(0.975)

        def _adjusted_percentile(z):
            return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

        p_lo = np.clip(_adjusted_percentile(z_lo), 0, 1)
        p_hi = np.clip(_adjusted_percentile(z_hi), 0, 1)
        lo = np.percentile(boot_deltas, 100 * p_lo)
        hi = np.percentile(boot_deltas, 100 * p_hi)
        return float(lo), float(hi), "bca"

    def _full_auprc_delta(y_s, pa_s, pb_s):
        d, _ = _delta(y_s, pa_s, pb_s)
        return d

    def _full_auroc_delta(y_s, pa_s, pb_s):
        _, d = _delta(y_s, pa_s, pb_s)
        return d

    auprc_lo, auprc_hi, method_used = _ci_for(theta_auprc, deltas_auprc, _full_auprc_delta)
    auroc_lo, auroc_hi, _ = _ci_for(theta_auroc, deltas_auroc, _full_auroc_delta)

    return {
        "auprc_ci_low": auprc_lo, "auprc_ci_high": auprc_hi,
        "auprc_frac_positive": float(np.mean(deltas_auprc > 0)),
        "auroc_ci_low": auroc_lo, "auroc_ci_high": auroc_hi,
        "n_bootstrap_valid": int(n_valid), "n_bootstrap_requested": int(n_bootstrap),
        "n_pos": int(n_pos), "n_neg": int(n_neg), "ci_method_used": method_used,
    }
