# linear_offset_logistic/nadeau_bengio_test.py
"""
Nadeau & Bengio (2003, "Inference for the generalization error", Machine
Learning 52(3):239-281) corrected resampled t-test/CI for a set of paired
performance-difference draws (e.g. delta_auprc) coming from REPEATED
k-fold (or nested) cross-validation.

Why this exists, not a plain t-test or percentile CI on the pooled draws:
repeated-CV fold estimates are NOT independent samples -- folds across
different repeats share overlapping training rows (and, within one
repeat, different outer folds share most of their training data with each
other too), so the usual "variance / n" standard-error formula
systematically UNDERSTATES the true variance, making a naive CI too
narrow and a naive p-value too small. Nadeau & Bengio's correction
inflates the variance estimate by a factor that accounts for the
train/test size ratio:

    sigma^2_corrected = s^2 * (1/n + n_test/n_train)

where s^2 is the ordinary sample variance of the n paired differences,
n_test/n_train is the (mean) ratio of held-out to training-fold size in
the resampling scheme that produced them. This is the standard
"corrected resampled t-test" already flagged, in this exact project's own
history, as the statistically correct tool for CV-native significance
testing that had been discussed but never implemented -- this module
implements it.

This does NOT correct for alpha-selection/gene-ranking instability itself
(that's what feeding it REPEATED, freshly-refit nested-CV draws already
does -- see repeated_nested_cv_offset_logistic.py); it corrects the
variance ESTIMATE computed from those draws so the resulting test/CI
isn't anti-conservative.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def nadeau_bengio_corrected_test(
    deltas: np.ndarray,
    n_train: np.ndarray,
    n_test: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    deltas: 1D array of paired performance differences (e.g. delta_auprc),
        one per (repeat, outer_fold) draw -- NOT averaged within a repeat
        first; every individual fold-level draw is its own observation.
    n_train, n_test: 1D arrays, same length as `deltas`, the outer-fold
        train/test set sizes that produced each draw (sizes can vary
        slightly across folds/repeats under scaffold-grouped splitting,
        so the ratio is computed per-draw and averaged, not assumed fixed).

    Returns dict: n, mean, naive_se, corrected_se, correction_factor,
    t_stat, p_value (two-sided), ci_low, ci_high (both at `1-alpha`), df.
    """
    deltas = np.asarray(deltas, dtype=float)
    n_train = np.asarray(n_train, dtype=float)
    n_test = np.asarray(n_test, dtype=float)
    n = len(deltas)
    if n < 2:
        raise ValueError(f"need at least 2 draws for a variance estimate, got n={n}")

    mean = float(deltas.mean())
    naive_var = float(deltas.var(ddof=1))
    naive_se = float(np.sqrt(naive_var / n))

    mean_ratio = float((n_test / n_train).mean())
    correction_factor = 1.0 / n + mean_ratio
    corrected_var = naive_var * correction_factor
    corrected_se = float(np.sqrt(corrected_var))

    df = n - 1
    t_stat = mean / corrected_se if corrected_se > 0 else np.inf * np.sign(mean) if mean != 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t_stat), df)) if corrected_se > 0 else (0.0 if mean != 0 else 1.0)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_low = mean - t_crit * corrected_se
    ci_high = mean + t_crit * corrected_se

    return {
        "n": n, "mean": mean,
        "naive_se": naive_se, "corrected_se": corrected_se,
        "correction_factor": correction_factor, "mean_test_train_ratio": mean_ratio,
        "t_stat": float(t_stat), "p_value": p_value, "df": df,
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "significant_nb": bool(ci_low > 0 or ci_high < 0),
    }
