"""Inference helpers specific to repeated cross-validation."""
from __future__ import annotations

import numpy as np

from publication.nadeau_bengio_test import nadeau_bengio_corrected_test


def repeated_cv_nadeau_bengio_test(
    deltas: np.ndarray,
    n_train: np.ndarray,
    n_test: np.ndarray,
    alpha: float = 0.05,
    variance_tolerance: float = 1e-15,
) -> dict:
    """Run corrected paired inference with an explicit degeneracy guard.

    Exact zero empirical variance cannot support a finite variance estimate
    from repeated resampling. Returning a zero p-value in that situation
    would overstate evidence, so the result is marked non-inferential.
    """
    deltas = np.asarray(deltas, dtype=float)
    n_train = np.asarray(n_train, dtype=float)
    n_test = np.asarray(n_test, dtype=float)
    if not (
        deltas.ndim == n_train.ndim == n_test.ndim == 1
        and len(deltas) == len(n_train) == len(n_test)
    ):
        raise ValueError("deltas, n_train, and n_test must be aligned 1D arrays")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("Repeated-CV deltas contain non-finite values")
    if np.any(n_train <= 0) or np.any(n_test <= 0):
        raise ValueError("Repeated-CV train/test sizes must be positive")

    result = nadeau_bengio_corrected_test(
        deltas,
        n_train,
        n_test,
        alpha=alpha,
    )
    empirical_variance = float(np.var(deltas, ddof=1))
    degenerate = empirical_variance <= variance_tolerance
    result["empirical_variance"] = empirical_variance
    result["variance_degenerate"] = degenerate
    result["inference_valid"] = not degenerate
    if degenerate:
        result.update(
            {
                "t_stat": float("nan"),
                "p_value": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "significant_nb": False,
            }
        )
    return result
