"""Ridge logistic regression with a fully fixed offset and no intercept.

The fitted model is

    logit(P(y_i = 1)) = offset_i + X_i @ beta

where the complete offset is fixed with coefficient one and only ``beta`` is
estimated. This solver is kept separate from ``offset_ridge_lbfgs.py``, whose
model includes an unpenalized fitted intercept.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class FixedOffsetRidgeFit:
    beta: np.ndarray
    alpha: float
    success: bool
    message: str
    n_iterations: int
    n_function_evaluations: int
    gradient_inf_norm: float
    objective: float
    elapsed_seconds: float
    warm_started: bool
    retry_used: bool


def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if X.shape[0] != y.shape[0] or y.shape[0] != offset.shape[0]:
        raise ValueError("X, y, and offset must have the same number of rows")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(offset)):
        raise ValueError("y or offset contains non-finite values")
    if not np.all(np.isin(y, [0.0, 1.0])) or np.unique(y).size != 2:
        raise ValueError("y must contain both binary classes")
    return X, y, offset


def fixed_offset_ridge_objective_gradient(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    """Return mean penalized negative log likelihood and its gradient."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    if beta.shape[0] != X.shape[1]:
        raise ValueError("beta has an incompatible dimension")
    eta = offset + X @ beta
    residual = expit(eta) - y
    objective = float(
        np.mean(np.logaddexp(0.0, eta) - y * eta)
        + 0.5 * float(alpha) * float(beta @ beta)
    )
    gradient = X.T @ residual / float(y.shape[0]) + float(alpha) * beta
    return objective, np.asarray(gradient, dtype=np.float64)


def _minimize_once(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    alpha: float,
    start_beta: np.ndarray,
    maxiter: int,
    gtol: float,
    ftol: float,
):
    def objective_and_gradient(beta):
        return fixed_offset_ridge_objective_gradient(
            beta, X, y, offset, alpha
        )

    return minimize(
        objective_and_gradient,
        np.asarray(start_beta, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(maxiter),
            "gtol": float(gtol),
            "ftol": float(ftol),
            "maxls": 50,
            "maxcor": 20,
        },
    )


def fit_fixed_offset_ridge_lbfgs(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    alpha: float,
    *,
    start_beta: np.ndarray | None = None,
    maxiter: int = 500,
    retry_maxiter: int = 2000,
    gtol: float = 1e-6,
    acceptance_gradient: float = 1e-5,
    ftol: float = 1e-14,
) -> FixedOffsetRidgeFit:
    X, y, offset = _validate_inputs(X, y, offset)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    cold_start = np.zeros(X.shape[1], dtype=np.float64)
    warm_started = start_beta is not None
    initial = (
        cold_start
        if start_beta is None
        else np.asarray(start_beta, dtype=np.float64).reshape(-1)
    )
    if initial.shape[0] != X.shape[1]:
        raise ValueError("start_beta has an incompatible dimension")

    started = perf_counter()
    result = _minimize_once(
        X, y, offset, float(alpha), initial, maxiter, gtol, ftol
    )
    objective, gradient = fixed_offset_ridge_objective_gradient(
        result.x, X, y, offset, float(alpha)
    )
    gradient_inf = float(np.linalg.norm(gradient, ord=np.inf))
    valid = bool(
        result.success
        and np.isfinite(objective)
        and np.all(np.isfinite(result.x))
        and gradient_inf <= acceptance_gradient
    )

    retry_used = False
    if not valid:
        retry_used = True
        result = _minimize_once(
            X,
            y,
            offset,
            float(alpha),
            cold_start,
            retry_maxiter,
            min(gtol, acceptance_gradient / 10.0),
            min(ftol, 1e-12),
        )
        objective, gradient = fixed_offset_ridge_objective_gradient(
            result.x, X, y, offset, float(alpha)
        )
        gradient_inf = float(np.linalg.norm(gradient, ord=np.inf))
        valid = bool(
            result.success
            and np.isfinite(objective)
            and np.all(np.isfinite(result.x))
            and gradient_inf <= acceptance_gradient
        )

    return FixedOffsetRidgeFit(
        beta=np.asarray(result.x, dtype=np.float64),
        alpha=float(alpha),
        success=valid,
        message=str(result.message),
        n_iterations=int(result.nit),
        n_function_evaluations=int(result.nfev),
        gradient_inf_norm=gradient_inf,
        objective=float(objective),
        elapsed_seconds=perf_counter() - started,
        warm_started=warm_started,
        retry_used=retry_used,
    )


def fit_fixed_offset_ridge_path(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    alphas: Iterable[float],
    *,
    maxiter: int = 500,
    retry_maxiter: int = 2000,
    gtol: float = 1e-6,
    acceptance_gradient: float = 1e-5,
    ftol: float = 1e-14,
) -> list[FixedOffsetRidgeFit]:
    alpha_values = sorted({float(alpha) for alpha in alphas}, reverse=True)
    if not alpha_values:
        raise ValueError("alphas must contain at least one value")
    if alpha_values[-1] < 0:
        raise ValueError("alphas must be non-negative")

    fits: list[FixedOffsetRidgeFit] = []
    start_beta = None
    for alpha in alpha_values:
        fit = fit_fixed_offset_ridge_lbfgs(
            X,
            y,
            offset,
            alpha,
            start_beta=start_beta,
            maxiter=maxiter,
            retry_maxiter=retry_maxiter,
            gtol=gtol,
            acceptance_gradient=acceptance_gradient,
            ftol=ftol,
        )
        fits.append(fit)
        start_beta = fit.beta if fit.success else None
    return fits


def predict_fixed_offset_ridge(
    fitted: FixedOffsetRidgeFit,
    X: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[1] != fitted.beta.shape[0]:
        raise ValueError("X has an incompatible shape")
    if X.shape[0] != offset.shape[0]:
        raise ValueError("X and offset must have the same number of rows")
    eta = offset + X @ fitted.beta
    if not np.all(np.isfinite(eta)):
        raise RuntimeError("prediction produced non-finite logits")
    return expit(eta), eta
