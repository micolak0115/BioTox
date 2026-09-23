"""Fast fixed-offset ridge logistic regression for calibrated Stage 2.

The fitted model is

    logit(P(y_i = 1)) = eta_chem_i + intercept + X_i @ beta

where ``eta_chem`` is a fixed coefficient-one offset, ``intercept`` is
unpenalized, and only ``beta`` receives an L2 penalty.  The implementation
uses an analytic gradient with L-BFGS-B and supports high-to-low
regularization-path warm starts.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit


@dataclass(frozen=True)
class OffsetInterceptFit:
    intercept: float
    objective: float
    gradient_abs: float
    elapsed_seconds: float


@dataclass(frozen=True)
class OffsetRidgeFit:
    params: np.ndarray
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
    sample_weighted: bool

    @property
    def intercept(self) -> float:
        return float(self.params[0])

    @property
    def beta(self) -> np.ndarray:
        return np.asarray(self.params[1:], dtype=np.float64)


def _as_float_vector(values, name: str, n_expected: int | None = None) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).reshape(-1)
    if n_expected is not None and out.shape[0] != n_expected:
        raise ValueError(f"{name} has {out.shape[0]} rows; expected {n_expected}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _validate_inputs(X, y, offset, sample_weight=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")

    y = _as_float_vector(y, "y", X.shape[0])
    offset = _as_float_vector(offset, "offset", X.shape[0])
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y must contain only binary labels 0 and 1")
    if np.unique(y).size != 2:
        raise ValueError("y must contain both classes")

    if sample_weight is None:
        weight = np.ones(X.shape[0], dtype=np.float64)
        weighted = False
    else:
        weight = _as_float_vector(sample_weight, "sample_weight", X.shape[0])
        if np.any(weight <= 0):
            raise ValueError("sample_weight must be strictly positive")
        weighted = True
    return X, y, offset, weight, weighted


def offset_ridge_objective_gradient(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return the mean penalized negative log-likelihood and its gradient."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    params = np.asarray(params, dtype=np.float64).reshape(-1)
    if params.shape[0] != X.shape[1] + 1:
        raise ValueError("params must contain one intercept and one coefficient per column")

    intercept = params[0]
    beta = params[1:]
    linear_predictor = offset + intercept + X @ beta
    per_sample_loss = np.logaddexp(0.0, linear_predictor) - y * linear_predictor

    if sample_weight is None:
        denominator = float(y.shape[0])
        residual = expit(linear_predictor) - y
        data_loss = float(np.sum(per_sample_loss) / denominator)
        intercept_gradient = float(np.sum(residual) / denominator)
        beta_gradient = X.T @ residual / denominator
    else:
        weight = np.asarray(sample_weight, dtype=np.float64)
        denominator = float(np.sum(weight))
        weighted_residual = weight * (expit(linear_predictor) - y)
        data_loss = float(np.sum(weight * per_sample_loss) / denominator)
        intercept_gradient = float(np.sum(weighted_residual) / denominator)
        beta_gradient = X.T @ weighted_residual / denominator

    objective = data_loss + 0.5 * float(alpha) * float(beta @ beta)
    gradient = np.concatenate(
        [[intercept_gradient], beta_gradient + float(alpha) * beta]
    )
    return float(objective), np.asarray(gradient, dtype=np.float64)


def fit_offset_intercept_lbfgs(
    y: np.ndarray,
    offset: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> OffsetInterceptFit:
    """Fit ``offset + intercept`` by solving its monotone score equation."""
    y = _as_float_vector(y, "y")
    offset = _as_float_vector(offset, "offset", y.shape[0])
    if not np.all(np.isin(y, [0.0, 1.0])) or np.unique(y).size != 2:
        raise ValueError("y must contain both binary classes")

    if sample_weight is None:
        weight = np.ones_like(y)
    else:
        weight = _as_float_vector(sample_weight, "sample_weight", y.shape[0])
        if np.any(weight <= 0):
            raise ValueError("sample_weight must be strictly positive")
    denominator = float(np.sum(weight))

    def score(intercept: float) -> float:
        return float(np.sum(weight * (expit(offset + intercept) - y)) / denominator)

    started = perf_counter()
    lower, upper = -40.0, 40.0
    while score(lower) > 0:
        lower *= 2.0
    while score(upper) < 0:
        upper *= 2.0
    intercept = float(brentq(score, lower, upper, xtol=1e-12, rtol=1e-12))
    eta = offset + intercept
    loss = np.logaddexp(0.0, eta) - y * eta
    objective = float(np.sum(weight * loss) / denominator)
    return OffsetInterceptFit(
        intercept=intercept,
        objective=objective,
        gradient_abs=abs(score(intercept)),
        elapsed_seconds=perf_counter() - started,
    )


def _minimize_once(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    alpha: float,
    start_params: np.ndarray,
    sample_weight: np.ndarray | None,
    maxiter: int,
    gtol: float,
    ftol: float,
):
    def objective_and_gradient(params):
        return offset_ridge_objective_gradient(
            params, X, y, offset, alpha, sample_weight=sample_weight
        )

    return minimize(
        objective_and_gradient,
        np.asarray(start_params, dtype=np.float64),
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


def fit_offset_ridge_lbfgs(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    alpha: float,
    start_params: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    maxiter: int = 500,
    retry_maxiter: int = 2000,
    gtol: float = 1e-6,
    acceptance_gradient: float = 1e-5,
    ftol: float = 1e-14,
) -> OffsetRidgeFit:
    """Fit one ridge value, retrying cold only when convergence is inadequate."""
    X, y, offset, weight, weighted = _validate_inputs(X, y, offset, sample_weight)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    null_fit = fit_offset_intercept_lbfgs(
        y, offset, sample_weight=weight if weighted else None
    )
    cold_start = np.concatenate([[null_fit.intercept], np.zeros(X.shape[1])])
    warm_started = start_params is not None
    if start_params is None:
        initial = cold_start
    else:
        initial = _as_float_vector(start_params, "start_params", X.shape[1] + 1)

    started = perf_counter()
    result = _minimize_once(
        X,
        y,
        offset,
        float(alpha),
        initial,
        weight if weighted else None,
        maxiter,
        gtol,
        ftol,
    )
    objective, gradient = offset_ridge_objective_gradient(
        result.x,
        X,
        y,
        offset,
        float(alpha),
        sample_weight=weight if weighted else None,
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
            weight if weighted else None,
            retry_maxiter,
            min(gtol, acceptance_gradient / 10.0),
            min(ftol, 1e-12),
        )
        objective, gradient = offset_ridge_objective_gradient(
            result.x,
            X,
            y,
            offset,
            float(alpha),
            sample_weight=weight if weighted else None,
        )
        gradient_inf = float(np.linalg.norm(gradient, ord=np.inf))
        valid = bool(
            result.success
            and np.isfinite(objective)
            and np.all(np.isfinite(result.x))
            and gradient_inf <= acceptance_gradient
        )

    return OffsetRidgeFit(
        params=np.asarray(result.x, dtype=np.float64),
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
        sample_weighted=weighted,
    )


def fit_offset_ridge_path(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    alphas: Iterable[float],
    sample_weight: np.ndarray | None = None,
    maxiter: int = 500,
    retry_maxiter: int = 2000,
    gtol: float = 1e-6,
    acceptance_gradient: float = 1e-5,
    ftol: float = 1e-14,
) -> list[OffsetRidgeFit]:
    """Fit a descending alpha path, warm-starting from the preceding solution."""
    alpha_values = sorted({float(alpha) for alpha in alphas}, reverse=True)
    if not alpha_values:
        raise ValueError("alphas must contain at least one value")
    if alpha_values[-1] < 0:
        raise ValueError("alphas must be non-negative")

    fitted: list[OffsetRidgeFit] = []
    start_params = None
    for alpha in alpha_values:
        fit = fit_offset_ridge_lbfgs(
            X,
            y,
            offset,
            alpha,
            start_params=start_params,
            sample_weight=sample_weight,
            maxiter=maxiter,
            retry_maxiter=retry_maxiter,
            gtol=gtol,
            acceptance_gradient=acceptance_gradient,
            ftol=ftol,
        )
        fitted.append(fit)
        start_params = fit.params if fit.success else None
    return fitted


def predict_offset_ridge_lbfgs(
    fitted: OffsetRidgeFit,
    X: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    offset = _as_float_vector(offset, "offset", X.shape[0])
    if X.ndim != 2 or X.shape[1] != fitted.beta.shape[0]:
        raise ValueError("X has an incompatible shape")
    linear_predictor = offset + fitted.intercept + X @ fitted.beta
    if not np.all(np.isfinite(linear_predictor)):
        raise RuntimeError("prediction produced non-finite logits")
    return expit(linear_predictor), linear_predictor
