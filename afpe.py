from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar

from .estimators import (
    leave_one_out_conditional_mean_at_design,
    leave_one_out_density_at_design,
    lagged_matrix,
    response_vector,
    validate_lags,
)
from .kernels import KernelName, base_kernel_at_zero


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("inf")
    return float(np.mean(finite))


def A_est(series: Iterable[float], lags: Iterable[int], kern: KernelName = "gaussian"):
    x = np.asarray(series, dtype=float)
    lag_arr = validate_lags(lags)
    max_lag = int(np.max(lag_arr))
    design = lagged_matrix(x, lag_arr, start=max_lag, stop=len(x))
    targets = response_vector(x, lag_arr)

    def objective(h: float) -> float:
        pred = leave_one_out_conditional_mean_at_design(design, targets, kern, h)
        return _finite_mean((targets - pred) ** 2)

    return objective


def B_est(series: Iterable[float], lags: Iterable[int], kern: KernelName = "gaussian"):
    x = np.asarray(series, dtype=float)
    lag_arr = validate_lags(lags)
    max_lag = int(np.max(lag_arr))
    design = lagged_matrix(x, lag_arr, start=max_lag, stop=len(x))
    targets = response_vector(x, lag_arr)

    def objective(h: float) -> float:
        pred = leave_one_out_conditional_mean_at_design(design, targets, kern, h)
        density = leave_one_out_density_at_design(design, kern, h)
        weighted = (targets - pred) ** 2 / np.where(np.abs(density) < 1e-14, np.nan, density)
        return _finite_mean(weighted)

    return objective


def AFPE_opt(
    series: Iterable[float],
    lags: Iterable[int],
    kern: KernelName = "gaussian",
    h_bounds: tuple[float, float] = (0.0005, 50.0),
) -> tuple[float, float]:
    x = np.asarray(series, dtype=float)
    lag_arr = validate_lags(lags)
    m = len(lag_arr)
    n_eff = len(x) - int(np.max(lag_arr))
    if n_eff <= 1:
        raise ValueError("series length must exceed max(lags) by at least two observations.")
    a_obj = A_est(x, lag_arr, kern)
    b_obj = B_est(x, lag_arr, kern)

    def fpe_objective(h: float) -> float:
        a_value = a_obj(h)
        b_value = b_obj(h)
        if not np.isfinite(a_value) or not np.isfinite(b_value):
            return float("inf")
        return float(a_value + 2.0 * (base_kernel_at_zero(kern) ** m) * b_value / (n_eff * (h**m)))

    opt = minimize_scalar(fpe_objective, bounds=h_bounds, method="bounded")
    if not opt.success:
        raise RuntimeError(f"AFPE optimization failed: {opt.message}")
    return float(opt.fun), float(opt.x)


def AFPE_combn(
    series: Iterable[float],
    m: int,
    k: int,
    kern: KernelName = "gaussian",
    h_bounds: tuple[float, float] = (0.0005, 50.0),
) -> dict[str, list]:
    if m <= 0 or k <= 0:
        raise ValueError("m and k must be positive integers.")
    if m > k:
        raise ValueError("m cannot exceed k.")
    fpe_values: list[float] = []
    picks: list[tuple[int, ...]] = []
    bandwidths: list[float] = []
    for size in range(1, m + 1):
        for lag_set in combinations(range(1, k + 1), size):
            fpe, bandwidth = AFPE_opt(series, lag_set, kern=kern, h_bounds=h_bounds)
            fpe_values.append(fpe)
            picks.append(lag_set)
            bandwidths.append(bandwidth)
    return {"FPE": fpe_values, "lags": picks, "band": bandwidths}


def AFPE_proc(
    series: Iterable[float],
    m: int,
    k: int,
    kern: KernelName = "gaussian",
    h_bounds: tuple[float, float] = (0.0005, 50.0),
) -> dict[str, object]:
    fpe_list = AFPE_combn(series, m, k, kern=kern, h_bounds=h_bounds)
    values = np.asarray(fpe_list["FPE"], dtype=float)
    if not np.any(np.isfinite(values)):
        raise RuntimeError("No finite AFPE value was obtained.")
    idx = int(np.nanargmin(values))
    return {"FPE": fpe_list["FPE"][idx], "lags": fpe_list["lags"][idx], "band": fpe_list["band"][idx]}
