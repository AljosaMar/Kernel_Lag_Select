from __future__ import annotations

from typing import Iterable

import numpy as np


VALID_MODELS = {"linear", "polynomial", "expo", "expo2", "treshold", "threshold"}


def _as_1d_float_array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr


def linear(
    eps: Iterable[float],
    dim: int,
    lags: Iterable[int],
    coeff: Iterable[float],
    initial_values: Iterable[float],
) -> np.ndarray:
    eps_arr = _as_1d_float_array(eps, "eps")
    lags_arr = np.asarray(list(lags), dtype=int)
    coeff_arr = _as_1d_float_array(coeff, "coeff")
    initial_arr = _as_1d_float_array(initial_values, "initial_values")
    if dim != len(coeff_arr) or dim != len(lags_arr):
        raise ValueError("dim must equal the number of coefficients and lags.")
    if np.any(lags_arr <= 0):
        raise ValueError("lags must be positive integers.")
    max_lag = int(np.max(lags_arr))
    if len(initial_arr) < max_lag:
        raise ValueError("initial_values must contain at least max(lags) entries.")
    x = eps_arr.copy()
    x[:max_lag] = initial_arr[:max_lag]
    for t in range(max_lag, len(eps_arr)):
        x[t] = float(np.dot(coeff_arr, x[t - lags_arr])) + eps_arr[t]
    return x


def polynomial(
    eps: Iterable[float],
    lags: Iterable[int],
    coeff: Iterable[float],
    exponents: Iterable[float],
    initial_values: Iterable[float],
) -> np.ndarray:
    eps_arr = _as_1d_float_array(eps, "eps")
    lags_arr = np.asarray(list(lags), dtype=int)
    coeff_arr = _as_1d_float_array(coeff, "coeff")
    exp_arr = _as_1d_float_array(exponents, "exponents")
    initial_arr = _as_1d_float_array(initial_values, "initial_values")
    if len(exp_arr) != len(coeff_arr) or len(exp_arr) != len(lags_arr):
        raise ValueError("exponents, coefficients, and lags must have equal length.")
    if np.any(lags_arr <= 0):
        raise ValueError("lags must be positive integers.")
    max_lag = int(np.max(lags_arr))
    if len(initial_arr) < max_lag:
        raise ValueError("initial_values must contain at least max(lags) entries.")
    x = eps_arr.copy()
    x[:max_lag] = initial_arr[:max_lag]
    for t in range(max_lag, len(eps_arr)):
        x[t] = float(np.dot(coeff_arr, x[t - lags_arr] ** exp_arr)) + eps_arr[t]
    return x


poly = polynomial


def treshold(eps: Iterable[float], initial_values: Iterable[float]) -> np.ndarray:
    eps_arr = _as_1d_float_array(eps, "eps")
    initial_arr = _as_1d_float_array(initial_values, "initial_values")
    x = np.zeros_like(eps_arr)
    init_len = len(initial_arr)
    if init_len < 5:
        raise ValueError("initial_values must contain at least five entries.")
    x[:init_len] = initial_arr
    for t in range(init_len, len(eps_arr)):
        if x[t - 2] >= 0:
            x[t] = 0.5 * x[t - 2] - 0.5 * x[t - 5] + eps_arr[t]
        else:
            x[t] = 0.5 * x[t - 5] + eps_arr[t]
    return x


threshold = treshold


def expo(eps: Iterable[float], initial_values: Iterable[float]) -> np.ndarray:
    eps_arr = _as_1d_float_array(eps, "eps")
    initial_arr = _as_1d_float_array(initial_values, "initial_values")
    x = np.zeros_like(eps_arr)
    init_len = len(initial_arr)
    if init_len < 5:
        raise ValueError("initial_values must contain at least five entries.")
    x[:init_len] = initial_arr
    for t in range(init_len, len(eps_arr)):
        x[t] = (
            (-0.28 - 0.49 * np.exp(-3.89 * x[t - 2] ** 2)) * x[t - 2]
            + (0.41 - 0.54 * np.exp(-3.89 * x[t - 5] ** 2)) * x[t - 5]
            + eps_arr[t]
        )
    return x


def expo2(eps: Iterable[float], initial_values: Iterable[float]) -> np.ndarray:
    eps_arr = _as_1d_float_array(eps, "eps")
    initial_arr = _as_1d_float_array(initial_values, "initial_values")
    x = np.zeros_like(eps_arr)
    init_len = len(initial_arr)
    if init_len < 5:
        raise ValueError("initial_values must contain at least five entries.")
    x[:init_len] = initial_arr
    for t in range(init_len, len(eps_arr)):
        x[t] = (
            (0.5 - 0.5 * np.exp(-50.0 * x[t - 2] ** 2)) * x[t - 2]
            + (0.5 - 2.1 * np.exp(-50.0 * x[t - 5] ** 2)) * x[t - 5]
            + eps_arr[t]
        )
    return x


def models(name: str, n: int, seed: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    if name not in VALID_MODELS:
        raise ValueError(f"Unknown model {name!r}. Expected one of {sorted(VALID_MODELS)}.")
    if n <= 5:
        raise ValueError("n must exceed five.")
    if rng is None:
        rng = np.random.default_rng(seed)
    eps = rng.normal(loc=0.0, scale=0.1, size=n)
    initial = np.zeros(5)
    if name == "linear":
        return linear(eps, 2, [2, 5], [0.5, -0.5], initial)
    if name == "polynomial":
        return polynomial(eps, [2, 5], [1.0, -1.0], [2.0, 2.0], initial)
    if name == "expo":
        return expo(eps, initial)
    if name == "expo2":
        return expo2(eps, initial)
    return treshold(eps, initial)
