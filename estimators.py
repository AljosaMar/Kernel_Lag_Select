from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .kernels import KernelName, product_kernel_values


_DENOMINATOR_EPS = 1e-14


def validate_lags(lags: Iterable[int]) -> np.ndarray:
    lag_arr = np.asarray(list(lags), dtype=int)
    if lag_arr.ndim != 1 or len(lag_arr) == 0:
        raise ValueError("lags must be a non-empty one-dimensional sequence.")
    if np.any(lag_arr <= 0):
        raise ValueError("lags must be positive integers.")
    if len(np.unique(lag_arr)) != len(lag_arr):
        raise ValueError("lags must be unique.")
    return np.sort(lag_arr)


def lagged_matrix(
    series: Iterable[float],
    lags: Iterable[int],
    start: int | None = None,
    stop: int | None = None,
) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    lag_arr = validate_lags(lags)
    max_lag = int(np.max(lag_arr))
    if start is None:
        start = max_lag
    if stop is None:
        stop = len(x)
    if start < max_lag:
        raise ValueError("start must be at least max(lags).")
    if stop > len(x):
        raise ValueError("stop cannot exceed the series length.")
    if stop <= start:
        raise ValueError("stop must be greater than start.")
    t = np.arange(start, stop, dtype=int)
    return x[t[:, None] - lag_arr[None, :]]


def response_vector(series: Iterable[float], lags: Iterable[int]) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    lag_arr = validate_lags(lags)
    max_lag = int(np.max(lag_arr))
    if len(x) <= max_lag:
        raise ValueError("series length must exceed max(lags).")
    return x[max_lag:]


def _as_points(points: Iterable[float] | float | np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 0:
        if dim != 1:
            raise ValueError(f"Expected points with dimension {dim}.")
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if dim == 1:
            return arr.reshape(-1, 1)
        if arr.size == dim:
            return arr.reshape(1, dim)
        raise ValueError(f"Expected points with dimension {dim}, got vector of length {arr.size}.")
    if arr.ndim == 2 and arr.shape[1] == dim:
        return arr
    raise ValueError(f"Expected points with dimension {dim}.")


def _evaluate_kernel_sums(
    points: np.ndarray,
    design: np.ndarray,
    kernel: KernelName,
    h: float,
    weights: np.ndarray | None = None,
    chunk_size: int = 512,
) -> np.ndarray:
    points_arr = _as_points(points, design.shape[1])
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape[0] != design.shape[0]:
            raise ValueError("weights length must equal the number of design rows.")
    else:
        weights_arr = None
    out = np.empty(points_arr.shape[0], dtype=float)
    for start in range(0, len(points_arr), chunk_size):
        end = min(start + chunk_size, len(points_arr))
        diff = points_arr[start:end, None, :] - design[None, :, :]
        kvals = product_kernel_values(diff, kind=kernel, h=h)
        if weights_arr is None:
            out[start:end] = np.mean(kvals, axis=1)
        else:
            out[start:end] = np.mean(kvals * weights_arr[None, :], axis=1)
    return out


def leave_one_out_density_at_design(
    design: np.ndarray,
    kernel: KernelName,
    h: float,
    chunk_size: int = 512,
) -> np.ndarray:
    design_arr = np.asarray(design, dtype=float)
    n = design_arr.shape[0]
    if n <= 1:
        raise ValueError("At least two design rows are required for leave-one-out estimates.")
    self_kernel = float(product_kernel_values(np.zeros(design_arr.shape[1]), kind=kernel, h=h))
    out = np.empty(n, dtype=float)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        diff = design_arr[start:end, None, :] - design_arr[None, :, :]
        kvals = product_kernel_values(diff, kind=kernel, h=h)
        out[start:end] = (np.sum(kvals, axis=1) - self_kernel) / (n - 1)
    return out


def leave_one_out_conditional_mean_at_design(
    design: np.ndarray,
    targets: np.ndarray,
    kernel: KernelName,
    h: float,
    chunk_size: int = 512,
) -> np.ndarray:
    design_arr = np.asarray(design, dtype=float)
    y = np.asarray(targets, dtype=float)
    n = design_arr.shape[0]
    if y.shape[0] != n:
        raise ValueError("targets length must equal the number of design rows.")
    if n <= 1:
        raise ValueError("At least two design rows are required for leave-one-out estimates.")
    self_kernel = float(product_kernel_values(np.zeros(design_arr.shape[1]), kind=kernel, h=h))
    out = np.empty(n, dtype=float)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        diff = design_arr[start:end, None, :] - design_arr[None, :, :]
        kvals = product_kernel_values(diff, kind=kernel, h=h)
        numerator = np.sum(kvals * y[None, :], axis=1) - self_kernel * y[start:end]
        denominator = np.sum(kvals, axis=1) - self_kernel
        out[start:end] = numerator / np.where(np.abs(denominator) < _DENOMINATOR_EPS, np.nan, denominator)
    return out


@dataclass
class JointDensityEstimator:
    series: np.ndarray
    lags: np.ndarray
    kernel: KernelName = "gaussian"
    h: float = 0.1

    def __post_init__(self) -> None:
        self.series = np.asarray(self.series, dtype=float)
        self.lags = validate_lags(self.lags)
        max_lag = int(np.max(self.lags))
        if len(self.series) <= max_lag:
            raise ValueError("series length must exceed max(lags).")
        self.design = lagged_matrix(self.series, self.lags, start=max_lag, stop=len(self.series))
        self.n_eff = self.design.shape[0]

    def evaluate_many(self, points: np.ndarray, chunk_size: int = 512) -> np.ndarray:
        return _evaluate_kernel_sums(points, self.design, self.kernel, self.h, chunk_size=chunk_size)

    def __call__(self, x: Iterable[float] | float) -> float:
        return float(self.evaluate_many(np.asarray(x, dtype=float))[0])


@dataclass
class KernelRegressionNumerator:
    series: np.ndarray
    lags: np.ndarray
    kernel: KernelName = "gaussian"
    h: float = 0.1
    power: int = 1

    def __post_init__(self) -> None:
        self.series = np.asarray(self.series, dtype=float)
        self.lags = validate_lags(self.lags)
        max_lag = int(np.max(self.lags))
        if len(self.series) <= max_lag:
            raise ValueError("series length must exceed max(lags).")
        self.design = lagged_matrix(self.series, self.lags, start=max_lag, stop=len(self.series))
        self.targets = self.series[max_lag:] ** self.power
        self.n_eff = self.design.shape[0]

    def evaluate_many(self, points: np.ndarray, chunk_size: int = 512) -> np.ndarray:
        return _evaluate_kernel_sums(
            points,
            self.design,
            self.kernel,
            self.h,
            weights=self.targets,
            chunk_size=chunk_size,
        )

    def __call__(self, x: Iterable[float] | float) -> float:
        return float(self.evaluate_many(np.asarray(x, dtype=float))[0])


@dataclass
class ConditionalMomentEstimator:
    series: np.ndarray
    lags: np.ndarray
    kernel: KernelName = "gaussian"
    h: float = 0.1
    power: int = 1

    def __post_init__(self) -> None:
        self.series = np.asarray(self.series, dtype=float)
        self.lags = validate_lags(self.lags)
        self.density = JointDensityEstimator(self.series, self.lags, self.kernel, self.h)
        self.numerator = KernelRegressionNumerator(self.series, self.lags, self.kernel, self.h, self.power)
        self.n_eff = self.density.n_eff

    def evaluate_many(self, points: np.ndarray, chunk_size: int = 512) -> np.ndarray:
        numerator = self.numerator.evaluate_many(points, chunk_size=chunk_size)
        denominator = self.density.evaluate_many(points, chunk_size=chunk_size)
        return numerator / np.where(np.abs(denominator) < _DENOMINATOR_EPS, np.nan, denominator)

    def __call__(self, x: Iterable[float] | float) -> float:
        return float(self.evaluate_many(np.asarray(x, dtype=float))[0])


@dataclass
class ConditionalVarianceEstimator:
    series: np.ndarray
    lags: np.ndarray
    kernel: KernelName = "gaussian"
    h: float = 0.1

    def __post_init__(self) -> None:
        self.series = np.asarray(self.series, dtype=float)
        self.lags = validate_lags(self.lags)
        self.mean_estimator = ConditionalMomentEstimator(self.series, self.lags, self.kernel, self.h, 1)
        self.second_moment_estimator = ConditionalMomentEstimator(self.series, self.lags, self.kernel, self.h, 2)
        self.n_eff = self.mean_estimator.n_eff

    def evaluate_many(self, points: np.ndarray, chunk_size: int = 512) -> np.ndarray:
        mean = self.mean_estimator.evaluate_many(points, chunk_size=chunk_size)
        second = self.second_moment_estimator.evaluate_many(points, chunk_size=chunk_size)
        return np.maximum(second - mean**2, 0.0)

    def __call__(self, x: Iterable[float] | float) -> float:
        return float(self.evaluate_many(np.asarray(x, dtype=float))[0])


def jdens(series: Iterable[float], lags: Iterable[int], kern: KernelName = "gaussian", h: float = 0.1) -> JointDensityEstimator:
    return JointDensityEstimator(np.asarray(series, dtype=float), validate_lags(lags), kern, h)


def estr(series: Iterable[float], lags: Iterable[int], kern: KernelName = "gaussian", h: float = 0.1) -> KernelRegressionNumerator:
    return KernelRegressionNumerator(np.asarray(series, dtype=float), validate_lags(lags), kern, h, power=1)


def estR(series: Iterable[float], lags: Iterable[int], kern: KernelName = "gaussian", h: float = 0.1) -> KernelRegressionNumerator:
    return KernelRegressionNumerator(np.asarray(series, dtype=float), validate_lags(lags), kern, h, power=2)


def Con_Mean(series: Iterable[float], lags: Iterable[int], kern: KernelName = "gaussian", h: float = 0.1) -> ConditionalMomentEstimator:
    return ConditionalMomentEstimator(np.asarray(series, dtype=float), validate_lags(lags), kern, h, power=1)


def Con_Second_Moment(series: Iterable[float], lags: Iterable[int], kern: KernelName = "gaussian", h: float = 0.1) -> ConditionalMomentEstimator:
    return ConditionalMomentEstimator(np.asarray(series, dtype=float), validate_lags(lags), kern, h, power=2)


def Con_Var(series: Iterable[float], lags: Iterable[int], kern: KernelName = "gaussian", h: float = 0.1) -> ConditionalVarianceEstimator:
    return ConditionalVarianceEstimator(np.asarray(series, dtype=float), validate_lags(lags), kern, h)
