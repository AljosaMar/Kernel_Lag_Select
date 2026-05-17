from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .estimators import Con_Mean, Con_Var, jdens, lagged_matrix, validate_lags
from .kernels import KernelName, kernel_l2_squared


def _nanmean(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    count = np.sum(finite, axis=axis)
    total = np.sum(np.where(finite, arr, 0.0), axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


@dataclass
class ProjectionEstimator:
    series: np.ndarray
    lags: np.ndarray
    x_mask: np.ndarray
    kernel: KernelName = "gaussian"
    h: float = 0.1
    moment: str = "mean"
    mask_is_one_based: bool = True

    def __post_init__(self) -> None:
        self.series = np.asarray(self.series, dtype=float)
        self.lags = validate_lags(self.lags)
        mask = np.asarray(list(self.x_mask), dtype=int)
        if self.mask_is_one_based:
            mask = mask - 1
        if mask.ndim != 1 or len(mask) == 0:
            raise ValueError("x_mask must be a non-empty one-dimensional sequence.")
        if len(np.unique(mask)) != len(mask):
            raise ValueError("x_mask must contain unique coordinates.")
        if np.any(mask < 0) or np.any(mask >= len(self.lags)):
            raise ValueError("x_mask contains coordinates outside the lag vector dimension.")
        self.x_mask = mask
        max_lag = int(np.max(self.lags))
        self.empirical_design = lagged_matrix(self.series, self.lags, start=max_lag, stop=len(self.series))
        if self.moment == "mean":
            self.conditional_moment = Con_Mean(self.series, self.lags, self.kernel, self.h)
        elif self.moment == "variance":
            self.conditional_moment = Con_Var(self.series, self.lags, self.kernel, self.h)
        else:
            raise ValueError("moment must be 'mean' or 'variance'.")

    def evaluate_many(self, x_values: Iterable[float] | np.ndarray, chunk_size: int = 512) -> np.ndarray:
        x_arr = np.asarray(x_values, dtype=float)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1, 1)
        elif len(self.x_mask) == 1 and x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        elif x_arr.ndim == 1 and x_arr.size == len(self.x_mask):
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2 or x_arr.shape[1] != len(self.x_mask):
            raise ValueError(f"Expected x values with {len(self.x_mask)} column(s).")
        out = np.empty(x_arr.shape[0], dtype=float)
        for i, x in enumerate(x_arr):
            points = self.empirical_design.copy()
            points[:, self.x_mask] = x
            out[i] = float(_nanmean(self.conditional_moment.evaluate_many(points, chunk_size=chunk_size)))
        return out

    def __call__(self, x: Iterable[float] | float) -> float:
        return float(self.evaluate_many(np.asarray(x, dtype=float))[0])


@dataclass
class SigmaMEstimator:
    series: np.ndarray
    lags: np.ndarray
    kernel: KernelName = "gaussian"
    h: float = 0.1
    error_variance: float = 0.01

    def __post_init__(self) -> None:
        self.series = np.asarray(self.series, dtype=float)
        self.lags = validate_lags(self.lags)
        self.density = jdens(self.series, self.lags, self.kernel, self.h)
        self.l2 = kernel_l2_squared(self.kernel, len(self.lags))
        self.n_eff = self.density.n_eff

    def evaluate_many(self, points: np.ndarray, chunk_size: int = 512) -> np.ndarray:
        p = self.density.evaluate_many(points, chunk_size=chunk_size)
        denominator = self.n_eff * (self.h ** len(self.lags)) * p
        variance = self.error_variance * self.l2 / np.where(np.abs(denominator) < 1e-14, np.nan, denominator)
        return np.sqrt(np.maximum(variance, 0.0))

    def __call__(self, x: Iterable[float] | float) -> float:
        return float(self.evaluate_many(np.asarray(x, dtype=float))[0])


@dataclass
class SigmaPEstimator:
    series: np.ndarray
    lags: np.ndarray
    x_mask: np.ndarray
    kernel: KernelName = "gaussian"
    h: float = 0.1
    error_variance: float = 0.01
    mask_is_one_based: bool = True

    def __post_init__(self) -> None:
        self.series = np.asarray(self.series, dtype=float)
        self.lags = validate_lags(self.lags)
        mask = np.asarray(list(self.x_mask), dtype=int)
        if self.mask_is_one_based:
            mask = mask - 1
        if mask.ndim != 1 or len(mask) == 0:
            raise ValueError("x_mask must be a non-empty one-dimensional sequence.")
        if len(np.unique(mask)) != len(mask):
            raise ValueError("x_mask must contain unique coordinates.")
        if np.any(mask < 0) or np.any(mask >= len(self.lags)):
            raise ValueError("x_mask contains coordinates outside the lag vector dimension.")
        self.x_mask = mask
        max_lag = int(np.max(self.lags))
        self.empirical_design = lagged_matrix(self.series, self.lags, start=max_lag, stop=len(self.series))
        self.sigma_m = SigmaMEstimator(self.series, self.lags, self.kernel, self.h, self.error_variance)

    def evaluate_many(self, x_values: Iterable[float] | np.ndarray, chunk_size: int = 512) -> np.ndarray:
        x_arr = np.asarray(x_values, dtype=float)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1, 1)
        elif len(self.x_mask) == 1 and x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        elif x_arr.ndim == 1 and x_arr.size == len(self.x_mask):
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2 or x_arr.shape[1] != len(self.x_mask):
            raise ValueError(f"Expected x values with {len(self.x_mask)} column(s).")
        out = np.empty(x_arr.shape[0], dtype=float)
        for i, x in enumerate(x_arr):
            points = self.empirical_design.copy()
            points[:, self.x_mask] = x
            se = self.sigma_m.evaluate_many(points, chunk_size=chunk_size)
            out[i] = float(np.sqrt(_nanmean(se**2)))
        return out

    def __call__(self, x: Iterable[float] | float) -> float:
        return float(self.evaluate_many(np.asarray(x, dtype=float))[0])


def Projection(
    series: Iterable[float],
    lags: Iterable[int],
    xMask: Iterable[int],
    kern: KernelName = "gaussian",
    h: float = 0.1,
) -> ProjectionEstimator:
    return ProjectionEstimator(np.asarray(series, dtype=float), validate_lags(lags), np.asarray(list(xMask), dtype=int), kern, h, "mean")


def Var_Proj(
    series: Iterable[float],
    lags: Iterable[int],
    xMask: Iterable[int],
    kern: KernelName = "gaussian",
    h: float = 0.1,
) -> ProjectionEstimator:
    return ProjectionEstimator(np.asarray(series, dtype=float), validate_lags(lags), np.asarray(list(xMask), dtype=int), kern, h, "variance")


def Sigma_M_est(
    series: Iterable[float],
    lags: Iterable[int],
    kern: KernelName = "gaussian",
    h: float = 0.1,
    error_variance: float = 0.01,
) -> SigmaMEstimator:
    return SigmaMEstimator(np.asarray(series, dtype=float), validate_lags(lags), kern, h, error_variance)


def Sigma_P_est(
    series: Iterable[float],
    lags: Iterable[int],
    xMask: Iterable[int],
    kern: KernelName = "gaussian",
    h: float = 0.1,
    error_variance: float = 0.01,
) -> SigmaPEstimator:
    return SigmaPEstimator(np.asarray(series, dtype=float), validate_lags(lags), np.asarray(list(xMask), dtype=int), kern, h, error_variance)


def mean_and_bands(replicates: list[np.ndarray], sigmas: list[np.ndarray] | None = None, multiplier: float = 1.96):
    curves = np.asarray(replicates, dtype=float)
    mean_curve = _nanmean(curves, axis=0)
    if sigmas is None:
        return mean_curve, None, None
    sigma_curve = _nanmean(np.asarray(sigmas, dtype=float), axis=0)
    return mean_curve, mean_curve - multiplier * sigma_curve, mean_curve + multiplier * sigma_curve
