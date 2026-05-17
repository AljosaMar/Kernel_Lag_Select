from __future__ import annotations

from typing import Callable

import numpy as np


KernelName = str
VALID_KERNELS = {"gaussian", "rectangular", "triangular", "parabolic", "biweight"}


def indicator(x: np.ndarray | float) -> np.ndarray | float:
    return np.where(np.abs(x) <= 1.0, 1.0, 0.0)


def gaussian_kernel(x: np.ndarray | float) -> np.ndarray | float:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-(np.asarray(x) ** 2) / 2.0)


def rectangular_kernel(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * indicator(x)


def triangular_kernel(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x)
    return (1.0 - np.abs(x)) * indicator(x)


def parabolic_kernel(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x)
    return 0.75 * (1.0 - x**2) * indicator(x)


def biweight_kernel(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x)
    return (15.0 / 16.0) * (1.0 - x**2) ** 2 * indicator(x)


def base_kernel(kind: KernelName = "gaussian") -> Callable[[np.ndarray | float], np.ndarray | float]:
    if kind not in VALID_KERNELS:
        raise ValueError(f"Unknown kernel {kind!r}. Expected one of {sorted(VALID_KERNELS)}.")
    return {
        "gaussian": gaussian_kernel,
        "rectangular": rectangular_kernel,
        "triangular": triangular_kernel,
        "parabolic": parabolic_kernel,
        "biweight": biweight_kernel,
    }[kind]


def base_kernel_at_zero(kind: KernelName = "gaussian") -> float:
    return float(base_kernel(kind)(0.0))


def product_kernel_values(x: np.ndarray, kind: KernelName = "gaussian", h: float = 0.1) -> np.ndarray | float:
    if h <= 0:
        raise ValueError("Bandwidth h must be positive.")
    x_arr = np.asarray(x, dtype=float)
    scalar_input = x_arr.ndim == 0
    if scalar_input:
        x_arr = x_arr.reshape(1)
    vals = base_kernel(kind)(x_arr / h) / h
    if vals.ndim == 0:
        return float(vals)
    if vals.ndim == 1:
        return float(np.prod(vals))
    return np.prod(vals, axis=-1)


def product_kernel(kind: KernelName = "gaussian", dim: int = 1) -> Callable[[np.ndarray, float], float]:
    if dim <= 0:
        raise ValueError("dim must be a positive integer.")
    if kind not in VALID_KERNELS:
        raise ValueError(f"Unknown kernel {kind!r}. Expected one of {sorted(VALID_KERNELS)}.")

    def evaluate(x: np.ndarray | list[float] | float, h: float = 0.1) -> float:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1)
        if x_arr.shape[-1] != dim:
            raise ValueError(f"Expected input with dimension {dim}, got shape {x_arr.shape}.")
        return float(product_kernel_values(x_arr, kind=kind, h=h))

    return evaluate


def kernel_l2_squared(kind: KernelName = "gaussian", dim: int = 1) -> float:
    if dim <= 0:
        raise ValueError("dim must be a positive integer.")
    if kind not in VALID_KERNELS:
        raise ValueError(f"Unknown kernel {kind!r}. Expected one of {sorted(VALID_KERNELS)}.")
    univariate = {
        "gaussian": 1.0 / (2.0 * np.sqrt(np.pi)),
        "rectangular": 0.5,
        "triangular": 2.0 / 3.0,
        "parabolic": 3.0 / 5.0,
        "biweight": 5.0 / 7.0,
    }[kind]
    return float(univariate**dim)
