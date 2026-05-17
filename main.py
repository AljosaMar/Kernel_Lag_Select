from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from afpe.estimators import Con_Mean
from afpe.models import linear, polynomial
from afpe.utilities import Projection, Sigma_M_est, Sigma_P_est, mean_and_bands


def _plot_replicates(
    grid: np.ndarray,
    curves: list[np.ndarray],
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    title: str,
    ylabel: str,
    output_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for curve in curves:
        ax.plot(grid, curve, linewidth=0.9)
    if lower is not None and upper is not None:
        ax.plot(grid, lower, linewidth=1.2)
        ax.plot(grid, upper, linewidth=1.2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def conditional_means_linear(output_dir: Path, n: int, reps: int, seed: int) -> None:
    grid = np.arange(-0.6, 0.6001, 0.05)
    h = 0.05
    rng = np.random.default_rng(seed)
    lags_to_plot = [1, 2, 4, 5]
    results = {lag: [] for lag in lags_to_plot}
    sigmas = {lag: [] for lag in lags_to_plot}
    for _ in range(reps):
        eps = rng.normal(0.0, 0.1, n)
        x = linear(eps, 2, [2, 5], [0.5, -0.5], [0, 0, 0, 0, 0])
        for lag in lags_to_plot:
            m_hat = Con_Mean(x, [lag], "gaussian", h)
            sigma_hat = Sigma_M_est(x, [lag], "gaussian", h)
            points = grid.reshape(-1, 1)
            results[lag].append(m_hat.evaluate_many(points))
            sigmas[lag].append(sigma_hat.evaluate_many(points))
    for lag in lags_to_plot:
        _, lower, upper = mean_and_bands(results[lag], sigmas[lag])
        title = f"Conditional means for the linear model, sample size {n}" if lag == 1 else ""
        _plot_replicates(
            grid,
            results[lag],
            lower,
            upper,
            title=title,
            ylabel=f"M lag {lag}",
            output_path=output_dir / f"linear_conditional_mean_lag_{lag}.png",
            xlim=(-0.4, 0.4),
            ylim=(-0.8, 0.8),
        )


def projections_polynomial(output_dir: Path, n: int, reps: int, seed: int) -> None:
    grid = np.arange(-0.6, 0.6001, 0.05)
    h = 0.05
    rng = np.random.default_rng(seed)
    lag_vector = [1, 2, 4, 5]
    masks = list(range(1, len(lag_vector) + 1))
    results = {mask: [] for mask in masks}
    sigmas = {mask: [] for mask in masks}
    for _ in range(reps):
        eps = rng.normal(0.0, 0.1, n)
        x = polynomial(eps, [2, 5], [1, -1], [2, 2], [0, 0, 0, 0, 0])
        for mask in masks:
            p_hat = Projection(x, lag_vector, [mask], "gaussian", h)
            sigma_hat = Sigma_P_est(x, lag_vector, [mask], "gaussian", h)
            results[mask].append(p_hat.evaluate_many(grid))
            sigmas[mask].append(sigma_hat.evaluate_many(grid))
    for mask in masks:
        lag = lag_vector[mask - 1]
        _, lower, upper = mean_and_bands(results[mask], sigmas[mask])
        title = f"Projections for the polynomial model, sample size {n}" if mask == 1 else ""
        _plot_replicates(
            grid,
            results[mask],
            lower,
            upper,
            title=title,
            ylabel=f"P lag {lag}",
            output_path=output_dir / f"polynomial_projection_lag_{lag}.png",
            xlim=(-0.5, 0.5),
            ylim=(-0.3, 0.3),
        )


def projection_deterioration(output_dir: Path, n: int, reps: int, seed: int) -> None:
    grid = np.arange(-0.5, 0.5001, 0.05)
    h = 0.05
    rng = np.random.default_rng(seed)
    lag_vector = [2, 5]
    masks = list(range(1, len(lag_vector) + 1))
    results = {mask: [] for mask in masks}
    for _ in range(reps):
        eps = rng.normal(0.0, 0.1, n)
        x = polynomial(eps, [2, 5], [1, -1], [2, 2], [0, 0, 0, 0, 0])
        for mask in masks:
            p_hat = Projection(x, lag_vector, [mask], "gaussian", h)
            results[mask].append(p_hat.evaluate_many(grid))
    for mask in masks:
        lag = lag_vector[mask - 1]
        _plot_replicates(
            grid,
            results[mask],
            lower=None,
            upper=None,
            title="Projection deterioration example" if mask == 1 else "",
            ylabel=f"P lag {lag}",
            output_path=output_dir / f"projection_deterioration_lag_{lag}.png",
            xlim=(-0.5, 0.5),
            ylim=(-0.3, 0.3),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AFPE examples.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for PNG plots.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for NumPy's random generator.")
    parser.add_argument("--quick", action="store_true", help="Use smaller N/repetition counts for a smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        conditional_means_linear(args.output_dir, n=300, reps=2, seed=args.seed)
        projections_polynomial(args.output_dir, n=180, reps=1, seed=args.seed + 1)
        projection_deterioration(args.output_dir, n=180, reps=2, seed=args.seed + 2)
    else:
        conditional_means_linear(args.output_dir, n=2000, reps=10, seed=args.seed)
        projections_polynomial(args.output_dir, n=1000, reps=10, seed=args.seed + 1)
        projection_deterioration(args.output_dir, n=400, reps=10, seed=args.seed + 2)
    print(f"Saved plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
