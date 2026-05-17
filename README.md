# KernelLagSelect

KernelLagSelect implements nonparametric autoregressive model analysis for time-series processes. It provides tools for generating nonlinear autoregressive processes, estimating kernel-based conditional means, computing projections of conditional means, evaluating AFPE-style lag-selection criteria, and visualizing conditional mean and projection behavior across candidate lag variables.

## Project Structure

```text
KernelLagSelect/
├── main.py
├── requirements.txt
├── README.md
└── afpe/
    ├── \_\_init\_\_.py
    ├── afpe.py
    ├── estimators.py
    ├── kernels.py
    ├── models.py
    └── utilities.py
```

## Files and Functionalities

### `main.py`

Executable script that runs the project examples and saves generated figures.

Main functionalities:

* Generates synthetic autoregressive time series.
* Computes one-dimensional conditional mean estimates for candidate lags.
* Computes projection functions for multidimensional lag sets.
* Demonstrates projection behavior under increasing dimensionality.
* Saves plots to the `outputs/` directory.
* Supports a quick-run mode for fast testing.

Example usage:

```bash
python main.py
```

Fast test run:

```bash
python main.py --quick
```

### `requirements.txt`

Lists the Python dependencies required to run the project.

Main dependencies:

* `numpy`
* `scipy`
* `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

### `afpe/\_\_init\_\_.py`

Package initialization file. It exposes the project modules as a Python package.

### `afpe/kernels.py`

Defines kernel functions used in nonparametric estimation.

Main functionalities:

* Gaussian kernel.
* Rectangular kernel.
* Triangular kernel.
* Parabolic kernel.
* Biweight kernel.
* Product-kernel evaluation for multivariate lag vectors.
* Kernel constants such as the squared L2 norm.

### `afpe/models.py`

Defines synthetic autoregressive data-generating processes.

Main functionalities:

* Linear autoregressive process.
* Polynomial autoregressive process.
* Exponential autoregressive process.
* Alternative exponential process.
* Threshold autoregressive process.
* General model wrapper for selecting and generating a chosen process.

### `afpe/estimators.py`

Contains the main nonparametric estimators.

Main functionalities:

* Joint density estimation for lag vectors.
* Kernel numerator estimation for conditional means.
* Nadaraya-Watson conditional mean estimation.
* Conditional second-moment estimation.
* Conditional variance estimation.
* Projection of conditional mean functions onto selected lag coordinates.
* Projection variance estimation.

### `afpe/afpe.py`

Implements AFPE-style model-selection functionality.

Main functionalities:

* Empirical prediction error estimation.
* Density-weighted residual error estimation.
* Bandwidth optimization for AFPE calculations.
* AFPE criterion evaluation for a candidate lag set.
* Enumeration of candidate lag subsets.
* Selection of the best lag subset according to the AFPE criterion.

### `afpe/utilities.py`

Contains recurring helper functions used across the project.

Main functionalities:

* Construction of lagged design matrices.
* Extraction of lagged vectors for a given time index.
* Safe numerical division.
* Grid construction for plotting conditional means and projections.
* Plotting utilities for estimated curves and confidence bands.
* Output-directory handling.

## Generated Outputs

Running `main.py` creates an `outputs/` directory containing PNG figures.

Typical outputs include:

```text
outputs/
├── linear\_conditional\_mean\_lag\_1.png
├── linear\_conditional\_mean\_lag\_2.png
├── linear\_conditional\_mean\_lag\_4.png
├── linear\_conditional\_mean\_lag\_5.png
├── polynomial\_projection\_coord\_1.png
├── polynomial\_projection\_coord\_2.png
├── polynomial\_projection\_coord\_3.png
├── polynomial\_projection\_coord\_4.png
├── projection\_deterioration\_coord\_1.png
└── projection\_deterioration\_coord\_2.png
```

## Example Workflows

### Conditional Mean Estimation

The script can estimate and plot conditional means of the form:

```text
E\[X\_t | X\_{t-lag}]
```

for selected candidate lags.

### Projection Estimation

The script can estimate one-dimensional projections of a multivariate conditional mean function. This is useful for visualizing how individual lag coordinates contribute to the estimated autoregressive structure.

### AFPE-Based Lag Selection

The AFPE module can evaluate candidate lag sets and select the lag subset with the smallest criterion value.

