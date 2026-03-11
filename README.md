# Two-Tier Stochastic Frontier Model (MSLE Implementation in Julia)

This repository provides an implementation of a **two-tier stochastic frontier model** estimated using **Maximum Simulated Likelihood Estimation (MSLE)** in Julia.

The model is designed to analyze **bidirectional inefficiency**, allowing firms to exhibit both:

- **Under-investment**
- **Over-investment**

This framework extends the traditional stochastic frontier model by introducing **two inefficiency components**.

The estimation allows for **flexible distributional assumptions** on the inefficiency terms, including:

- Exponential
- Half-Normal
- Weibull
- Pareto

The model is estimated using **quasi-Monte Carlo simulation with Halton sequences**.

---

# Model Specification

The empirical model follows a **two-tier stochastic frontier structure**:

$$y_i = \alpha + x_i\beta + v_i - u_i + w_i$$

where:

- $y_i$ : dependent variable (investment rate)
- $x_i$ : frontier covariates
- $v_i$ : symmetric statistical noise
- $u_i$ : under-investment inefficiency
- $w_i$ : over-investment inefficiency

Distributional assumptions:

$$v_i \sim N(0, \sigma_v^2)$$
$$u_i \ge 0$$
$$w_i \ge 0$$

The inefficiency scale parameters depend on firm characteristics:

$$\sigma_{ui} = \exp(C_u + z_i\beta_u)$$
$$\sigma_{wi} = \exp(C_w + w_i\beta_w)$$

---

# Supported Inefficiency Distributions

Four alternative specifications are implemented:

| Model | Distribution |
|------|-------------|
| `TTEX_msle` | Exponential |
| `TTNHN_msle` | Half-Normal |
| `TTWB_msle` | Weibull |
| `TTPT_msle` | Pareto |

These specifications allow robustness checks for inefficiency distributional assumptions.

---

# Estimation Method

Parameters are estimated using **Maximum Simulated Likelihood Estimation (MSLE)**.

The likelihood is approximated via simulation:

$$L_i = \log \left( \frac{1}{R} \sum_{r=1}^{R} f_v(\epsilon_i + u_i^r - w_i^r) \right)$$

Simulation uses:

- **Halton sequences**
- base 2 for $u$
- base 3 for $w$

Default number of simulation draws:

$$R = 2^{10} - 1 = 1023$$

Optimization procedure:

1. **Nelder–Mead** for initial global search
2. **Newton method** for local convergence

Gradients and Hessians are computed using **ForwardDiff.jl**.

## Dependencies

To run this code, you will need the following Julia packages:
- `Random`, `Distributions`, `StatsFuns`, `SpecialFunctions`
- `Optim`, `ForwardDiff`
- `DataFrames`, `CSV`, `GLM`
- `LinearAlgebra`
- `HaltonSequences`
- `FLoops`, `Base.Threads`
- `PrettyTables`

---

## Usage

### 1. Data Preparation
The script expects a CSV file containing your empirical data (e.g., `4Sector_keiretsu_work_data_5.csv`). The matrices are populated as follows:
- **Dependent Variable (y)**: `investment_rate`
- **Frontier Covariates (x)**: `ln_Tobins_q`, `ln_sales_rate`, `trend`, `D_l`, `D_m`, `D_s`, `D_xs`
- **Inefficiency Covariates (z and w)**: `K`, `ln_asset`

### 2. Initialization
Initial values for the frontier variables (α and βx) are obtained by first estimating a baseline Ordinary Least Squares (OLS) model. The inefficiency terms, scale parameters, and variance terms are initialized with a default value of 0.1.

### 3. Execution
The script sequentially defines and estimates the models. For example, the Exponential model function (`TTEX_msle`) is wrapped in a `TwiceDifferentiable` objective function for automatic differentiation:

## Robust Optimization & Hessian Adjustments

Due to the complexity of Maximum Simulated Likelihood Estimation, the script implements robust numerical mechanisms to handle non-positive definite Hessian matrices and successfully calculate standard errors:

- **Two-Step Optimization**: The script first utilizes `NelderMead()` for an initial global search, then feeds those results as the starting values into `Newton()` for precise local optimization.
- **Hessian Eigenvalue Correction**: If the estimated Hessian matrix is not positive definite, a custom function (`make_positive_definite`) automatically corrects the negative eigenvalues to ensure invertibility.
- **Robust Variance-Covariance Matrix Calculation**: The variance-covariance matrix is computed using the inverse of the Hessian. If direct inversion fails, the script uses a cascading fallback mechanism:
    1. Cholesky decomposition
    2. Tikhonov regularization ($H + \lambda I$)
    3. Pseudo-inverse calculation (`pinv`)

---

## Output

For each implemented distribution model, the script outputs:
1. **Console Summary**: Displays the optimization results for both Nelder-Mead and Newton algorithms, Hessian positive-definiteness checks, and a cleanly formatted table (via `PrettyTables`) displaying the Coefficients, Mean, Standard Errors, and T-Statistics.
2. **CSV Export**: The full estimation table is saved to your local directory (e.g., `estimation_TTEX.csv`, `estimation_TTNHN.csv`, `estimation_TTWB.csv`, `estimation_TTPT.csv`) for easy integration into research papers or further analysis.
