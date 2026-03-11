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

yᵢ = α + xᵢβ + vᵢ − uᵢ + wᵢ

where

- yᵢ : dependent variable (investment rate)
- xᵢ : frontier covariates
- vᵢ : symmetric statistical noise
- uᵢ : under-investment inefficiency
- wᵢ : over-investment inefficiency

Distributional assumptions:

vᵢ ~ N(0, σᵥ²)

uᵢ ≥ 0  
wᵢ ≥ 0  

The inefficiency scale parameters depend on firm characteristics:

σᵤᵢ = exp(Cᵤ + zᵢβᵤ)

σwᵢ = exp(Cw + wᵢβw)

---

# Supported Inefficiency Distributions

Four alternative specifications are implemented.

| Model | Distribution |
|------|-------------|
| TTEX_msle | Exponential |
| TTNHN_msle | Half-Normal |
| TTWB_msle | Weibull |
| TTPT_msle | Pareto |

These specifications allow robustness checks for inefficiency distributional assumptions.

---

# Estimation Method

Parameters are estimated using **Maximum Simulated Likelihood Estimation (MSLE)**.

The likelihood is approximated via simulation:

Lᵢ = log( (1/R) Σ fᵥ(εᵢ + uᵢʳ − wᵢʳ) )

Simulation uses:

- **Halton sequences**
- base 2 for u
- base 3 for w

Default number of simulation draws:

R = 2^10 − 1 = 1023

Optimization procedure:

1. **Nelder–Mead** for initial global search
2. **Newton method** for local convergence

Gradients and Hessians are computed using **ForwardDiff.jl**.
