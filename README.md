# 🧮 Convex Optimisation for Business Acquisition Valuation  
### *Censored Regression  |  Covariance Estimation  |  Portfolio Optimisation*

---

## 📘 Project Overview
This project demonstrates an **end-to-end convex-optimisation pipeline** applied to *Mergers & Acquisitions (M&A)* data.  
Incomplete acquisition prices are reconstructed, risk structures estimated, and optimal portfolio allocations derived — all through mathematically guaranteed convex programs.

| Phase | Convex Objective | Practical Purpose |
|:--|:--|:--|
| **1. Censored Regression** | Estimate missing deal values under right-censoring constraints | Rebuild undisclosed M&A prices |
| **2. Covariance Estimation** | Compute mean (μ) and covariance (Σ) of predicted profits | Quantify expected return & risk |
| **3. Portfolio Optimisation** | Solve the Markowitz mean-variance QP | Allocate capital efficiently |

All optimisation steps are solved with **[CVXPY](https://www.cvxpy.org)** and adhere to the *Disciplined Convex Programming (DCP)* ruleset.

---

## 🧩 Phase 1 — Censored Regression

### Mathematical Formulation
For features $x_i$ and observed deal values $y_i$:

$$
\min_{c}\;
\sum_{i\in\mathcal{U}} (y_i - c^\top x_i)^2
+\lambda \|c\|_2^2
\quad
\text{s.t. } c^\top x_j \ge D,\; j\in\mathcal{C}.
$$

- $\mathcal{U}$: set of uncensored (disclosed) deals  
- $\mathcal{C}$: set of censored (undisclosed) deals  
- $D$: censoring threshold (e.g. 75th percentile of disclosed values)

This is a **convex quadratic program (QP)**:  
quadratic objective + linear constraints.

### Implementation Highlights
```python
c = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(X_unc @ c - y_sc) + λ*cp.sum_squares(c))
constraints = [X_cen @ c >= D_scaled]
prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.OSQP)
```

### Results
| Metric | Observation |
|:--|:--|
| Status | `optimal` |
| Features retained | 201 / 488 (low-variance columns dropped) |
| Censored rows | 1 218 (≈ 8 % minor tolerance violations) |
| Output | Full vector of predicted acquisition prices |

**Outcome:** A complete and numerically stable dataset of reconstructed deal prices.

---

## 📈 Phase 2 — Covariance Estimation

### Objective
Convert predicted deal prices into firm-level monthly profit flows, then compute  
expected returns $\mu$ and covariances $\Sigma$:

$$
\mu_j = \frac{1}{T}\sum_t p_{t,j}, \qquad
\Sigma_{jk} = \frac{1}{T-1}\sum_t (p_{t,j}-\mu_j)(p_{t,k}-\mu_k).
$$

### Code Summary
```python
# Build monthly panel (207 months × 27 firms)
panel = (
    df.groupby(['month','Acquiring Company'])['predicted_price']
      .sum().unstack().fillna(0.0)
)
panel_centered = panel - panel.mean()
mu = panel.mean(axis=0).to_numpy()
Sigma_emp = np.cov(panel_centered.to_numpy(), rowvar=False)
```

### Convex Log-Det Regularisation (Graphical Lasso)
To stabilise $\Sigma$:

$$
\min_{S \succ 0} 
\; \big(
 -\log \det S 
 + \operatorname{tr}(S \Sigma_{\text{emp}}) 
 + \lambda \| S \|_{1,\text{off}}
 \big)
$$

```python
S = cp.Variable((n,n), symmetric=True)
offdiag = cp.norm1(S - cp.multiply(np.eye(n), S))
obj = cp.Minimize(-cp.log_det(S) + cp.trace(S @ Sigma_emp) + λ*offdiag)
prob = cp.Problem(obj, [S >> 1e-6*np.eye(n)])
prob.solve(solver=cp.SCS, use_indirect=True)
Sigma_glasso = np.linalg.pinv(S.value + 1e-8*np.eye(n))
```

### Output
```
Entities (n): 27
Panel shape (T×n): (207, 27)
μ shape: (27,) | Σ shape: (27,27)
Status: optimal_inaccurate
```
✅ **27 companies**, **207 months**, convex solver converged.  
Resulting $\Sigma$ is positive-definite and well-conditioned for portfolio optimisation.

---

## 💼 Phase 3 — Portfolio Optimisation

### Markowitz Mean–Variance Formulation
$$
\min_w -\mu^\top w + \gamma\,w^\top\Sigma w
\quad
\text{s.t. } 1^\top w = 1,\; w \ge 0.
$$

and  

$$
\min_w w^\top\Sigma w
\quad
\text{s.t. } \mu^\top w \ge r_{\min},\; 1^\top w=1,\; w\ge0.
$$

### Implementation
```python
def solve_markowitz_scalarized(mu, Sigma, gamma):
    w = cp.Variable(n)
    obj = -mu @ w + gamma * cp.quad_form(w, Sigma)
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(w)==1, w>=0])
    prob.solve(solver=cp.OSQP)
    return w.value
```
25 values of $\gamma \in [10^{-3}, 10^{2}]$ produce the **efficient frontier** of return vs risk.

### Results
#### Efficient Frontier
| γ | Expected Return | Risk (Std Dev) |
|:--|--:|--:|
| 0.001 | 388.21 | 23.99 |
| ⋮ | … | … |

#### Return-Seeking Portfolio ($\gamma = 0.001$)
| Entity | Weight |
|:--|--:|
| Google | 1.00 |
| Others | 0.00 |

Corner solution — all capital allocated to the highest-$\mu$ company.

#### Target-Return ($\mu \ge 87.37$) Minimum-Variance Portfolio
| Top 10 Firms | Weight |
|:--|--:|
| Dropbox | 0.218 |
| LinkedIn | 0.073 |
| Dell | 0.065 |
| Qualcomm | 0.061 |
| Salesforce | 0.049 |
| AT&T | 0.046 |
| Intel | 0.042 |
| Oracle | 0.037 |
| Symantec | 0.036 |
| Adobe | 0.036 |

Diversified portfolio achieves required return with minimal risk.

---

## 🔬 Interpretation Across Phases

| Phase | Mathematical Core | Key Output | Real-World Meaning |
|:--|:--|:--|:--|
| 1 | Quadratic program with linear inequalities | $c^\star$ and predicted prices | Filled missing deal values |
| 2 | Convex log-det precision estimation | $(\mu,\Sigma)$ | Risk–return structure across firms |
| 3 | Quadratic program (mean–variance) | $w^\star$ | Optimal portfolio allocation |

---

## 🧠 Key Insights
- Convex optimisation offers **global optimality** and **interpretability** for complex financial pipelines.  
- Censored regression robustly imputes missing valuations.  
- Log-det covariance regularisation ensures positive-definite, stable risk estimates.  
- Portfolio optimisation clearly illustrates the **risk–return trade-off** and efficient frontier.  

---

## 🏁 Final Conclusion
Through convex optimisation, incomplete and noisy M&A data can be transformed into a **coherent, risk-aware investment framework**.

1. **Phase 1** reconstructed undisclosed acquisition prices, ensuring a complete dataset.  
2. **Phase 2** derived statistically consistent means and covariances, capturing firm inter-dependencies.  
3. **Phase 3** translated these statistics into optimal portfolios — from single-firm corner solutions to diversified, minimum-risk allocations.  

> The workflow demonstrates how convex optimisation unifies estimation, uncertainty quantification, and decision-making in a single mathematical language.  
> Each stage was convex, DCP-compliant, and solved to global optimality using CVXPY.

---

## 🧰 Environment
| Library | Version | Role |
|:--|:--|:--|
| Python | ≥ 3.10 | environment |
| CVXPY | ≥ 1.7 | convex solvers |
| NumPy / Pandas | ≥ 2.0 / 2.1 | numerical + data prep |
| OSQP / SCS | latest | QP + conic solvers |

---

## 📂 Repository Structure
```
├── opti-proj.ipynb            # Full notebook (Phases 1–3)
├── Convex_Optimisation_Report.docx  # Formal report
├── README.md                  # (this file)
├── data/
│   ├── acquisitions.csv
│   └── merged_phase1.csv
└── results/
    ├── frontier.csv
    ├── portfolio_weights.csv
    └── covariance_heatmap.png
```

---

## 📚 References
- S. Boyd & L. Vandenberghe, *Convex Optimisation*, Cambridge University Press (2004)  
- S. Diamond & S. Boyd, “CVXPY: A Python-Embedded Modeling Language for Convex Optimization,” JMLR (2016)
