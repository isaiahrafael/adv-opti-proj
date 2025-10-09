# Advanced Portfolio Optimisation Pipeline

End-to-end workflow for preparing returns, estimating covariance, and constructing a Markowitz efficient frontier from SP500-style panel data. The primary entry point is the notebook `adv-opti-project.ipynb`.

**Phases**
- Phase 1 — Censored target construction and bounds
- Phase 2 — Covariance estimation (regularised)
- Phase 3 — Markowitz mean–variance portfolio and frontier

---

## Project Structure
- `adv-opti-project.ipynb` — main notebook with the full pipeline
- `financial_data_sp500_extended_synth_return_censored_bounds_FIXED.csv` — input for Phase 1
- `financial_data_sp500_convex_imputed.csv` — imputed return panel used in Phases 2–3
- `results/` — saved artifacts (frontier, weights, plots)

---

## Phase 1 — Censored Target + Bounds
Build a clean target “return” column and censor a subset of rows to simulate missing labels, while keeping predictors intact.

Key steps:
- Auto-detect important columns (company key, date, gross profit, operating expense)
- Compute `return = GrossProfit / OperatingExpense` with safe division
- Randomly censor only the `return` values to create `is_hole` flags
- Compute per-company bounds for censored rows from original history:
  - global: min/max over all history
  - trailing: min/max within a trailing window before the hole date
- Sort and save the extended dataset

Outputs:
- `financial_data_sp500_extended_synth_return_censored_bounds.csv`

Notes:
- Predictors are forward-filled per company, then filled with company medians from the original data; remaining NaNs → 0
- Only the target `return` is censored — predictors remain usable for modeling

---

## Phase 2 — Covariance Estimation (Panel of Returns)
Load an imputed panel of firm returns (wide format: dates × firms), impute any residual gaps by column mean, then compute expected returns and a regularised covariance matrix.

Key steps:
- Build full panel across all months and firms from `financial_data_sp500_convex_imputed.csv`
- Mean-impute per firm; fallback to zero if a column is entirely missing
- Compute `mu` (per-firm mean) and centre data
- Regularise covariance:
  - Prefer `LedoitWolf` shrinkage (from scikit-learn)
  - Fallback to ridge shrinkage if Ledoit–Wolf isn’t available
- Ensure numerical stability (symmetrise, small diagonal jitter)

Artifacts in-memory:
- `panel_imputed` (DataFrame T×n), `mu` (n,), `Sigma_reg` (n×n)

Optional outputs (can be added):
- `phase2_panel_full.csv`, `phase2_mu_full.csv`, `phase2_Sigma_full.csv`

---

## Phase 3 — Markowitz Efficient Frontier
Solve a long-only mean–variance problem for a grid of risk aversion values `gamma`, using projected gradient descent (PGD) with projection onto the simplex (`w ≥ 0`, `sum w = 1`).

Key steps:
- Helper: Euclidean projection to the simplex for feasibility
- PGD optimiser for `min_w -mu^T w + gamma w^T Σ w`
- Sweep `gamma` (log-spaced) to build the frontier
- Compute and print key portfolios: max Sharpe (tangency), min variance, and max return
- Visualise frontier, Sharpe profile, top firms, and concentration (HHI)

Saved outputs (if enabled in the notebook):
- `results/frontier.csv` — columns: `gamma, expected_return, risk_std`
- `results/portfolio_weights.csv` — top portfolio weights (example export)
- `results/covariance_heatmap.png` — optional visualisation

---

## How To Run
1) Open `adv-opti-project.ipynb` and run all cells in order.
2) Inspect artifacts:
   - View all 40 frontier points in the `frontier` DataFrame (`frontier`, `frontier.tail()`).
   - Explore weights with `pd.Series(weights[i], index=firms)` for any frontier index `i`.
3) Persist results by uncommenting the `to_csv(...)` lines in Phase 3.

Dependencies used in the notebook’s first import cell:
- `numpy`, `pandas`, `matplotlib`
- `cvxpy` (optional for alternative solvers)
- `scikit-learn` (`LedoitWolf`)

---

## Practical Notes
- Check return scaling: ensure input returns are in decimals (e.g., 0.01 = 1%) before optimisation to keep risk/Sharpe magnitudes realistic.
- Add constraints as needed (e.g., position caps) if concentration is undesirable.
- Reproducibility: random seeds are set where applicable; results depend on the chosen imputation and regularisation methods.

