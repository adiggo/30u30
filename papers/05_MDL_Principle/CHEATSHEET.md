# MDL Principle Cheatsheet

Quick reference for Grünwald (2004) and our implementation.

---

## The Paper's Contribution (30 seconds)

**Problem:** How to select the best model from candidate models without overfitting?
**Insight:** Treat model selection as code selection. The best model compresses data the most.
**Result:** Three principled formulations (two-part, prequential, NML) that avoid arbitrary penalty terms.

This is an ~80-page tutorial/survey systematizing the MDL principle. The framework underlies Day 4's Bayesian weight compression and connects to Kolmogorov complexity.

---

## Key Equations

### Two-Part MDL (Crude)

```
Total cost = L(H) + L(D|H)
  L(H)   = k * b              (k params, b bits each)
  L(D|H) = (n/2) * log2(2*pi*e*var)   (Gaussian residuals)

Best model = argmin { L(H) + L(D|H) }
```

### Prequential MDL (Sequential)

```
L_preq = sum_i  -log2 P(x_i | x_1, ..., x_{i-1})
```

No model description needed. Cumulative prediction surprise.

### NML (Theoretical Gold Standard, Sec. 2.5.3)

```
P_NML(x) = P(x | theta_hat(x)) / C_n,   COMP_n = log C_n          (eqs. 2.17-2.18)
Stochastic complexity of the data = -log P(x | theta_hat(x)) + COMP_n   (eq. 2.20)

Asymptotics (eq. 2.21, Sec. 2.6):
  COMP_n ~ (k/2) log(n / 2pi) + log integral sqrt(det I(theta)) dtheta
```

**Gaussian regression: C_n is infinite** (Example 2.20, Sec. 2.7.2; Sec. 2.8). A polynomial fits
data centred at 10^6 as well as data centred at 0, and a fitted sigma can shrink to 0. Restrict first:

```
sigma known, fitted curve's mean square <= R  (nml_complexity_polynomial):
  COMP ~ (k/2) log2(n / 2pi) + log2[ V_k (R / sigma^2)^(k/2) ]    bits, V_k = unit-ball volume
  -> finite, >= 0, grows with the degree, +(k/2) log2 10 bits per 10x of R
sigma unknown, bounds chosen from the data and renormalised (Rissanen's RNML, eq. 2.36;
nml_mdl_polynomial), in nats before / ln 2:
  L(y | k) = (n-k)/2 ln sigma_hat^2 + k/2 ln R_hat - lnGamma(k/2) - lnGamma((n-k)/2) + n/2 ln(n pi)
```

---

## MDL vs AIC vs BIC

| Criterion | Penalty (bits per parameter) | Formula |
|-----------|---------|---------|
| AIC | 1/ln 2 = 1.44 (2 on the -2 lnL scale) | -2*logL + 2k |
| BIC | (1/2)*log2(n) (log n on the -2 lnL scale) | -2*logL + k*log(n) |
| MDL (NML) | ~(1/2)*log2(n/2pi) + share of log2 integral sqrt(det I) | -log2 L + COMP |

**Key difference:** AIC/BIC count parameters. NML's complexity also depends on the model's functional
form (the Fisher-information integral, Example 1.4) and, for regression, on how the parameters are bounded.
The tutorial treats BIC in Sec. 2.9.2 and does not analyse AIC (Sec. 2.9).

---

## Quick Start

```bash
# Polynomial degree selection demo
python train_minimal.py

# Monte Carlo comparison (MDL vs AIC vs BIC)
python train_minimal.py --monte-carlo
```

### In Python

```python
from implementation import PolynomialMDL

# Fit a degree-3 polynomial and get its MDL score
model = PolynomialMDL(degree=3)
model.fit(X, y)
score = model.total_mdl(X, y)

# Or use the standalone function
from implementation import two_part_mdl_polynomial
total, model_cost, data_cost = two_part_mdl_polynomial(X, y, degree=3)

# Refined MDL: restricted NML complexity (sigma known) and RNML selection (sigma unknown)
from implementation import nml_complexity_polynomial, nml_model_selection
comp_bits = nml_complexity_polynomial(X, degree=3, R=100.0, sigma=1.0)
best_degree = nml_model_selection(X, y, max_degree=10)['best_degree']
```

---

## Common Patterns

### Model Selection Loop

```python
best_score = float('inf')
for degree in range(1, max_degree + 1):
    coeffs = np.polyfit(X, y, degree)
    residuals = y - np.polyval(coeffs, X)
    score = (degree + 1) * 32 + (n/2) * np.log2(2*np.pi*np.e*np.var(residuals))
    if score < best_score:
        best_score, best_degree = score, degree
```

### Comparing Criteria

```python
def compare(log_likelihood, k, n):
    AIC = -2 * log_likelihood + 2 * k
    BIC = -2 * log_likelihood + k * np.log(n)
    MDL = -log_likelihood / np.log(2) + k * 32
    return AIC, BIC, MDL
```

---

## Debugging

### MDL selects too-simple model
- The crude code charges 33 bits per coefficient, about 12x BIC's (1/2)log2(n); the optimal precision costs ~(1/2)log2(n) bits per parameter (eq. 2.9). Try `nml_model_selection`
- May need more data — MDL is conservative with small samples

### MDL selects too-complex model
- Residual variance may be underestimated
- Check that L(D|H) computation uses unbiased variance

### All criteria agree
- Normal for well-separated models. Disagreements show up when models are close in quality.

---

## What's From the Paper vs. Our Additions

| Concept | Source |
|---------|--------|
| Two-part codes: L(H) + L(D\|H) | Paper, Secs. 1.3 and 2.4 (eqs. 2.8-2.9) |
| Prequential MDL | Paper, Sec. 2.6.4 (eqs. 2.29-2.30) |
| NML / regret / parametric complexity | Paper, Sec. 2.5.3 (eqs. 2.15-2.19); asymptotics Sec. 2.6 (eq. 2.21) |
| Infinite COMP and its remedies | Paper, Sec. 2.7.2 (Example 2.20, eqs. 2.33-2.36); regression Sec. 2.8 |
| Restricted-NML closed form for polynomials, RNML selection | Rissanen (2000), our derivation and checks |
| MDL vs BIC (AIC not treated) | Paper, Sec. 2.9.2 (eq. 2.42) |
| Kolmogorov complexity connection | Paper, Secs. 1.2.1 and 2.9.4 |
| Polynomial degree selection example | Paper, Example 1.2 |
| Monte Carlo robustness comparison | Our addition |
| All exercises | Our pedagogical additions implementing paper concepts |

---

*For paper details, see [paper_notes.md](paper_notes.md). For implementation guide, see [README.md](README.md).*
