# Day 5: A Tutorial Introduction to the Minimum Description Length Principle

> Grünwald (2004) — [arXiv:math/0406077](https://arxiv.org/abs/math/0406077)

**Time:** 3-4 hours
**Prerequisites:** Day 4 (Hinton's MDL for neural nets)
**Code:** NumPy + SciPy

---

## What This Paper Is Actually About

This is an ~80-page tutorial/survey by Peter Grünwald that systematizes the MDL principle — the idea that model selection is code selection, and the best model is the one that compresses data the most.

Day 4 (Hinton & van Camp) used MDL to regularize neural network weights. Today's paper explains the **mathematical foundation** underneath that idea: what "description length" means formally, how to compute it for different model classes, and why minimizing it selects good models.

### The paper covers:

1. **Two-part codes** — the simplest MDL formulation: L(H) + L(D|H)
2. **Prequential (predictive) codes** — MDL via sequential prediction
3. **Normalized Maximum Likelihood (NML)** — the minimax-optimal universal code
4. **Parametric and stochastic complexity** — the log of NML's normaliser (COMP) as model complexity, and the data's NML code length
5. **Connection to Kolmogorov complexity** — MDL as a practical approximation of algorithmic complexity
6. **Comparison with AIC, BIC, cross-validation**

---

## The Core Concept: Two-Part Codes

To communicate data $D$ to someone, you can either send it raw, or send a model plus residuals. If the model captures real structure, the total message is shorter.

**The MDL Principle:**

$$\text{Best model} = \arg\min_H \left[ L(H) + L(D \mid H) \right]$$

- $L(H)$ = bits to describe the model
- $L(D \mid H)$ = bits to describe the data given the model (residuals encoded as a code)

### Why this prevents overfitting

| Scenario | L(H) | L(D\|H) | Total |
|----------|-------|---------|-------|
| Too simple | Small | Large (poor fit) | Large |
| Just right | Moderate | Small (good fit) | **Minimum** |
| Too complex | Large (many params) | Near zero | Large |
| Extreme overfit | Exceeds raw data cost | Zero | Worse than no model |

The crude version still has a knob: how many bits a coefficient costs (the tutorial's "bad news", Sec. 2.4.2; the optimal precision gives about $\frac12\log_2 n$ bits per parameter, eq. 2.9). Refined MDL replaces the hypothesis code with the model's parametric complexity (NML, below).

### Encoding details

**Model description** (for parametric models):
$$L(H) = k \cdot b \text{ bits}$$
where $k$ = number of parameters, $b$ = bits per parameter.

**Data given model** (Gaussian residuals):
$$L(D \mid H) = \frac{n}{2}\log_2(2\pi e \hat{\sigma}^2)$$

---

## Prequential MDL

Instead of describing a model then encoding residuals, predict each point using the previous points:

$$L_{\text{preq}} = \sum_{i=1}^{n} -\log_2 P(x_i \mid x_1, \ldots, x_{i-1})$$

This is a valid code length (Kraft inequality holds). No model description is needed — the model class's complexity is implicitly penalized through prediction performance.

**Advantages over two-part codes:**
- No arbitrary model encoding scheme
- Natural for sequential/online data
- Connected to Bayesian model averaging via the plug-in code

---

## Normalized Maximum Likelihood (NML)

The tutorial's theoretical gold standard (Sec. 2.5.3). Given model class $\mathcal{M}$:

$$P_{\text{NML}}(x^n) = \frac{P(x^n \mid \hat{\theta}(x^n))}{\sum_{y^n} P(y^n \mid \hat{\theta}(y^n))}$$

Numerator: the best the model can do on this data. Denominator: the same quantity summed (integrated, for continuous data) over all possible datasets, which measures how many datasets the model can fit well. Its log is the **parametric complexity** (eq. 2.17), and NML's regret equals it for every dataset (eq. 2.19):

$$\text{COMP}_n(\mathcal{M}) = \log \sum_{y^n} P(y^n \mid \hat{\theta}(y^n)) = \frac{k}{2}\log\frac{n}{2\pi} + \log\int_\Theta \sqrt{\det I(\theta)}\,d\theta + o(1) \qquad \text{(eq. 2.21, Sec. 2.6)}$$

The integral is the model's volume counted in distinguishable distributions; it is why two models with the same number of parameters $k$ can differ in complexity (Example 1.4). The code length $-\log P(D\mid\hat\theta) + \text{COMP}_n$ is the data's *stochastic complexity* (eq. 2.20).

### Why plain NML is undefined for polynomial regression

For Gaussian models the integral is **infinite**. A polynomial fits data centred at $10^6$ exactly as well as data centred at 0, so integrating over all datasets diverges (Example 2.20, Sec. 2.7.2). With $\sigma$ fitted it diverges a second time, as the residual variance goes to 0. Sec. 2.8 says the NML distribution for regression "is not well-defined". The remedies are in Sec. 2.7.2: restrict the parameters, then pay for the restriction with a meta-two-part code (eq. 2.34) or renormalise (RNML, eq. 2.36). Sec. 2.10.2 warns never to cut the range at an arbitrary value.

### What the code computes

- `nml_complexity_polynomial(x, degree, R, sigma)`: COMP of the polynomial model with known noise $\sigma$ and the fitted curve's mean square bounded, $\frac1n\sum_i (X\beta)_i^2 \le R$. Exact closed form, whose leading term is eq. 2.21:
  $$\text{COMP} \approx \frac{k}{2}\log_2\frac{n}{2\pi} + \log_2\Big[V_k\,\big(R/\sigma^2\big)^{k/2}\Big] \text{ bits}, \qquad V_k = \text{volume of the unit ball in } \mathbb{R}^k$$
  The second term is $\log_2\int\sqrt{\det I}\,d\beta$ over the allowed region, a $k$-ball of radius $\sqrt R/\sigma$ in noise units. COMP is finite and $\ge 0$, grows with the degree, and grows by $\frac k2\log_2 10$ bits for every 10x increase of $R$: NML's price depends on how you bound the parameters. For degree 0 it reduces to Example 2.20's $\log K + \frac12\log\frac{n}{2\pi} - \log\sigma + 1$ with $K = \sqrt R$.
- `nml_mdl_polynomial(x, y, degree)` and `nml_model_selection`: selection with unknown $\sigma$ by Rissanen's (2000) renormalised NML for linear regression (eq. 2.36; the tutorial cites it for regression in Sec. 2.8). No bound is chosen by hand. Up to a constant that is the same for every degree, the code length is $\big[\frac{n-k}{2}\ln\hat\sigma^2 + \frac k2\ln\hat R - \ln\Gamma(\frac k2) - \ln\Gamma(\frac{n-k}{2}) + \frac n2\ln(n\pi)\big]/\ln 2$ bits, where $\hat R$ is the fitted curve's mean square.
- `nml_complexity_asymptotic` (eq. 2.21) and `nml_complexity_monte_carlo` check the closed form. A Monte Carlo estimate of the integral must divide by the sampling density; for a uniform box that means multiplying by the box volume. An earlier version left that factor out and reported about -380 bits.

`validation/day05_nml_check.py` verifies all of this: closed form against eq. 2.21 and against Monte Carlo, Rissanen's normalisers, and selection rates.

---

## MDL vs. AIC vs. BIC

The tutorial relates MDL to BIC (Sec. 2.9.2, eq. 2.42) but does not analyse AIC ("relations between MDL and Akaike's AIC are subtle", Sec. 2.9). The table below is standard background, with every penalty in bits ($-\log_2$ likelihood scale).

| Criterion | Penalty per Parameter (bits) | Derivation | Key Property |
|-----------|-----------------------|-----------|-------------|
| AIC | $1/\ln 2 \approx 1.44$ (that is 2 on the $-2\ln L$ scale) | Expected KL divergence (prediction) | Does not grow with $n$; over-fits in our experiments |
| BIC | $\frac{1}{2}\log_2 n$ | Laplace approx. to Bayesian marginal likelihood | Consistent (finds true model as $n \to \infty$) |
| MDL (NML) | $\approx \frac{1}{2}\log_2\frac{n}{2\pi}$ plus a share of $\log_2\int\sqrt{\det I}$ | Minimax regret universal code (Sec. 2.5.3) | Sees functional form; for regression it needs a bounded parameter space |

**Key differences:**
- AIC's penalty doesn't grow with $n$, so it over-fits; over 100 datasets in `train_minimal.py --monte-carlo` it picks the true degree 58 times
- BIC matches refined MDL to leading order for a fixed number of parameters, but "MDL = BIC" is wrong in general (Sec. 2.9.2)
- MDL's complexity also depends on the model's functional form through $\log\int\sqrt{\det I}$, not only on $k$ (Example 1.4)
- MDL doesn't assume a "true model" exists in the class (Sec. 1.5)

---

## Connection to Kolmogorov Complexity

MDL is the practical approximation of Kolmogorov complexity $K(x)$ — the length of the shortest program that produces $x$.

$K(x)$ is uncomputable (halting problem), and for small samples it depends on the choice of programming language (Sec. 1.2.1). Practical MDL therefore restricts itself to codes built from a model class and minimizes within it (Sec. 1.2.2).

Grünwald is careful to note: MDL captures statistical regularity relative to a model class. It does not claim to find "true understanding" in any philosophical sense. Domain knowledge matters for choosing the model class.

---

## The Architecture

```
MDL Model Selection Pipeline:

  Input: Data D, Candidate models {H_1, H_2, ..., H_k}

  For each model H_i:
    1. Compute L(H_i)     -- model description length
    2. Fit H_i to D       -- maximum likelihood
    3. Compute L(D|H_i)   -- residual code length
    4. Score = L(H_i) + L(D|H_i)

  Output: H* = argmin Score(H_i)
```

---

## What to Build

### Quick Start

```bash
python train_minimal.py                  # Polynomial degree selection demo
python train_minimal.py --monte-carlo    # Statistical comparison across trials
```

### Training Script

The demo generates noisy polynomial data (true degree = 2, noise 1.5, 50 points) and selects the degree with the crude two-part code (MDL), Rissanen's renormalised NML, AIC, and BIC. Default run: MDL, NML and BIC pick degree 2, AIC picks 3. With `--monte-carlo` (100 datasets with random coefficients) the true degree is picked 87 times by the two-part code, 97 by NML, 85 by BIC and 58 by AIC.

---

## Exercises (in `Exercises/`)

| # | Task | What You'll Learn | Source |
|---|------|-------------------|--------|
| 1 | Two-Part Code | Basic MDL scoring for polynomials | Paper, Secs. 1.3 and 2.4 |
| 2 | Prequential MDL | Sequential prediction coding | Paper, Sec. 2.6.4 (eqs. 2.29-2.30) |
| 3 | Model Selection | Compare polynomial degrees with MDL | Paper, Example 1.2 |
| 4 | NML Complexity | Restricted NML complexity, Monte Carlo check, RNML selection | Paper, Secs. 2.5.3, 2.6, 2.7.2, 2.8 |
| 5 | MDL vs AIC vs BIC | Head-to-head comparison | Paper, Sec. 2.9.2 (BIC; AIC is not treated) |

All exercises implement concepts directly from the paper. The Monte Carlo robustness test in exercise 5 is our extension.

Solutions in `solutions.py`.

---

## Key Takeaways

1. **MDL = compression-based model selection.** The best model minimizes total description length: model bits + residual bits.

2. **Three formulations, one principle.** Two-part codes (simple), prequential codes (sequential), NML (optimal) — all minimize description length.

3. **MDL's penalty is the model's own complexity.** Unlike AIC (fixed 2k) or BIC (fixed k·log n), NML's complexity depends on the model's functional form and on how its parameters are bounded; for Gaussian regression it is only finite once they are bounded (Sec. 2.7.2).

4. **Connection to Kolmogorov complexity.** MDL is the practical, computable approximation of algorithmic complexity.

5. **No "true model" assumption.** MDL works from compression alone — it doesn't assume the data was generated by any model in the class.

---

## Implementation Notes

Our code in `implementation.py` provides:

- `two_part_mdl_polynomial()` — the crude two-part MDL score for regression (Secs. 1.3 and 2.4)
- `prequential_mdl_polynomial()` — sequential prediction coding (Sec. 2.6.4)
- `nml_complexity_polynomial()` — exact parametric complexity of the restricted polynomial model, sigma known (Secs. 2.5.3, 2.6, 2.7.2); `nml_complexity_asymptotic()` is eq. 2.21 and `nml_complexity_monte_carlo()` a Monte Carlo check
- `nml_mdl_polynomial()` / `nml_model_selection()` — Rissanen's renormalised NML for unknown sigma (eq. 2.36, Sec. 2.8)
- `PolynomialMDL` — model selection for polynomial regression (the tutorial's Example 1.2)
- `compare_mdl_aic_bic()` — MDL vs AIC vs BIC scoring

Polynomial selection is the tutorial's Example 1.2 (the running example of Chapter 2 is Markov chains). The Monte Carlo comparison (running the experiment many times to compare accuracy across criteria) is our addition.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | MDL computation library: two-part, prequential, NML |
| `train_minimal.py` | Polynomial degree selection demo + Monte Carlo comparison |
| `visualization.py` | MDL score plots, fit comparisons, criterion comparison charts |
| `notebook.ipynb` | Interactive walkthrough |
| `Exercises/` | 5 exercises covering all three MDL variants |
| `paper_notes.md` | Detailed notes on Grünwald's tutorial |
| `CHEATSHEET.md` | Quick reference for MDL formulas |

---

## Further Reading

- [Grünwald (2004)](https://arxiv.org/abs/math/0406077) — this paper
- Grünwald, *The Minimum Description Length Principle* (MIT Press, 2007) — the full textbook
- Rissanen, *Stochastic Complexity in Statistical Inquiry* (1989) — the original MDL inventor
- [Hinton & van Camp (1993)](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf) — Day 4, MDL applied to neural networks

---

**Next:** [Day 6](../06_Complexodynamics/)
