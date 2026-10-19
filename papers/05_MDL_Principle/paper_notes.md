# Paper Notes: A Tutorial Introduction to the Minimum Description Length Principle

## ELI5 (Explain Like I'm 5)

You have data and you want to find the pattern in it. MDL says: the best model is the one that lets you compress the data the most. Not the model with the best fit, and not the simplest model — the best *compressor*.

The intuition: if your model captures real structure in the data, you can describe the data briefly ("here's the pattern, plus these small corrections"). If your model is wrong or too complex, the total description is longer.

This is formalized as: **Total cost = cost to describe the model + cost to describe the data given the model.** Minimize the sum.

Note: This sounds simple, but the paper's real contribution is making it rigorous — showing exactly how to measure "description length" for different model classes, and proving that different approaches (two-part codes, prequential codes, NML) are all valid and connected.

---

## What This Paper Actually Is

**"A Tutorial Introduction to the Minimum Description Length Principle"** by Peter Grünwald (2004). Published in *Advances in Minimum Description Length: Theory and Applications*.

This is a **tutorial/survey paper** — roughly 80 pages covering the theory, philosophy, and practice of MDL. It is not a single-result paper. It systematizes decades of work by Rissanen, Wallace, and others into a coherent framework.

Grünwald later expanded this into a full textbook: *The Minimum Description Length Principle* (MIT Press, 2007).

### What the paper covers

1. **Two-part codes** (crude MDL) — the simplest formulation
2. **Prequential codes** — sequential prediction without explicit model description
3. **Normalized Maximum Likelihood (NML)** — the theoretically optimal universal code
4. **Stochastic complexity** — the NML codelength as a model selection criterion
5. **Connection to Kolmogorov complexity** — MDL as a practical approximation of algorithmic complexity
6. **Comparison with AIC, BIC, cross-validation** — when MDL agrees and disagrees with other criteria
7. **Consistency results** — when and why MDL selects the correct model asymptotically

---

## The Core Framework: Two-Part Codes

### The Idea

To communicate a dataset $D$ to someone, you could send the raw data. Or you could:
1. First send a model $H$ (a pattern or hypothesis)
2. Then send the deviations from that model (residuals)

If the model captures real structure, the total message is shorter than sending raw data.

### The MDL Principle (Two-Part Code Version)

Choose the hypothesis $H$ that minimizes:

$$L(H) + L(D \mid H)$$

where:
- $L(H)$ = bits to describe the model (parameters, structure)
- $L(D \mid H)$ = bits to describe the data given the model (residual errors encoded as a code)

### Why This Prevents Overfitting

- A too-simple model: $L(H)$ is small, but $L(D \mid H)$ is large (poor fit, many residuals to encode)
- A too-complex model: $L(D \mid H)$ is small (good fit), but $L(H)$ is large (many parameters to specify)
- An overfit model (e.g., degree-99 polynomial for 100 points): $L(H)$ can exceed the raw data cost. The "model" is longer than just listing the data.

The sum finds the balance, but the crude version still has a knob: how many bits a coefficient costs. Different hypothesis codes prefer different models (the tutorial's "bad news", Sec. 2.4.2); coding each parameter to the optimal precision ~1/sqrt(n) costs about (1/2) log2 n bits (eq. 2.9).

### Example: Polynomial Degree Selection

`train_minimal.py`'s default data: a degree-2 polynomial plus noise (sigma 1.5, 50 points). The crude code spends 33 bits per coefficient plus 42 bits for sigma; the last column is the refined (NML) code length of Sec. 2.6, computed as described below. Numbers are bits.

| Degree | L(H) (model bits) | L(D\|H) (residual + sigma bits) | Two-part total | NML: fit + complexity |
|--------|-------------------|------------------------|-------|-------|
| 0 | 34.5 | 388.2 | 422.7 | 346.2 + 2.6 = 348.9 |
| 1 | 68.5 | 294.7 | 363.2 | 252.7 + 10.4 = 263.2 |
| 2 | 102.8 | 165.7 | **268.5** | 123.7 + 23.1 = **146.8** |
| 3 | 136.5 | 163.5 | 300.0 | 121.5 + 30.4 = 151.9 |
| 4 | 170.3 | 162.5 | 332.9 | 120.5 + 37.4 = 158.0 |
| 9 | 337.4 | 161.3 | 498.7 | 119.3 + 69.9 = 189.3 |

Both select degree 2, the true generating degree. Going from degree 2 to 3 saves about 2 bits of fit; the crude code charges 33.7 more bits of model cost, NML 7.3 more bits of complexity.

---

## Encoding Details: How to Count Bits

### Model Description Length L(H)

For a parametric model with $k$ parameters at $b$ bits of precision each:

$$L(H) = k \cdot b$$

This is the crude version. The paper discusses more refined approaches:
- **Universal integer codes** for model order (e.g., polynomial degree)
- **Optimal precision** — coding each parameter to precision about $1/\sqrt{n}$ costs about $\frac12\log n$ bits per parameter, giving the BIC-like form of eq. 2.9; finer precision wastes bits (Sec. 2.4.2)
- **Two-stage codes** — first encode the model class, then the parameters within that class

### Data Description Length L(D|H)

For regression with Gaussian residuals:

$$L(D \mid H) = \frac{n}{2} \log_2(2\pi e \hat{\sigma}^2)$$

where $\hat{\sigma}^2$ is the residual variance. Better fit (smaller $\hat{\sigma}^2$) = fewer bits.

For classification, it's the negative log-likelihood:

$$L(D \mid H) = -\sum_i \log_2 P(x_i \mid H)$$

---

## Beyond Two-Part Codes: Prequential MDL

### The Problem with Two-Part Codes

The two-part code requires choosing how to encode the model, which introduces some arbitrariness (how many bits per parameter? what code for the model structure?). Different encoding choices can lead to different MDL rankings.

### The Prequential Solution

Instead of describing a model and then residuals, predict each data point using only the previous data points:

$$L_{\text{preq}} = \sum_{i=1}^{n} -\log_2 P(x_i \mid x_1, \ldots, x_{i-1})$$

This is the **cumulative log-loss** (surprise) of a sequential predictor. No model description is needed — the model is implicitly defined by the prediction procedure.

**Key property:** This is a valid code length (it satisfies the Kraft inequality), so it's a legitimate MDL score even though it never explicitly describes a model.

**Advantages over two-part codes:**
- No arbitrary model description length
- Natural for online/sequential settings
- The model class's complexity is automatically accounted for through prediction performance

Under regularity conditions the plug-in code pays the same $\frac{k}{2}\log n$ penalty as NML (Sec. 2.6.4, eq. 2.30). The plain maximum-likelihood plug-in can fail at small samples (after "0, 0" it gives the next "1" probability zero, Example 2.19). Laplace's estimator ($\lambda = 1$, equivalent to Bayes with a uniform prior) and Jeffreys' ($\lambda = 1/2$) avoid this.

---

## The Gold Standard: Normalized Maximum Likelihood (NML)

### The Idea

Given a model class $\mathcal{M}$ (e.g., "all polynomials of degree $\leq k$"), define:

$$P_{\text{NML}}(x^n) = \frac{P(x^n \mid \hat{\theta}(x^n))}{\sum_{y^n} P(y^n \mid \hat{\theta}(y^n))}$$

where $\hat{\theta}(x^n)$ is the maximum likelihood estimate for data $x^n$.

### What This Means

The numerator is the best the model class can do on THIS data (maximum likelihood). The denominator sums this over ALL possible datasets — it measures how flexible the model class is.

A more flexible model class has a larger denominator (it can fit many different datasets well), so each individual dataset gets a lower probability — more bits to encode.

### Parametric and Stochastic Complexity

The log of NML's normaliser is the **parametric complexity** of the model class (Sec. 2.5.3, eq. 2.17):

$$\text{COMP}_n(\mathcal{M}) = \log \sum_{y^n} P(y^n \mid \hat{\theta}(y^n))$$

It measures how many different datasets the model class can fit well, and NML's regret equals it for every dataset (eq. 2.19). The NML code length of the data, $-\log P(x^n \mid \hat\theta(x^n)) + \text{COMP}_n(\mathcal{M})$, is the data's **stochastic complexity** (eq. 2.20). Asymptotically (eq. 2.21, Sec. 2.6),

$$\text{COMP}_n(\mathcal{M}) = \frac{k}{2} \log \frac{n}{2\pi} + \log \int_\Theta \sqrt{\det I(\theta)}\, d\theta + o(1)$$

where $k$ is the number of parameters and $I$ the Fisher information. The first term is similar to the BIC penalty; the integral is the model's volume counted in distinguishable distributions, and it is what lets two models with the same $k$ differ (Example 1.4). The tutorial warns that the expansion can be badly off at small $n$ and is valid only for $k \ll n$ (Sec. 2.6.2, Sec. 2.10.2).

### Gaussian Models: COMP Is Infinite

For Gaussian models the integral diverges. In the location model with known $\sigma$, the data's own mean is always a perfect fit, so the maximised likelihood does not shrink as the data move away from 0 and the integral over all datasets is infinite (Example 2.20). In regression with $\sigma$ fitted it diverges a second time, as the residual variance goes to 0. Sec. 2.8 states that the NML distribution for regression "is not well-defined". The tutorial's remedies (Sec. 2.7.2):

- **Restrict, then pay for the restriction.** Bound the parameters (e.g. $|\hat\mu| \le K$, giving $\text{COMP} = \log K + \frac12\log\frac{n}{2\pi} - \log\sigma + 1$), and encode the bound first with a meta-two-part code (eqs. 2.34-2.35). Restricting the parameter values instead of the data is Foster and Stine's variant.
- **Renormalise** (Rissanen's RNML, eq. 2.36): choose the bound from the data, then normalise again.
- Never cut the range at an arbitrary value (Sec. 2.10.2).

**What our code does.** `nml_complexity_polynomial` computes COMP for the polynomial model with known $\sigma$ and the fitted curve's mean square bounded by $R$. Its leading term is eq. 2.21, $\frac k2\log_2\frac{n}{2\pi} + \log_2[V_k (R/\sigma^2)^{k/2}]$ bits ($V_k$ = unit-ball volume): finite, $\ge 0$, growing with the degree and by $\frac k2\log_2 10$ bits per 10x of $R$. `nml_mdl_polynomial` selects the degree with Rissanen's (2000) RNML for unknown $\sigma$. Both are checked against eq. 2.21 and Monte Carlo in `validation/day05_nml_check.py`. Over 100 datasets per setting, RNML picks the true degree 99 and 97 times; BIC does so 93 and 85 times.

---

## MDL vs. AIC vs. BIC

The tutorial relates MDL to BIC (Sec. 2.9.2, eq. 2.42) but deliberately does not analyse AIC: "relations between MDL and Akaike's AIC are subtle" (Sec. 2.9). The table is standard background; penalties are in bits per parameter (the $-\log_2$ likelihood scale).

| Criterion | Penalty (bits per parameter) | Derivation | Properties |
|-----------|---------|-----------|------------|
| AIC | $1/\ln 2 \approx 1.44$ ($2k$ on the $-2\ln L$ scale) | Expected KL divergence (prediction) | Penalty does not grow with $n$; over-fits in our experiments |
| BIC | $\frac12\log_2 n$ ($k \ln n$ on the $-2\ln L$ scale) | Laplace approximation to Bayesian marginal likelihood | Consistent (selects true model as $n \to \infty$) |
| MDL (NML) | $\approx \frac{1}{2} \log_2 \frac{n}{2\pi}$ plus the model's share of $\log_2\int\sqrt{\det I}$ | Minimax regret universal code (Sec. 2.5.3) | Sees functional form; for regression it needs a bounded parameter space |

### Key differences

1. **AIC under-penalizes for finding the true degree.** Its penalty does not grow with $n$; in `train_minimal.py --monte-carlo` it picks the true degree in 58 of 100 datasets (BIC 85, NML 97). This is our experiment, not the tutorial's.

2. **BIC is close to MDL, but not equal.** For a fixed number of parameters, BIC's $\frac{k}{2}\log n$ is the leading term of COMP (eq. 2.21). The tutorial still calls "MDL = BIC" wrong: the $O(1)$ terms (functional form) matter at small $n$, and when the number of parameters grows with $n$ each one costs much less than $\frac12 \log n$ (Sec. 2.9.2).

3. **MDL sees functional form.** The $\log\int\sqrt{\det I(\theta)}\,d\theta$ term of eq. 2.21 lets two models with the same $k$ differ in complexity (Stevens' vs Fechner's law, Example 1.4). AIC and BIC only count parameters. For Gaussian regression the integral, and COMP, is infinite unless the parameters are bounded, so NML's price also depends on the bound (Sec. 2.7.2).

4. **MDL does not assume a "true model" exists.** It is motivated by compression and does not need the data to come from a distribution in the class (Sec. 1.5).

---

## Connection to Kolmogorov Complexity

The paper traces MDL's intellectual lineage:

**Kolmogorov Complexity $K(x)$**: The length of the shortest program that produces $x$. This is the "ideal" description length — the absolute limit of compression.

**Problem**: $K(x)$ is uncomputable (undecidable), and for small samples it depends on arbitrary details of the programming language's syntax (Sec. 1.2.1).

**MDL's relationship**: practical MDL restricts itself to codes built from a model class and minimizes description length within that class (Sec. 1.2.2). Sec. 2.9.4 returns to Kolmogorov's structure function and "ideal MDL".

This connection, emphasized by Rissanen, gives MDL a philosophical foundation: learning is compression, and the best compression of a dataset captures its regularities.

### Important nuance from the paper

Grünwald is careful to distinguish MDL from naive "compression = intelligence" claims. MDL is relative to a model class. Without a good model class, MDL can't find patterns that the class can't express. Domain knowledge matters for choosing the model class.

---

## What the Paper Doesn't Do

- **No deep learning**: The examples are polynomials, Markov models, and simple parametric families. Neural networks are mentioned only in passing.
- **No general algorithm for NML**: the normaliser is usually evaluated numerically or approximated with eq. 2.21; the only closed form the tutorial knows of is the Gaussian case of Example 2.20 (Sec. 2.6). For regression it refers to Rissanen (2000), whose closed form our code uses.
- **No claim that MDL is always better**: Grünwald is explicit that MDL, AIC, and BIC all have regimes where they work well. MDL's advantage is theoretical principled-ness, not necessarily practical superiority in all settings.

---

## Our Implementation (Going Beyond the Paper)

The paper is theoretical. Our implementation makes it concrete:

1. **Two-part code model selection** — polynomial regression with bits computation (crude MDL, Secs. 1.3 and 2.4)
2. **Prequential MDL** — sequential prediction for time-series-like data (Sec. 2.6.4)
3. **NML complexity computation** — exact parametric complexity of the restricted polynomial model with known sigma, checked against eq. 2.21 and Monte Carlo; Rissanen's renormalised NML for selection with unknown sigma (Secs. 2.5.3, 2.6, 2.7.2, 2.8)
4. **MDL vs. AIC vs. BIC comparison** — head-to-head on polynomial selection (background for Sec. 2.9)
5. **Monte Carlo trials** — statistical comparison across many random datasets (our addition for robustness testing)

Polynomial selection is the tutorial's Example 1.2; the running example of its Chapter 2 is Markov chains. The Monte Carlo comparison is our extension.

---

## Key Equations Summary

| Concept | Equation | Source |
|---------|----------|--------|
| Two-Part MDL | $L(H) + L(D \mid H)$ | Paper, Sec. 1.3; Sec. 2.4 (eqs. 2.8-2.9) |
| Gaussian residual code | $\frac{n}{2}\log_2(2\pi e \hat{\sigma}^2)$ | Paper, eq. 2.40 at the ML fit (Sec. 2.8) |
| Prequential MDL | $\sum_i -\log_2 P(x_i \mid x_{<i})$ | Paper, Sec. 2.6.4 (eq. 2.29) |
| NML distribution | $P_{\text{NML}}(x) = P(x \mid \hat{\theta}(x)) / C_n$ | Paper, Sec. 2.5.3 (eq. 2.18) |
| Parametric complexity | $\text{COMP}_n = \log C_n \approx \frac{k}{2}\log\frac{n}{2\pi} + \log\int\sqrt{\det I}$ | Paper, eqs. 2.17 and 2.21 |
| Restricted Gaussian location COMP | $\log K + \frac12 \log\frac{n}{2\pi} - \log\sigma + 1$ | Paper, Example 2.20 (Sec. 2.7.2) |
| AIC | $-2\log L + 2k$ | Akaike (1974); not analysed in the tutorial (Sec. 2.9) |
| BIC | $-2\log L + k\log n$ | Schwarz (1978); Paper, Sec. 2.9.2 (eq. 2.42) |

---

## Questions Worth Thinking About

1. The two-part code requires choosing how many bits per parameter. What happens if you use 16-bit vs. 64-bit precision? How does this change which model MDL selects, and why?

2. Prequential MDL depends on the order you process the data. For i.i.d. data this doesn't matter asymptotically, but for small samples it can. What does this imply about using prequential MDL in practice?

3. Grünwald notes that MDL and Bayesian model selection are related but not identical. When do they disagree, and whose answer do you trust?

4. The paper argues that "compression = finding regularities." But a JPEG compresses an image using the DCT, which may have nothing to do with the image's semantic content. Does MDL really capture "understanding," or just statistical regularity?

---

**Next:** [Day 6](../06_Complexodynamics/)
