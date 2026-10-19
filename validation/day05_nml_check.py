"""Day 05: checks of the redesigned NML complexity (papers/05_MDL_Principle/implementation.py,
Section 4) against the tutorial (Grunwald 2004, arXiv math/0406077).

Plain NML is undefined for Gaussian regression (Example 2.20, Sec. 2.7.2; Sec. 2.8). The code
now computes the complexity of a RESTRICTED model (sigma known, mean squared fit <= R) and selects
the degree with Rissanen's renormalised NML (RNML, eq. 2.36) when sigma is unknown.

1. Pre-fix estimator, re-implemented exactly as shipped before 2026-10-18: about -380 bits, and it
   shifts by -n log2 10 when the box widens 10x (no volume term). With the volume put back it no
   longer depends on the box, but its target is infinite when sigma is fitted: on a tiny case the
   median estimate keeps growing with the number of samples, while the restricted sigma-known
   integral estimated the same way stays put.
2. COMP (sigma known) on a grid of n, degree and R/sigma^2: finite, >= 0, non-decreasing in
   degree; the restriction-size slope d COMP / d log2 R tends to k/2.
3. Closed form vs eq. 2.21. Eq. 2.21 is evaluated independently: the Fisher information from a
   finite-difference Hessian of the log-likelihood, and the volume of the restricted parameter set
   by Monte Carlo in a bounding box. It matches the 'data' closed form (and Example 2.20's formula
   for degree 0). The default 'parameters' closed form approaches it as n grows (gap ~ 1/sqrt n).
4. Closed form vs a Monte Carlo of the Shtarkov integral in the raw data space, written here
   independently: tiny cases (k = 1, n = 3; k = 2, n = 4; k = 3, n = 5), a uniform box (whose
   volume term is explicit) and a Gaussian proposal, both restrictions. Also
   implementation.nml_complexity_monte_carlo against the closed form.
5. Sigma unknown: Rissanen's closed form C(tau0, R) vs Monte Carlo (an upper bound on
   sigma_hat^2 keeps the region bounded); the RNML renormalisation constant
   (k^2/4) ln(R2/R1) ln(tau2/tau1) vs Monte Carlo; nml_mdl_polynomial = fit + log2 C + log2(k^2/4).
6. Selection rates over 100 seeds on the notebook's data (y = 3 + 2x - 0.5x^2, noise 2) and
   train_minimal's (random coefficients, noise 1.5): NML (RNML), crude two-part, prequential,
   BIC, AIC.
"""
import os
import sys
import time
import warnings
from math import lgamma, log, pi, sqrt

import numpy as np

warnings.simplefilter('ignore')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '05_MDL_Principle')
sys.path.insert(0, DAY)
import implementation as I  # noqa: E402

LN2 = log(2)
T0 = time.time()
results = []


def report(name, passed, detail):
    results.append((name, passed))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def log_mean_exp(log_w):
    """log of the mean of exp(log_w), and the relative standard error of that mean."""
    finite = log_w[np.isfinite(log_w)]
    top = finite.max()
    w = np.exp(log_w - top)                        # rescaled into [0, 1]; -inf -> 0
    return top + np.log(w.mean()), w.std() / np.sqrt(len(w)) / w.mean()


def basis(x, degree):
    """Orthonormal basis (n, k) of the column space of [1, x, ..., x^degree]."""
    return np.linalg.qr(np.vander(x / np.abs(x).max(), degree + 1, increasing=True))[0]


# =============================================================== 1. pre-fix estimator
def old_estimator(x, degree, y_range=(-100, 100), num_samples=1000):
    """implementation.nml_complexity_polynomial as shipped before 2026-10-18 (same RNG calls)."""
    n = len(x)
    lls = []
    for _ in range(num_samples):
        y_s = np.random.uniform(y_range[0], y_range[1], n)
        r = y_s - np.poly1d(np.polyfit(x, y_s, min(degree, n - 1)))(x)
        s = max(np.sqrt(np.mean(r**2)) if np.any(r) else 1e-10, 1e-10)
        lls.append(-0.5 * n * np.log(2 * np.pi * s**2) - np.sum(r**2) / (2 * s**2))
    lls = np.array(lls)
    top = lls.max()
    return (top + np.log(np.mean(np.exp(lls - top)))) / LN2


x50 = np.linspace(0, 10, 50)
np.random.seed(0)
old = [old_estimator(x50, d, (-100, 100), 500) for d in range(4)]
np.random.seed(0)
old_wide = old_estimator(x50, 2, (-1000, 1000), 500)
report("1a pre-fix estimator", all(v < 0 for v in old) and abs((old_wide - old[2]) + 50 * np.log2(10)) < 10,
       f"degrees 0-3: {[round(float(v), 1) for v in old]} bits (COMP must be >= 0); box x10 shifts degree 2 by "
       f"{old_wide - old[2]:.1f} vs -n log2 10 = {-50 * np.log2(10):.1f} (the missing volume term)")

# With the volume term the box estimator is scale-free, but with sigma fitted its target is
# infinite: near the column space sigma_hat -> 0 and f(y | theta_hat) = (2 pi e sigma_hat^2)^(-n/2)
# blows up faster than the volume shrinks. Sample means of such heavy-tailed weights keep growing.
rng = np.random.default_rng(1)
n_t, L = 4, 1.0
growth_fitted, growth_known = [], []
for N in (10**3, 10**4, 10**5, 10**6):
    est_fit, est_known = [], []
    for _ in range(15):
        Y = rng.uniform(-L, L, size=(N, n_t))                  # (N, 4) datasets in the box
        resid = Y - Y.mean(axis=1, keepdims=True)              # degree-0 fit: subtract each row's mean
        rss = np.sum(resid**2, axis=1)                         # (N,)
        log_f_fitted = -0.5 * n_t * np.log(2 * np.pi * np.e * rss / n_t)    # sigma fitted (old target)
        log_f_known = -0.5 * n_t * np.log(2 * np.pi * 0.1**2) - rss / (2 * 0.1**2)  # sigma = 0.1 known
        vol = n_t * np.log(2 * L)                              # + log box volume
        est_fit.append((log_mean_exp(log_f_fitted + vol)[0]) / LN2)
        est_known.append((log_mean_exp(log_f_known + vol)[0]) / LN2)
    growth_fitted.append(np.median(est_fit))
    growth_known.append(np.median(est_known))
slope_fit = np.polyfit(np.arange(4), growth_fitted, 1)[0]     # bits per decade of samples
slope_known = np.polyfit(np.arange(4), growth_known, 1)[0]
report("1b infinite target with sigma fitted", slope_fit > 0.5 and abs(slope_known) < 0.1,
       f"median log2 estimate (volume included), n=4, degree 0, N=1e3..1e6: fitted sigma "
       f"{[round(float(v), 2) for v in growth_fitted]} (+{slope_fit:.2f} bits/decade, theory ~1.1: diverges); "
       f"known sigma {[round(float(v), 2) for v in growth_known]} ({slope_known:+.2f} bits/decade: converges)")

# =============================================================== 2. properties of COMP
ok_prop, worst = True, ""
for n in (3, 10, 50, 200):
    x = np.linspace(0, 10, n)
    for snr in (0.0, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6):
        comps = [I.nml_complexity_polynomial(x, d, snr, 1.0) for d in range(min(11, n))]
        if not (all(np.isfinite(comps)) and min(comps) >= 0
                and all(b >= a - 1e-12 for a, b in zip(comps, comps[1:]))):
            ok_prop, worst = False, f"n={n}, R/sigma^2={snr}: {comps}"
report("2a COMP finite, >= 0, non-decreasing in degree (restrict='parameters')", ok_prop,
       worst or "all 4 n x 7 R/sigma^2 x up to 11 degrees")
ok_data = True
for n in (10, 50, 200):
    x = np.linspace(0, 10, n)
    for snr in (1.0, 1e2, 1e4):
        comps = [I.nml_complexity_polynomial(x, d, snr, 1.0, restrict='data') for d in range(min(11, n - 2))]
        ok_data &= min(comps) >= 0 and all(b >= a for a, b in zip(comps, comps[1:]))
neg = I.nml_complexity_polynomial(np.linspace(0, 10, 3), 0, 0.3, 1.0, restrict='data')
report("2b 'data' version (Example 2.20 style) >= 0 and monotone when R >= sigma^2, k <= n-2", ok_data,
       f"can be negative below that: n=3, degree 0, R=0.3: {neg:.3f} bits")
slopes = []
for d in range(6):
    c1 = I.nml_complexity_polynomial(x50, d, 1e6, 1.0)
    c2 = I.nml_complexity_polynomial(x50, d, 1e6 * 2**10, 1.0)
    slopes.append((c2 - c1) / 10)                               # bits per doubling of R
small = [I.nml_complexity_polynomial(x50, 2, R, 1.0) for R in (1e-2, 1, 1e2, 1e4)]
report("2c restriction-size dependence: d COMP / d log2 R -> k/2",
       all(abs(s_ - (d + 1) / 2) < 0.01 for d, s_ in enumerate(slopes)),
       f"slopes for degrees 0-5 at R ~ 1e6: {[round(s_, 4) for s_ in slopes]} (k/2 = 0.5..3); degree 2 at "
       f"R/sigma^2 = 1e-2, 1, 1e2, 1e4: {[round(v, 2) for v in small]} bits (+(3/2)log2 100 = 9.97 per step)")

# =============================================================== 3. closed form vs eq. 2.21
sigma = 1.0
rng = np.random.default_rng(2)
lines, ok21 = [], True
for d, R in ((0, 100.0), (1, 100.0), (2, 100.0)):
    k, n = d + 1, len(x50)
    X = np.vander(x50, k, increasing=True)
    y = rng.standard_normal(n)
    beta0 = rng.standard_normal(k)

    def loglik(b):
        return -0.5 * n * np.log(2 * np.pi * sigma**2) - np.sum((y - X @ b) ** 2) / (2 * sigma**2)
    h = 1e-3 / np.abs(X).max(axis=0)                            # step per coefficient (columns differ by 1e2)
    H = np.zeros((k, k))
    for i in range(k):                                          # central-difference Hessian
        for j in range(k):
            ei, ej = np.eye(k)[i] * h[i], np.eye(k)[j] * h[j]
            H[i, j] = (loglik(beta0 + ei + ej) - loglik(beta0 + ei - ej) - loglik(beta0 - ei + ej)
                       + loglik(beta0 - ei - ej)) / (4 * h[i] * h[j])
    fisher = -H / n                                             # per-observation Fisher information
    # Volume of {beta : (1/n)||X beta||^2 <= R} by Monte Carlo in its bounding box
    # (half-widths sqrt(n R [(X^T X)^-1]_ii), widened 1% so the box surely contains the set).
    half = 1.01 * np.sqrt(n * R * np.diag(np.linalg.inv(X.T @ X)))
    hits, N = 0, 0
    for _ in range(40):
        B = rng.uniform(-half, half, size=(100_000, k))         # (M, k) candidate coefficient vectors
        hits += np.count_nonzero(np.mean((B @ X.T) ** 2, axis=1) <= R)   # (M, n) fitted values -> mean square
        N += 100_000
    frac = hits / N
    vol = frac * np.prod(2 * half)
    eq221 = (k / 2) * np.log2(n / (2 * pi)) + np.log2(np.sqrt(np.linalg.det(fisher)) * vol)
    se = np.sqrt((1 - frac) / hits) / LN2                       # relative binomial error, in bits
    closed_data = I.nml_complexity_polynomial(x50, d, R, sigma, restrict='data')
    closed_par = I.nml_complexity_polynomial(x50, d, R, sigma)
    asym_fn = I.nml_complexity_asymptotic(x50, d, R, sigma)
    ok21 &= abs(eq221 - closed_data) < 4 * se + 1e-3 and abs(asym_fn - closed_data) < 1e-9
    lines.append(f"degree {d}: eq.2.21 (FD Fisher x MC volume) {eq221:.3f} +- {se:.3f}, nml_complexity_asymptotic "
                 f"{asym_fn:.4f}, closed 'data' {closed_data:.4f}, closed 'parameters' {closed_par:.4f}")
report("3a eq. 2.21 evaluated independently = 'data' closed form", ok21, "; ".join(lines))
ex220 = []
for n, R, sg in ((5, 4.0, 1.0), (50, 100.0, 2.0), (1000, 0.5, 0.3)):
    K = sqrt(R)
    tut = np.log2(K) + 0.5 * np.log2(n / (2 * pi)) - np.log2(sg) + 1
    ex220.append(abs(tut - I.nml_complexity_polynomial(np.linspace(0, 10, n), 0, R, sg, restrict='data')))
report("3b degree 0 = Example 2.20: log2 K + 1/2 log2(n/2pi) - log2 sigma + 1 (K = sqrt R)",
       max(ex220) < 1e-12, f"max |difference| {max(ex220):.1e} over 3 (n, R, sigma)")
gap_lines, ok_gap = [], True
for d in (0, 1, 2):
    gaps = []
    for n in (10, 100, 1000, 10**4, 10**5):
        x = np.linspace(0, 10, n)
        gaps.append(I.nml_complexity_polynomial(x, d, 1.0) - I.nml_complexity_asymptotic(x, d, 1.0))
    ok_gap &= all(b < a for a, b in zip(gaps, gaps[1:])) and gaps[-1] < 0.02 and min(gaps) > 0
    ratio = gaps[-2] / gaps[-1]                                 # 1/sqrt(n) scaling predicts sqrt(10) = 3.16
    gap_lines.append(f"degree {d}: {[round(g, 4) for g in gaps]} (ratio per 10x n: {ratio:.2f})")
report("3c 'parameters' closed form -> eq. 2.21 as n grows (R = sigma^2 = 1)", ok_gap, "; ".join(gap_lines))


# =============================================================== 4. raw-space Monte Carlo
def mc_shtarkov(x, degree, R, sigma, N, restrict, proposal, rng):
    """log2 int max_{mu allowed} N(y; mu, sigma^2 I) dy by importance sampling in R^n."""
    n = len(x)
    Q = basis(x, degree)
    r = sqrt(n * R)
    L = r + 8 * sigma
    tau = sqrt(R) + 2 * sigma
    out = []
    for _ in range(N // 250_000):
        M = 250_000
        if proposal == 'box':
            Y = rng.uniform(-L, L, size=(M, n))
            log_q = np.full(M, -n * np.log(2 * L))              # 1 / q = (2L)^n: the volume term
        else:
            Y = rng.standard_normal((M, n)) * tau
            log_q = -0.5 * n * np.log(2 * pi * tau**2) - np.sum(Y**2, axis=1) / (2 * tau**2)
        Z = Y @ Q                                               # (M, k) fit coordinates
        rho = np.linalg.norm(Z, axis=1)
        if restrict == 'parameters':
            Z = Z * np.minimum(1.0, r / np.maximum(rho, 1e-300))[:, None]   # nearest allowed fit
            inside = np.ones(M, dtype=bool)
        else:
            inside = rho <= r
        log_f = -0.5 * n * np.log(2 * pi * sigma**2) - np.sum((Y - Z @ Q.T) ** 2, axis=1) / (2 * sigma**2)
        out.append(np.where(inside, log_f - log_q, -np.inf))
    m, rel = log_mean_exp(np.concatenate(out))
    return m / LN2, rel / LN2


rng = np.random.default_rng(3)
lines, ok_mc = [], True
for n, d, R, sg in ((3, 0, 4.0, 1.0), (3, 0, 0.3, 1.0), (4, 1, 2.0, 1.0), (5, 2, 3.0, 0.7)):
    x = np.linspace(0, 10, n)
    for restrict in ('parameters', 'data'):
        exact = I.nml_complexity_polynomial(x, d, R, sg, restrict=restrict)
        for proposal in ('box', 'gaussian'):
            est, se = mc_shtarkov(x, d, R, sg, 1_000_000, restrict, proposal, rng)
            z = (est - exact) / se
            ok_mc &= abs(z) < 4
            lines.append(f"n={n} deg={d} R={R} {restrict[:4]}/{proposal[:3]}: {est:.3f}+-{se:.3f} vs {exact:.3f} (z={z:+.1f})")
report("4a raw-space Monte Carlo (own code) vs closed form, |z| < 4", ok_mc, "\n      " + "\n      ".join(lines))
x3 = np.linspace(0, 10, 3)
lines, ok_imc = [], True
for restrict in ('parameters', 'data'):
    for proposal in ('box', 'gaussian'):
        est, se = I.nml_complexity_monte_carlo(x3, 0, 4.0, 1.0, n_samples=1_000_000, restrict=restrict,
                                               proposal=proposal, seed=7)
        exact = I.nml_complexity_polynomial(x3, 0, 4.0, 1.0, restrict=restrict)
        ok_imc &= abs(est - exact) < 4 * se
        lines.append(f"{restrict}/{proposal}: {est:.3f}+-{se:.3f} vs {exact:.3f}")
report("4b implementation.nml_complexity_monte_carlo vs closed form (n=3, degree 0, R=4)", ok_imc, "; ".join(lines))


# =============================================================== 5. sigma unknown (Rissanen)
def log_c_rissanen(n, k, tau0, R, tau_max=np.inf):
    """ln int over {tau0 <= sigma_hat^2 <= tau_max, R_hat <= R} of f(y | beta_hat, sigma_hat^2) dy."""
    base = log(4 / k**2) + (k / 2) * log(R / tau0) + (n / 2) * log(n / (2 * np.e)) - lgamma(k / 2) - lgamma((n - k) / 2)
    return base + (0.0 if np.isinf(tau_max) else log(1 - (tau0 / tau_max) ** (k / 2)))


def mc_unknown_sigma(x, degree, region, N, rng, tau_q, renormalise=False):
    """Raw-space Monte Carlo of int_region f(y | beta_hat, sigma_hat^2) [/ C(sigma_hat^2, R_hat)] dy."""
    n, k = len(x), degree + 1
    Q = basis(x, degree)
    out = []
    for _ in range(N // 250_000):
        M = 250_000
        Y = rng.standard_normal((M, n)) * tau_q
        log_q = -0.5 * n * np.log(2 * pi * tau_q**2) - np.sum(Y**2, axis=1) / (2 * tau_q**2)
        Z = Y @ Q
        s2 = np.sum((Y - Z @ Q.T) ** 2, axis=1) / n             # sigma_hat^2 per dataset
        Rh = np.sum(Z**2, axis=1) / n                           # R_hat = mean squared fit
        log_f = -0.5 * n * np.log(2 * pi * np.e * s2)           # maximised likelihood (beta and sigma fitted)
        inside = region(s2, Rh)
        if renormalise:                                         # divide by the first-level normaliser
            log_f = log_f - (np.log(4 / k**2) + (k / 2) * np.log(Rh / s2) + (n / 2) * np.log(n / (2 * np.e))
                             - lgamma(k / 2) - lgamma((n - k) / 2))
        out.append(np.where(inside, log_f - log_q, -np.inf))
    return log_mean_exp(np.concatenate(out))


rng = np.random.default_rng(4)
lines, ok_c = [], True
for n, d, tau0, tmax, R, tq in ((4, 0, 0.5, 3.0, 2.0, 1.6), (5, 1, 0.3, 2.0, 1.5, 1.3), (6, 2, 0.4, 2.5, 3.0, 1.8)):
    k = d + 1
    exact = log_c_rissanen(n, k, tau0, R, tmax) / LN2
    est, rel = mc_unknown_sigma(np.linspace(0, 10, n), d, lambda s2, Rh: (s2 >= tau0) & (s2 <= tmax) & (Rh <= R),
                                2_000_000, rng, tq)
    ok_c &= abs(est / LN2 - exact) < 4 * rel / LN2
    impl = I.nml_complexity_unknown_sigma(n, k, tau0, R) + np.log2(1 - (tau0 / tmax) ** (k / 2))
    ok_c &= abs(impl - exact) < 1e-12
    lines.append(f"n={n} k={k}: closed {exact:.4f}, MC {est / LN2:.4f} +- {rel / LN2:.4f}")
report("5a Rissanen's C(tau0, R) (sigma fitted) vs Monte Carlo; implementation matches the formula", ok_c,
       "; ".join(lines))
lines, ok_c2 = [], True
for n, d, t1, t2, R1, R2, tq in ((4, 0, 0.5, 2.0, 1.0, 4.0, 1.6), (5, 1, 0.3, 1.5, 0.5, 3.0, 1.4)):
    k = d + 1
    exact = (k**2 / 4) * log(R2 / R1) * log(t2 / t1)
    est, rel = mc_unknown_sigma(np.linspace(0, 10, n), d,
                                lambda s2, Rh: (s2 >= t1) & (s2 <= t2) & (Rh >= R1) & (Rh <= R2),
                                2_000_000, rng, tq, renormalise=True)
    ok_c2 &= abs(np.exp(est) / exact - 1) < 4 * rel
    lines.append(f"n={n} k={k}: (k^2/4) ln(R2/R1) ln(t2/t1) = {exact:.4f}, MC {np.exp(est):.4f} (rel se {rel:.4f})")
report("5b RNML second normaliser vs Monte Carlo", ok_c2, "; ".join(lines))
np.random.seed(42)
coef = np.random.randn(3) * 2
y42 = np.poly1d(coef)(x50) + np.random.randn(50) * 1.5          # train_minimal's default data
dec_err = 0.0
for d in range(11):
    total, fit, comp = I.nml_mdl_polynomial(x50, y42, d)
    k, n = d + 1, 50
    yh = np.polyval(np.polyfit(x50, y42, d), x50)
    s2, Rh = np.mean((y42 - yh) ** 2), np.mean(yh**2)
    rissanen = ((n - k) / 2 * log(s2) + k / 2 * log(Rh) - lgamma(k / 2) - lgamma((n - k) / 2) + n / 2 * log(n * pi)) / LN2
    parts = 0.5 * n * np.log2(2 * pi * np.e * s2) + log_c_rissanen(n, k, s2, Rh) / LN2 + np.log2(k**2 / 4)
    dec_err = max(dec_err, abs(total - rissanen), abs(total - parts), abs(fit + comp - total))
report("5c nml_mdl_polynomial = fit + log2 C(sigma_hat^2, R_hat) + log2(k^2/4) = Rissanen's formula",
       dec_err < 1e-9, f"max |difference| over degrees 0-10 on train_minimal's data: {dec_err:.1e}")

# =============================================================== 6. selection rates
def two_part(x, y, d, precision=32):                           # notebook / train_minimal crude code
    n = len(x)
    r = y - np.polyval(np.polyfit(x, y, d), x)
    s = max(np.std(r), 1e-10)
    return (I.universal_code_integers(d + 1) + (d + 1) * (precision + 1)
            + 0.5 * n * np.log2(2 * pi * s**2) + np.sum(r**2) / (2 * s**2 * LN2) + precision + 10)


rates = {}
for setting in ('notebook', 'train_minimal'):
    counts = {m: np.zeros(11, int) for m in ('NML', 'two-part', 'prequential', 'BIC', 'AIC')}
    for seed in range(100):
        np.random.seed(seed)
        if setting == 'notebook':
            y = 3 + 2 * x50 - 0.5 * x50**2 + np.random.randn(50) * 2.0
        else:
            y = np.poly1d(np.random.randn(3) * 2)(x50) + np.random.randn(50) * 1.5
        counts['NML'][I.nml_model_selection(x50, y, 10)['best_degree']] += 1
        counts['two-part'][int(np.argmin([two_part(x50, y, d) for d in range(11)]))] += 1
        counts['prequential'][I.prequential_model_selection(x50, y, 10)['best_degree']] += 1
        counts['BIC'][int(np.argmin([I.bic_score(x50, y, d) for d in range(11)]))] += 1
        counts['AIC'][int(np.argmin([I.aic_score(x50, y, d) for d in range(11)]))] += 1
    rates[setting] = counts
    print(f"      {setting} data, true degree 2, 100 seeds: " + "; ".join(
        f"{m} {c[2]}% (under {c[:2].sum()}, over {c[3:].sum()})" for m, c in counts.items()))
report("6 NML (RNML) picks the true degree in >= 90% of seeds on both datasets, and at least as often as BIC",
       all(rates[s]['NML'][2] >= 90 and rates[s]['NML'][2] >= rates[s]['BIC'][2] for s in rates),
       f"NML {rates['notebook']['NML'][2]}% / {rates['train_minimal']['NML'][2]}%, BIC {rates['notebook']['BIC'][2]}% / "
       f"{rates['train_minimal']['BIC'][2]}%, AIC {rates['notebook']['AIC'][2]}% / {rates['train_minimal']['AIC'][2]}%")

n_pass = sum(p for _, p in results)
print(f"\n{n_pass}/{len(results)} checks passed in {time.time() - T0:.0f} s")
print("PASS" if n_pass == len(results) else "FAIL")
