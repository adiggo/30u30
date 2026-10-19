"""Day 05 reference checks (Grunwald 2004 tutorial, arXiv math/0406077, vs implementation.py and
train_minimal.py).

1. Two-part code: L(D|H) equals -sum log2 N(r_i; 0, sigma_ML^2) (density at unit precision,
   tutorial Sec. 2.2.2 'Continuous Sample Spaces'); decompose train_minimal's 268.45 bits at
   degree 2; put AIC/BIC on the same bit scale (BIC/(2 ln 2) = -log2 L + (k/2) log2 n, eq. 2.42).
2. Universal integer code L*(n) (Rissanen log-star, c0 = 2.865064): independent re-implementation
   and Kraft inequality sum_n 2^-L(n) <= 1 (Fig. 2.1/2.2); same for the tutorial's 2 log k + 1
   code (Example 2.4).
3. Prequential step code length vs scipy.stats.norm.logpdf (eq. 2.29), and the effect of the
   per-degree start index (fixed by the 2026-10-18 patch).
4. Bernoulli NML parametric complexity: exact Shtarkov sum (eq. 2.17) vs the asymptotic
   expansion (eq. 2.21) with the correct Fisher information I(theta) = 1/(theta(1-theta)), and vs
   the constant printed in the tutorial's Example 2.17 (which uses theta(1-theta)).
5. implementation.nml_complexity_polynomial. Before 2026-10-18 it was a Monte Carlo average without
   the box volume (about -380 bits; it shifted by -n log2 10 when the box widened 10x). It now
   computes COMP of the restricted model (sigma known, mean squared fit <= R) in closed form; here
   we check COMP >= 0 and monotone in degree. The full checks (pre-fix estimator, eq. 2.21, Monte
   Carlo, sigma-unknown RNML, selection rates) are in validation/day05_nml_check.py.
"""
import os
import sys
import warnings
import numpy as np
from scipy.stats import norm
from scipy.special import gammaln

warnings.simplefilter('ignore')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '05_MDL_Principle')
sys.path.insert(0, DAY)
import implementation as I  # noqa: E402
import train_minimal as T   # noqa: E402  (functions only; main() not run)

LN2 = np.log(2)
ok = True

# ------------------------------------------------------------------ 1. two-part code + units
np.random.seed(42)                                  # identical to run_experiment(seed=42)
x = np.linspace(0, 10, 50)
coef = np.random.randn(3) * 2
y = np.poly1d(coef)(x) + np.random.randn(50) * 1.5
r = y - np.polyval(np.polyfit(x, y, 2), x)          # residuals of the degree-2 ML fit
s_ml = np.sqrt(np.mean(r**2))
nll_bits = -norm.logpdf(r, 0, s_ml).sum() / LN2     # -log2 density of the residuals
total, model_cost, data_cost = T.two_part_mdl(x, y, 2)
d = abs((data_cost - 42) - nll_bits)                # data_cost = NLL + 42-bit sigma code
print(f"[2-part] train_minimal degree 2: total {total:.2f} = model {model_cost:.2f} "
      f"(L*(3)={T.universal_code_length(3):.3f} + 3x33) + NLL {nll_bits:.2f} + sigma 42; |d|={d:.1e}")
ok &= d < 1e-9 and abs(total - 268.45) < 0.01
n, k = 50, 4                                         # 3 coefficients + variance
bic, aic = T.bic(x, y, 2), T.aic(x, y, 2)
print(f"[units] AIC={aic:.2f}, BIC={bic:.2f} are on the -2 ln L scale (nats x 2).")
print(f"[units] BIC/(2 ln 2) = {bic / (2 * LN2):.2f} bits = -log2 L {nll_bits:.2f} + (k/2)log2 n "
      f"{k / 2 * np.log2(n):.2f}; two-part MDL charges {model_cost + 42:.2f} bits for the same model")
ok &= abs(bic / (2 * LN2) - (nll_bits + k / 2 * np.log2(n))) < 1e-9
print(f"[units] per extra coefficient: two-part code 33 bits vs BIC/eq.2.42 (1/2)log2 n = "
      f"{0.5 * np.log2(n):.2f} bits; AIC 1/ln2 = {1 / LN2:.2f} bits")

# ------------------------------------------------------------------ 2. universal integer code


def logstar_len(m, c0=2.865064):
    """Rissanen's L*(m) = log2 c0 + log2 m + log2 log2 m + ... (positive terms only)."""
    L, t = np.log2(c0), np.log2(m) if m > 1 else 0.0
    while t > 0:
        L += t
        t = np.log2(t) if t > 1 else 0.0
    return L


ms = np.arange(1, 200_001)
mx = max(abs(I.universal_code_integers(int(m)) - logstar_len(m)) for m in ms[:5000])
kraft = sum(2.0 ** -I.universal_code_integers(int(m)) for m in ms)
kraft_simple = np.sum(2.0 ** -(2 * np.log2(ms.astype(float)) + 1))   # Example 2.4 code
print(f"[L*] |code - independent| max over n<=5000: {mx:.1e}; Kraft sum n<=2e5: {kraft:.4f} (<=1); "
      f"2log k+1 code: {kraft_simple:.4f} (<=1, tends to pi^2/12={np.pi**2 / 12:.4f})")
ok &= mx < 1e-9 and kraft <= 1 and kraft_simple <= 1

# ------------------------------------------------------------------ 3. prequential step formula
xs, ys = x[:12], y[:12]
cf = np.polyfit(xs, ys, 2)
sig = np.std(ys - np.polyval(cf, xs))
step_code = 0.5 * np.log2(2 * np.pi * sig**2) + (y[12] - np.polyval(cf, x[12]))**2 / (2 * sig**2 * LN2)
step_ref = -norm.logpdf(y[12], np.polyval(cf, x[12]), sig) / LN2
print(f"[preq] one-step code length {step_code:.6f} vs -log2 N(...) {step_ref:.6f}")
ok &= abs(step_code - step_ref) < 1e-9
shipped = {dg: I.prequential_mdl_polynomial(x, y, dg) for dg in range(11)}          # start=dg+2
common = I.prequential_model_selection(x, y, max_degree=10)                         # patched
print(f"[preq] per-degree start picks {min(shipped, key=shipped.get)} "
      f"(degree-2 total {shipped[2]:.0f} bits, from 46 coded points); common start picks "
      f"{common['best_degree']} (degree-2 total {common['scores'][2]:.1f} bits, 38 points)")
ok &= common['best_degree'] == 2

# ------------------------------------------------------------------ 4. Bernoulli NML


def bern_comp_exact(nn):
    """log2 sum_k C(n,k) (k/n)^k ((n-k)/n)^(n-k), in log space (gammaln = log Gamma)."""
    kk = np.arange(nn + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = np.where(kk > 0, kk * np.log(kk / nn), 0.0)
        t2 = np.where(kk < nn, (nn - kk) * np.log((nn - kk) / nn), 0.0)
    logs = gammaln(nn + 1) - gammaln(kk + 1) - gammaln(nn - kk + 1) + t1 + t2
    mxl = logs.max()
    return (mxl + np.log(np.exp(logs - mxl).sum())) / LN2


print(f"[NML] {'n':>6} {'exact COMP':>11} {'eq.2.21':>8} {'Ex.2.17':>8}")
for nn in (10, 100, 1000, 10000):
    ex = bern_comp_exact(nn)
    asym = 0.5 * np.log2(nn / (2 * np.pi)) + np.log2(np.pi)       # integral of sqrt(I) = pi
    tut = 0.5 * np.log2(nn) - 2.674251935                         # tutorial's printed constant
    print(f"[NML] {nn:>6} {ex:>11.4f} {asym:>8.4f} {tut:>8.4f}")
ok &= abs(bern_comp_exact(10000) - (0.5 * np.log2(10000 / (2 * np.pi)) + np.log2(np.pi))) < 0.01
print("[NML] correct constant: (1/2)log2(pi/2) = %.4f; Example 2.17 prints -2.6743 = that minus 3 "
      "(it integrates sqrt(theta(1-theta)) = pi/8 instead of 1/sqrt(theta(1-theta)) -> pi)"
      % (0.5 * np.log2(np.pi / 2)))

# ------------------------------------------------------------------ 5. code's NML complexity
# [PATCH 2026-10-18] The pre-fix estimator (characterised here in round 1) was replaced; its
# reproduction and the full checks of the new code live in validation/day05_nml_check.py.
vals = {dg: I.nml_complexity_polynomial(x, dg, R=100.0, sigma=1.0) for dg in range(11)}
print("[NML-code] nml_complexity_polynomial(x, d, R=100, sigma=1), d=0..10:",
      [round(v, 2) for v in vals.values()])
mono = all(vals[d + 1] >= vals[d] for d in range(10))
print(f"[NML-code] all >= 0: {all(v >= 0 for v in vals.values())}; non-decreasing in degree: {mono} "
      "(COMP = log int max_theta f(y|theta) dy >= log int f(y|theta_0) dy = 0)")
ok &= all(v >= 0 for v in vals.values()) and mono

print("PASS" if ok else "FAIL")
