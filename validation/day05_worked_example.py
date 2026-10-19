"""Day 05 page worked example: code lengths (bits) for n=10 coin flips with 2 heads, and n=1000.
Literal, ML-in-hindsight, crude two-part (count then data), NML (eq. 2.18), Bayes-uniform (Laplace).
"""
import numpy as np
from math import comb, log2, pi


def comp(n):
    """COMP_n of the Bernoulli model, eq. 2.17: log2 sum_k C(n,k) (k/n)^k (1-k/n)^(n-k)."""
    tot = sum(comb(n, k) * (k / n) ** k * (1 - k / n) ** (n - k) for k in range(n + 1))  # 0**0 == 1
    return log2(tot)


n, k = 10, 2
ml = -(k * log2(k / n) + (n - k) * log2(1 - k / n))
two_part = log2(n + 1) + ml
nml = ml + comp(n)
bayes = log2((n + 1) * comb(n, k))
print(f"literal {n}  ML {ml:.4f}  two-part {two_part:.4f}  COMP10 {comp(n):.4f}  NML {nml:.4f}  "
      f"Bayes-uniform {bayes:.4f} (regret {bayes - ml:.4f})")
print(f"all-tails: Bayes-uniform {log2(n + 1):.4f} vs NML {comp(n):.4f}")
# NML regret is the same for every sequence (eq. 2.19); Bayes-uniform's max regret is at k=0 or n
regs = [log2((n + 1) * comb(n, j)) + (j * log2(j / n) if j else 0) + ((n - j) * log2(1 - j / n) if j < n else 0)
        for j in range(n + 1)]
print(f"Bayes-uniform regret range over k: {min(regs):.4f}..{max(regs):.4f}; NML regret {comp(n):.4f} for all")
from scipy.special import gammaln  # noqa: E402
N = 1000
kk = np.arange(N + 1)
with np.errstate(divide='ignore', invalid='ignore'):
    lg = gammaln(N + 1) - gammaln(kk + 1) - gammaln(N - kk + 1) + \
        np.where(kk > 0, kk * np.log(kk / N), 0) + np.where(kk < N, (N - kk) * np.log(1 - kk / N), 0)
m = lg.max()
print(f"n=1000: COMP {(m + np.log(np.exp(lg - m).sum())) / np.log(2):.4f}, eq.2.21 "
      f"{0.5 * log2(N / (2 * pi)) + log2(pi):.4f}, crude count code {log2(N + 1):.4f}")
