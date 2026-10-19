"""Day 04 worked example: bits-back cost of one Gaussian weight (paper eqs. 6-9, Sec. 5.2).

Prior P = N(0, 1), posterior Q = N(0.5, 0.1^2), quantisation width tau = 2^-10.
  C(w) = -log2 tau - log2 P(w)   bits to send w under the prior         (eq. 6)
  R(w) = -log2 tau - log2 Q(w)   random bits used to pick w, refunded    (eq. 7)
  <C - R>_Q = KL(Q || P)                                                 (eq. 8, closed form eq. 9)
Checks the closed forms against Monte Carlo and prints the table used on the page.
"""
import numpy as np

LN2 = np.log(2)
tau_bits = 10.0                                   # -log2(tau) for tau = 2**-10


def kl_bits(mq, sq, mp=0.0, sp=1.0):
    """eq. 9 in nats, converted to bits."""
    return (np.log(sp / sq) + (sq**2 - sp**2 + (mp - mq)**2) / (2 * sp**2)) / LN2


rng = np.random.default_rng(0)
print(f"{'sigma_q':>8} {'E[C] bits':>10} {'E[R] bits':>10} {'net bits':>9} {'KL bits':>8} {'MC net':>8}")
for sq in (0.3, 0.1, 0.01):
    mq = 0.5
    EC = tau_bits + 0.5 * np.log2(2 * np.pi) + (mq**2 + sq**2) / (2 * LN2)   # E_Q[-log2 P(w)]
    ER = tau_bits + 0.5 * np.log2(2 * np.pi * np.e * sq**2)                  # entropy of Q in bits
    w = rng.normal(mq, sq, 2_000_000)
    C = tau_bits + 0.5 * np.log2(2 * np.pi) + w**2 / (2 * LN2)
    R = tau_bits + 0.5 * np.log2(2 * np.pi * sq**2) + (w - mq)**2 / (2 * sq**2 * LN2)
    print(f"{sq:>8} {EC:>10.4f} {ER:>10.4f} {EC - ER:>9.4f} {kl_bits(mq, sq):>8.4f} {(C - R).mean():>8.4f}")
    assert abs((EC - ER) - kl_bits(mq, sq)) < 1e-12
    assert abs((C - R).mean() - kl_bits(mq, sq)) < 0.01
# Without bits back, sending w at precision tau costs E[C] ~ 11.5 bits regardless of sigma_q.
print("PASS")
