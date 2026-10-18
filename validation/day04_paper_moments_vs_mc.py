"""Day 04: the paper's deterministic objective (Sec. 5.3, eqs. 10-12) vs the code's Monte Carlo.

Hinton & van Camp do NOT sample weights. For a net with one hidden layer and linear outputs they
propagate means and variances: each hidden unit's total input is exactly Gaussian (fixed inputs x
times independent Gaussian weights), its output mean/variance came from a lookup table (logistic
units), and for a linear output unit
    ybar   = sum_h ybar_h * wbar_h                                   (eq. 10)
    V_y    = sum_h [wbar_h^2 V_h + ybar_h^2 V_w + V_h V_w]           (eq. 11)
    <E>    = (d - ybar)^2 + V_y                                      (eq. 12)
The repo's MDLNetwork has ReLU hidden units, for which the table is not needed: a rectified
Gaussian has closed-form moments. This script checks that
  (a) the code's sampled squared error, averaged over many weight draws, equals eq. 12, and
  (b) the code's reparameterisation gradients, averaged over draws, equal d<E>/d(mu, rho)
      computed by autograd on the closed form -- i.e. the code optimises the paper's objective
      with an unbiased stochastic gradient instead of the paper's exact one.
"""
import os
import sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '04_Minimizing_Description_Length')
sys.path.insert(0, DAY)
from implementation import MDLNetwork  # noqa: E402

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(1)
np.random.seed(1)
H = 8
x = np.array([[-2.5], [-1.2], [0.0], [0.7], [2.2]])      # (5, 1): five fixed inputs
d = np.sin(x)                                             # (5, 1): targets
net = MDLNetwork(1, H, 1)
for L in (net.layer1, net.layer2):
    L.w_mu = rng.normal(0, 0.8, L.w_mu.shape)
    L.w_rho = rng.normal(-1.2, 0.5, L.w_rho.shape)
    L.b_mu = rng.normal(0, 0.3, L.b_mu.shape)
    L.b_rho = rng.normal(-1.2, 0.5, L.b_rho.shape)

P = {f'{n}.{p}': torch.tensor(getattr(L, p), requires_grad=True)
     for n, L in (('l1', net.layer1), ('l2', net.layer2)) for p in ('w_mu', 'b_mu', 'w_rho', 'b_rho')}
SP = torch.nn.functional.softplus
Phi = lambda t: 0.5 * (1 + torch.erf(t / np.sqrt(2)))                 # standard normal CDF
phi = lambda t: torch.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)           # standard normal pdf


def expected_sq_err():
    """Closed form of E_q[(d - y)^2] per input (paper eqs. 10-12, ReLU moments)."""
    xt = torch.tensor(x)
    # hidden pre-activation a = x W1 + b1 is Gaussian: mean m (5,H), variance v (5,H)
    m = xt @ P['l1.w_mu'] + P['l1.b_mu']
    v = (xt**2) @ SP(P['l1.w_rho'])**2 + SP(P['l1.b_rho'])**2
    s = torch.sqrt(v)
    a = m / s
    Er = m * Phi(a) + s * phi(a)                    # E[relu(a)]
    Er2 = (m**2 + v) * Phi(a) + m * s * phi(a)      # E[relu(a)^2]
    Vr = Er2 - Er**2                                # V_h in eq. 11
    w2m, w2v = P['l2.w_mu'][:, 0], SP(P['l2.w_rho'][:, 0])**2
    ybar = Er @ w2m + P['l2.b_mu'][0]                                   # eq. 10 (+ bias)
    Vy = (w2m**2 * Vr + Er**2 * w2v + Vr * w2v).sum(1) + SP(P['l2.b_rho'][0])**2   # eq. 11 (+ bias)
    return (torch.tensor(d[:, 0]) - ybar)**2 + Vy                       # eq. 12, shape (5,)


E_closed = expected_sq_err()
# (a) Monte Carlo with the repo's own forward pass: one weight draw per call, shared by the 5 inputs
S = 60_000
sq = np.empty((S, 5))
for s_ in range(S):
    sq[s_] = (d - net.forward(x))[:, 0]**2
mc, se = sq.mean(0), sq.std(0) / np.sqrt(S)          # axis 0 = over draws -> per-input mean, s.e.
za = np.abs(mc - E_closed.detach().numpy()) / se
print("[eq.12] closed form :", np.round(E_closed.detach().numpy(), 5))
print("[eq.12] code MC mean:", np.round(mc, 5), " s.e.", np.round(se, 5))
print(f"[eq.12] max z-score = {za.max():.2f}, max |d| = {np.abs(mc - E_closed.detach().numpy()).max():.2e}")

# (b) gradients of the data term mean_i <E_i> w.r.t. (mu, rho) vs averaged reparam gradients
E_closed.mean().backward()
G = 40_000
acc = {k: np.zeros_like(v.detach().numpy()) for k, v in P.items()}
acc2 = {k: np.zeros_like(v.detach().numpy()) for k, v in P.items()}
for _ in range(G):
    pred = net.forward(x)
    net.backward(2 * (pred - d) / len(x))            # gradient of mean squared error, as in train
    for n, L in (('l1', net.layer1), ('l2', net.layer2)):
        for p in ('w_mu', 'b_mu', 'w_rho', 'b_rho'):
            g = L.grads['nll_' + p]
            acc[f'{n}.{p}'] += g
            acc2[f'{n}.{p}'] += g**2
zmax, relmax = 0.0, 0.0
for k in P:
    mean = acc[k] / G
    sd = np.sqrt(np.maximum(acc2[k] / G - mean**2, 1e-30))
    zk = np.abs(mean - P[k].grad.numpy()) / (sd / np.sqrt(G))
    zmax = max(zmax, zk.max())
    big = np.abs(P[k].grad.numpy()) > 1e-3
    if big.any():
        relmax = max(relmax, (np.abs(mean - P[k].grad.numpy())[big] / np.abs(P[k].grad.numpy())[big]).max())
print(f"[grad] mean of {G} reparam grads vs autograd of closed form: max z = {zmax:.2f}, "
      f"max rel err (|g|>1e-3) = {relmax:.3f}")
print("PASS" if za.max() < 4.5 and zmax < 4.5 else "FAIL")
