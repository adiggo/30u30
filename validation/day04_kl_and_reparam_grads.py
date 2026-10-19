"""Day 04 reference checks (Hinton & van Camp 1993 / implementation.py).

1. BayesianLinear.kl_divergence() vs torch.distributions KL(N(mu,sigma^2) || N(0,1)),
   vs the paper's eq. 9 with mu_p=0, sigma_p=1, and vs a Monte-Carlo estimate.
   Also checks that the paper's eq. 9 (general prior mean/std) equals torch's KL(Q||P).
2. Hand-written reparameterisation gradients (MDLNetwork.backward + the KL gradients that
   update_weights() combines) vs torch.autograd, using the *same* noise epsilon.
3. The same gradients vs central finite differences on a sample of coordinates.

Run:  OMP_NUM_THREADS=1 python validation/day04_kl_and_reparam_grads.py
"""
import os
import sys
import numpy as np
import torch
from torch.distributions import Normal, kl_divergence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '04_Minimizing_Description_Length')
sys.path.insert(0, DAY)
from implementation import BayesianLinear, MDLNetwork, softplus  # noqa: E402

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)
np.random.seed(0)
ok = True

# ---------------------------------------------------------------- 1. KL closed form
layer = BayesianLinear(7, 5)
layer.w_mu = rng.normal(0, 0.8, (7, 5))           # (in, out) means
layer.w_rho = rng.normal(-1.0, 1.5, (7, 5))       # sigma = softplus(rho) spans ~0.02..3
layer.b_mu = rng.normal(0, 0.8, 5)
layer.b_rho = rng.normal(-1.0, 1.5, 5)
layer.sample_weights()                            # sets w_sigma/b_sigma (kl_divergence reads them)
kl_code = layer.kl_divergence()

# np.concatenate of the flattened (7,5) block (.ravel() -> 35,) and the (5,) bias -> one (40,) vector
mu = np.concatenate([layer.w_mu.ravel(), layer.b_mu])
sig = np.concatenate([softplus(layer.w_rho).ravel(), softplus(layer.b_rho)])
kl_torch = kl_divergence(Normal(torch.tensor(mu), torch.tensor(sig)), Normal(0.0, 1.0)).sum().item()


def eq9(mu_q, s_q, mu_p, s_p):
    """Paper eq. 9: G(P,Q) = log(s_p/s_q) + (s_q^2 - s_p^2 + (mu_p-mu_q)^2) / (2 s_p^2)."""
    return np.log(s_p / s_q) + (s_q**2 - s_p**2 + (mu_p - mu_q)**2) / (2 * s_p**2)


kl_eq9 = eq9(mu, sig, 0.0, 1.0).sum()
# Monte Carlo: E_q[log q(w) - log p(w)], 400k draws per weight (vectorised: (400000, 40) array)
eps = rng.standard_normal((400_000, mu.size))
w = mu + sig * eps                                  # broadcasting (40,) over the 400k rows
logq = -0.5 * np.log(2 * np.pi * sig**2) - 0.5 * eps**2
logp = -0.5 * np.log(2 * np.pi) - 0.5 * w**2
per_draw = (logq - logp).sum(axis=1)                # axis=1: sum over the 40 weights per draw
kl_mc, kl_se = per_draw.mean(), per_draw.std() / np.sqrt(len(per_draw))

d1 = abs(kl_code - kl_torch)
d2 = abs(kl_eq9 - kl_torch)
z = abs(kl_mc - kl_torch) / kl_se
print(f"[KL] code={kl_code:.10f} torch={kl_torch:.10f} eq9={kl_eq9:.10f} MC={kl_mc:.4f}+-{kl_se:.4f}")
print(f"[KL] |code-torch|={d1:.2e}  |eq9-torch|={d2:.2e}  MC z-score={z:.2f}")
ok &= d1 < 1e-5 and d2 < 1e-10 and z < 4

# eq. 9 with a non-standard prior (the paper learns mu_p, sigma_p in Sec. 6)
mq, sq = rng.normal(0, 1, 50), np.exp(rng.normal(-1, 0.7, 50))
mp, sp = rng.normal(0, 1, 50), np.exp(rng.normal(0, 0.5, 50))
d3 = np.max(np.abs(eq9(mq, sq, mp, sp) -
                   kl_divergence(Normal(torch.tensor(mq), torch.tensor(sq)),
                                 Normal(torch.tensor(mp), torch.tensor(sp))).numpy()))
print(f"[KL] eq.9 with learned prior (mu_p, sigma_p) vs torch KL(Q||P): max |d| = {d3:.2e}")
ok &= d3 < 1e-12

# ---------------------------------------------------------------- 2. gradients vs autograd
N, D_IN, H = 12, 3, 6
X = rng.normal(0, 1, (N, D_IN))
y = rng.normal(0, 1, (N, 1))
net = MDLNetwork(D_IN, H, 1)
for L in (net.layer1, net.layer2):
    L.w_mu = rng.normal(0, 0.7, L.w_mu.shape)
    L.w_rho = rng.normal(-1.5, 0.8, L.w_rho.shape)
    L.b_mu = rng.normal(0, 0.3, L.b_mu.shape)
    L.b_rho = rng.normal(-1.5, 0.8, L.b_rho.shape)
kl_weight = 0.1
c = kl_weight / N                                   # train_minimal.py: update_weights(lr, beta/N)
preds = net.forward(X)                              # samples epsilon inside each layer
net.backward(2 * (preds - y) / N)                   # d(mean sq err)/d(pred), as in train_minimal.py
code_grad = {}
for name, L in (('l1', net.layer1), ('l2', net.layer2)):
    for p in ('w_mu', 'b_mu', 'w_rho', 'b_rho'):
        code_grad[f'{name}.{p}'] = L.grads['nll_' + p] + c * L.grads['kl_' + p]

# torch replica with the SAME epsilons
T = {}
for name, L in (('l1', net.layer1), ('l2', net.layer2)):
    for p in ('w_mu', 'b_mu', 'w_rho', 'b_rho'):
        T[f'{name}.{p}'] = torch.tensor(getattr(L, p), requires_grad=True)
E = {'l1': (net.layer1.w_epsilon, net.layer1.b_epsilon), 'l2': (net.layer2.w_epsilon, net.layer2.b_epsilon)}


def layer_t(x, name):
    sw, sb = torch.nn.functional.softplus(T[f'{name}.w_rho']), torch.nn.functional.softplus(T[f'{name}.b_rho'])
    w_ = T[f'{name}.w_mu'] + sw * torch.tensor(E[name][0])
    b_ = T[f'{name}.b_mu'] + sb * torch.tensor(E[name][1])
    kl = 0.5 * (sw**2 + T[f'{name}.w_mu']**2 - 1 - torch.log(sw**2)).sum() + \
        0.5 * (sb**2 + T[f'{name}.b_mu']**2 - 1 - torch.log(sb**2)).sum()
    return x @ w_ + b_, kl


h, kl1 = layer_t(torch.tensor(X), 'l1')
out, kl2 = layer_t(torch.relu(h), 'l2')
loss = ((out - torch.tensor(y))**2).mean() + c * (kl1 + kl2)
loss.backward()
max_ad = 0.0
for k, t in T.items():
    d = np.max(np.abs(code_grad[k] - t.grad.numpy()))
    max_ad = max(max_ad, d)
print(f"[grad] reparameterisation grads vs torch.autograd (same eps): max |d| = {max_ad:.2e}")
ok &= max_ad < 1e-6

# ---------------------------------------------------------------- 3. central finite differences


def loss_np(params):
    """Loss with the epsilons frozen (reparameterised objective for this draw)."""
    tot_kl, hcur = 0.0, X
    for name, act in (('l1', True), ('l2', False)):
        sw, sb = softplus(params[f'{name}.w_rho']), softplus(params[f'{name}.b_rho'])
        w_ = params[f'{name}.w_mu'] + sw * E[name][0]
        b_ = params[f'{name}.b_mu'] + sb * E[name][1]
        tot_kl += 0.5 * np.sum(sw**2 + params[f'{name}.w_mu']**2 - 1 - np.log(sw**2))
        tot_kl += 0.5 * np.sum(sb**2 + params[f'{name}.b_mu']**2 - 1 - np.log(sb**2))
        hcur = hcur @ w_ + b_
        if act:
            hcur = np.maximum(hcur, 0)
    return np.mean((hcur - y)**2) + c * tot_kl


base = {k: t.detach().numpy().copy() for k, t in T.items()}
max_fd, h_fd = 0.0, 1e-6
for k in base:
    flat_idx = rng.choice(base[k].size, size=min(6, base[k].size), replace=False)
    for fi in flat_idx:
        idx = np.unravel_index(fi, base[k].shape)  # flat index -> tuple index for any shape
        pp = {kk: v.copy() for kk, v in base.items()}
        pm = {kk: v.copy() for kk, v in base.items()}
        pp[k][idx] += h_fd
        pm[k][idx] -= h_fd
        fd = (loss_np(pp) - loss_np(pm)) / (2 * h_fd)
        max_fd = max(max_fd, abs(fd - code_grad[k][idx]))
print(f"[grad] vs central finite differences (h=1e-6, 6 coords/param): max |d| = {max_fd:.2e}")
ok &= max_fd < 1e-6

print("PASS" if ok else "FAIL")
