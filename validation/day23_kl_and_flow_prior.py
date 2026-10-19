"""Day 23: the KL / prior terms of the VLAE loss.

1. loss_function's log q(z|x) equals torch.distributions.Normal(mu, std).log_prob(z).sum().
2. use_flow=False: the single-sample KL estimate log q(z|x) - log p(z) inside loss_function,
   averaged over many reparameterised draws, equals the closed form
   KL(N(mu, diag sigma^2) || N(0, I)) = 0.5 * sum(sigma^2 + mu^2 - 1 - log sigma^2).
3. Flow prior (IAFBlock / MADE): Jacobian of each block from torch.autograd must be lower
   triangular with diagonal sigma, so log|det J| = sum log sigma (slogdet check).
   Run on the patched masks (np.tile) and on a reconstruction of the original masks (np.repeat).
4. Flow prior density in 2-D: integrate exp(log p(z)) on a grid. The patched code gives a
   normalised density (~1); the original formula log N(z) - sum log sigma does not.
5. The original formula's KL estimate can be pushed to -infinity (set the log-sigma bias very
   negative); the patched one cannot.
"""
import os
import sys
import numpy as np
import torch
from torch.distributions import Normal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '23_variational_lossy_autoencoder')
sys.path.insert(0, DAY)
import implementation as IM  # noqa: E402
from implementation import VLAE, IAFBlock, loss_function  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ok = True
LOG2PI = np.log(2 * np.pi)

# ---------------------------------------------------------------- 1 + 2. Gaussian prior
B, D = 4, 6
mu = torch.randn(B, D)
logvar = torch.randn(B, D) * 0.5 - 0.5
z = mu + torch.randn(B, D) * torch.exp(0.5 * logvar)
logits = torch.zeros(B, 1, 2, 2)
x = torch.zeros(B, 1, 2, 2)
recon = torch.nn.functional.binary_cross_entropy_with_logits(logits, x, reduction='sum')
kl_code = (loss_function(logits, x, mu, logvar, z, torch.zeros(B)) - recon).item()
lq = Normal(mu, torch.exp(0.5 * logvar)).log_prob(z).sum()
lp = Normal(0.0, 1.0).log_prob(z).sum()
d1 = abs(kl_code - (lq - lp).item())
print(f"[1] loss_function KL term vs torch log-probs (same z): |d| = {d1:.1e}")
ok &= d1 < 1e-10

S = 200_000
mu1, lv1 = mu[:1], logvar[:1]
zs = mu1 + torch.randn(S, D) * torch.exp(0.5 * lv1)            # (S, D): broadcast (1,D) over S rows
log_q = -0.5 * (LOG2PI + lv1 + (zs - mu1)**2 / torch.exp(lv1)).sum(1)   # the code's formula, per draw
log_p = -0.5 * (LOG2PI + zs**2).sum(1)
mc = (log_q - log_p)
closed = 0.5 * (torch.exp(lv1) + mu1**2 - 1 - lv1).sum().item()
z_ = abs(mc.mean().item() - closed) / (mc.std().item() / np.sqrt(S))
print(f"[2] MC KL {mc.mean():.4f} +- {mc.std() / np.sqrt(S):.4f} vs closed form {closed:.4f}  (z = {z_:.2f})")
ok &= z_ < 4


# ---------------------------------------------------------------- 3. IAF block Jacobian


def block_check(block, d):
    zz = torch.randn(d)
    J = torch.autograd.functional.jacobian(lambda v: block(v[None])[0][0], zz)   # (d, d)
    upper = torch.triu(J, diagonal=1).abs().max().item()     # entries that must be 0
    sign, logabs = torch.linalg.slogdet(J)
    _, ld = block(zz[None])
    return upper, abs(logabs.item() - ld.item())


d = 8
blk = IAFBlock(d, 64)
up, dd = block_check(blk, d)
print(f"[3] patched MADE masks: max |upper-triangular J| = {up:.1e}, |slogdet - sum log sigma| = {dd:.1e}")
ok &= up < 1e-12 and dd < 1e-9

orig = IAFBlock(d, 64)                                        # rebuild the original masks
lin = [m for m in orig.made.net if isinstance(m, IM.MaskedLinear)]
m_out = np.repeat(np.arange(d), 2)                            # original: [0,0,1,1,...]
orig.made.m[len(lin) - 1] = m_out
orig.made.m[-1] = np.arange(d)
mask = orig.made.m[len(lin) - 2][:, None] < m_out[None, :]
lin[-1].set_mask(torch.from_numpy(mask.astype(np.float64)))
up0, dd0 = block_check(orig, d)
print(f"[3] original masks (np.repeat): max |upper-triangular J| = {up0:.2e}, "
      f"|slogdet - sum log sigma| = {dd0:.2e}   <- violation")
ok &= up0 > 1e-6

# ---------------------------------------------------------------- 4. normalisation in 2-D
torch.manual_seed(3)
model = VLAE(input_dim=1, latent_dim=2, hidden_dim=8, n_layers=1, use_flow=True)


def flow_terms(zgrid):
    zf, ld = zgrid, torch.zeros(len(zgrid))
    for f in model.flow_steps:
        zf, l_ = f(zf)
        ld = ld + l_
    return zf, ld


class StubEncoder(torch.nn.Module):
    def forward(self, xx):
        return torch.zeros(len(xx), 2), torch.zeros(len(xx), 2)


# sigma = sigmoid(.) < 1 in each block, so p(z) is ~1/(sigma_1 sigma_2) wider than N(0,I):
# size the integration box from the smallest per-coordinate scale product seen on a coarse grid
with torch.no_grad():
    coarse = torch.cartesian_prod(torch.linspace(-30, 30, 61), torch.linspace(-30, 30, 61))
    zf_c, s_tot = coarse, torch.ones_like(coarse)
    for f in model.flow_steps:
        st = torch.sigmoid(torch.chunk(f.made(zf_c), 2, dim=1)[1])
        zf_c, s_tot = (zf_c - torch.chunk(f.made(zf_c), 2, dim=1)[0]) * st, s_tot * st
    half = float(12.0 / s_tot.min())
g = torch.linspace(-half, half, 2001)
cell = (g[1] - g[0]).item() ** 2
print(f"[4] min sigma product {s_tot.min():.3f} -> integrating over [-{half:.1f}, {half:.1f}]^2")
model.encoder = StubEncoder()
mass_p = mass_o = 0.0
with torch.no_grad():
    # repo code path: VLAE.forward returns the prior correction for a given z; drive it with grid
    # rows by stubbing the encoder/reparameterisation (1x1 'images' keep the decoder cheap)
    for row in g.split(100):                                  # chunks of 100 x 1601 grid points
        Zc = torch.cartesian_prod(row, g)
        model.reparameterize = lambda m_, lv_, Zc=Zc: Zc
        _, _, _, zr, adj = model(torch.zeros(len(Zc), 1, 1, 1))
        logp_patched = -0.5 * (LOG2PI + zr**2).sum(1) + adj    # = loss_function's log_p per point
        zf, ld = flow_terms(Zc)
        logp_orig = -0.5 * (LOG2PI + Zc**2).sum(1) - ld        # original loss_function formula
        mass_p += (torch.exp(logp_patched).sum() * cell).item()
        mass_o += (torch.exp(logp_orig).sum() * cell).item()
print(f"[4] integral of exp(log p(z)) over the box: patched {mass_p:.4f}, original formula {mass_o:.4f}")
ok &= abs(mass_p - 1) < 0.01 and abs(mass_o - 1) > 0.05

# ---------------------------------------------------------------- 5. exploit
torch.manual_seed(4)
m2 = VLAE(input_dim=1, latent_dim=8, n_layers=1, use_flow=True)
xb = (torch.rand(16, 1, 28, 28) > 0.5).double()
m2 = m2.double()
for f in m2.flow_steps:
    last = [m for m in f.made.net if isinstance(m, IM.MaskedLinear)][-1]
    last.bias.data[8:] = -20.0                                # log_sigma half -> sigma ~ 2e-9
with torch.no_grad():
    logits_, mu_, lv_, z2, adj2 = m2(xb)
    rec = torch.nn.functional.binary_cross_entropy_with_logits(logits_, xb, reduction='sum')
    kl_patched = (loss_function(logits_, xb, mu_, lv_, z2, adj2) - rec).item() / 16
    zf2, ld2 = z2, torch.zeros(16)
    for f in m2.flow_steps:
        zf2, l_ = f(zf2)
        ld2 = ld2 + l_
    lq2 = -0.5 * (LOG2PI + lv_ + (z2 - mu_)**2 / torch.exp(lv_)).sum(1)
    kl_orig = (lq2 - (-0.5 * (LOG2PI + z2**2).sum(1) - ld2)).mean().item()
print(f"[5] sigma -> 0 via bias -20: KL estimate per image, original formula {kl_orig:.1f} nats "
      f"(unbounded below), patched {kl_patched:.1f} nats")
ok &= kl_orig < -100 and kl_patched > 100
print("PASS" if ok else "FAIL")
