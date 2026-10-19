"""Day 23: free bits (Kingma et al. 2016 App. C.8, eq. 15; VLAE App. A) and whether z is used.

Objective checked (implementation.free_bits_loss / free_bits_kl), per minibatch of B images:
    recon + B * sum_j max(lambda, mean_b KL_bj),   KL_bj = log q(z_j|x_b) - log p(z_j | z_<j)
with one group per latent dimension and lambda = 0.01 nats x 784 pixels / latent_dim.

Unit checks (float64):
1. free_bits_kl value: B * sum_j max(lam, mean_b KL_bj); equals the plain KL sum when every group is
   above lam and B * K * lam when every group is below.
2. d free_bits_kl / d KL_bj (autograd) is exactly 0 for groups whose batch average is below lam and
   exactly 1 above it (the plain KL's gradient). Also with K < D (groups of several dimensions).
3. Through a VLAE (both priors): the gradient of free_bits_loss on every parameter (and on mu, logvar)
   equals the gradient of recon + (plain KL restricted to the above-lam dimensions), max relative
   error; without the flow, the KL term's gradient on mu/logvar of below-lam dimensions is exactly 0.
   Central finite differences of the objective on mu/logvar entries (lam kept away from the kinks).
4. free_bits_loss(free_bits=0) == loss_function; sum_d log_prior_terms == log N(z) + forward's
   flow correction (both priors).
5. AF prior, chain rule: log_prior_terms[:, d] depends only on z_<=d (Jacobian upper triangle 0), and
   in 2-D exp(term_0) and exp(term_1(z_0, .)) each integrate to 1 (conditional densities).
6. flow_inverse (AF sampling direction) inverts the flow: flow(flow_inverse(eps)) == eps.

Training comparison (MNIST, single core, about 5 min): same seed and data order, N_BATCHES (150)
batches with and without free bits (Gaussian prior, 3 decoder layers), then latent_usage on 1,000
test images: KL to the prior, -log p(x|z) with each image's own z vs another image's z (gap), and an
I(x; z) estimate. PASS if with free bits the decoder uses z (gap > 2 nats, I > 0.5 nat), the plain
ELBO collapses (KL < 1 nat, I < 0.5 nat), and the gaps differ by > 1 nat. Free bits only removes
the KL penalty below lambda, so the KL itself is reported, not tested against the budget.
DAY23_BROADCAST=1 adds the repo's original decoder (the same z vector at every pixel) with free
bits: its KL sits at a few nats while I(x; z) stays near 0. DAY23_BATCHES overrides N_BATCHES.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '23_variational_lossy_autoencoder')
sys.path.insert(0, DAY)
from implementation import (VLAE, loss_function, kl_terms, free_bits_kl,  # noqa: E402
                            free_bits_loss, latent_usage)

torch.set_num_threads(1)
LOG2PI = np.log(2 * np.pi)
ok = True


def check(name, passed, detail):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# ---------------------------------------------------------------- 1 + 2. free_bits_kl on a raw KL table
B, D = 6, 8
kl = torch.rand(B, D) * 0.4                     # per-dimension KLs in [0, 0.4)
kl[:, :3] += 1.0                                # dims 0-2: batch average > 1, the rest < 0.4
lam = 0.7
obj = free_bits_kl(kl, lam)
ref = B * torch.maximum(kl.mean(0), torch.full((D,), lam)).sum()
check("1a value = B * sum_j max(lam, mean_b KL_bj)", abs(obj - ref) < 1e-12, f"|d| = {abs(obj - ref).item():.1e}")
d_all_above = abs(free_bits_kl(kl, 0.0) - kl.sum()).item()
d_all_below = abs(free_bits_kl(kl, 5.0) - B * D * 5.0).item()
check("1b all groups above lam -> plain KL sum; all below -> B*K*lam", d_all_above < 1e-12 and d_all_below < 1e-12,
      f"|d| = {d_all_above:.1e}, {d_all_below:.1e}")

for K in (D, 4):                                # one group per dimension, then pairs of dimensions
    kl_leaf = kl.clone().requires_grad_(True)
    free_bits_kl(kl_leaf, lam * D / K, K).backward()
    group_mean = kl.reshape(B, K, D // K).sum(2).mean(0)          # (K,)
    above = (group_mean > lam * D / K).repeat_interleave(D // K)  # (D,): group flag for each dimension
    g = kl_leaf.grad
    below_max = g[:, ~above].abs().max().item()
    above_dev = (g[:, above] - 1).abs().max().item()
    check(f"2 dObjective/dKL_bj, K={K} groups ({int(above.sum())}/{D} dims above lam)",
          below_max == 0.0 and above_dev == 0.0,
          f"below lam: max |g| = {below_max:.1e}; above lam: max |g - 1| = {above_dev:.1e}")


# ---------------------------------------------------------------- 3. gradients through a VLAE
def objective_pieces(model, x, mu, logvar, eps):
    z = mu + eps * torch.exp(0.5 * logvar)      # reparameterisation with fixed noise
    logits = model.decode(x, z)
    return logits, z


for use_flow in (False, True):
    torch.manual_seed(1)
    model = VLAE(input_dim=1, latent_dim=8, hidden_dim=8, n_layers=1, use_flow=use_flow)
    x = (torch.rand(B, 1, 28, 28) > 0.6).double()
    with torch.no_grad():
        mu0, lv0 = model.encoder(x)
        mu0 = mu0 + torch.randn(B, 8) * torch.linspace(0.05, 1.5, 8)   # spread the per-dim KLs out
        lv0 = lv0 - torch.linspace(0.0, 1.0, 8)
    eps = torch.randn(B, 8)
    logits, z = objective_pieces(model, x, mu0, lv0, eps)
    kl_avg = kl_terms(mu0, lv0, z, model.log_prior_terms(z)).mean(0).detach()   # (D,) batch averages
    srt = kl_avg.sort().values
    lam_g = 0.5 * (srt[3] + srt[4]).item()      # 4 dims below, 4 above, half-way between two averages
    margin = 0.5 * (srt[4] - srt[3]).item()
    free_bits = lam_g * 8 / 784                 # nats per data dim that gives lam_g per group
    above = kl_avg > lam_g

    params = [p for p in model.parameters()]
    mu, lv = mu0.clone().requires_grad_(True), lv0.clone().requires_grad_(True)
    logits, z = objective_pieces(model, x, mu, lv, eps)
    obj, parts = free_bits_loss(model, logits, x, mu, lv, z, free_bits=free_bits)
    g_fb = torch.autograd.grad(obj, [mu, lv] + params, allow_unused=True)

    mu_r, lv_r = mu0.clone().requires_grad_(True), lv0.clone().requires_grad_(True)
    logits_r, z_r = objective_pieces(model, x, mu_r, lv_r, eps)
    recon_r = F.binary_cross_entropy_with_logits(logits_r, x, reduction='sum')
    kl_r = kl_terms(mu_r, lv_r, z_r, model.log_prior_terms(z_r))
    ref_obj = recon_r + (kl_r * above.double()).sum()     # below-lam groups contribute only a constant
    g_ref = torch.autograd.grad(ref_obj, [mu_r, lv_r] + params, allow_unused=True)
    errs = [rel_err(a, b) for a, b in zip(g_fb, g_ref) if a is not None and b.abs().max() > 0]
    const = B * lam_g * int((~above).sum())
    check(f"3a flow={use_flow}: grad of free_bits_loss vs recon + above-lam KL, {len(errs)} tensors "
          f"(mu, logvar, every parameter)", max(errs) < 1e-6 and abs(obj.item() - ref_obj.item() - const) < 1e-8,
          f"max rel err {max(errs):.1e}; objective - reference = {obj.item() - ref_obj.item():.6f} "
          f"(B * lam * #below = {const:.6f}); lam/group {lam_g:.3f}, margin {margin:.3f}")

    if not use_flow:   # factorised prior: dim d's KL depends only on mu_d, logvar_d
        mu_k, lv_k = mu0.clone().requires_grad_(True), lv0.clone().requires_grad_(True)
        z_k = mu_k + eps * torch.exp(0.5 * lv_k)
        kl_k = kl_terms(mu_k, lv_k, z_k, model.log_prior_terms(z_k))
        gm, gl = torch.autograd.grad(free_bits_kl(kl_k, lam_g), [mu_k, lv_k], retain_graph=True)
        gm_p, gl_p = torch.autograd.grad(kl_k.sum(), [mu_k, lv_k])
        below_max = max(gm[:, ~above].abs().max().item(), gl[:, ~above].abs().max().item())
        above_err = max(rel_err(gm[:, above], gm_p[:, above]), rel_err(gl[:, above], gl_p[:, above]))
        check("3b flow=False: KL-term gradient on mu/logvar of below-lam dims is 0, above = plain KL's",
              below_max == 0.0 and above_err < 1e-12, f"below max |g| = {below_max:.1e}; above rel err {above_err:.1e}")

    # central finite differences on mu / logvar entries (lam is `margin` away from every kink).
    # The objective is ~3,000 nats, so roundoff in the difference is ~1e-16 * 3e3 / h: use h = 1e-5
    # and the mixed error |fd - g| / max(|g|, 1)
    h, fd_err, g_max = 1e-5, 0.0, 0.0
    rng = np.random.RandomState(0)
    for which, base, g in (('mu', mu0, g_fb[0]), ('logvar', lv0, g_fb[1])):
        for _ in range(6):
            b, d = rng.randint(B), rng.randint(8)
            vals = []
            for sgn in (+1, -1):
                pert = base.clone()
                pert[b, d] += sgn * h
                m_, l_ = (pert, lv0) if which == 'mu' else (mu0, pert)
                lg, zz = objective_pieces(model, x, m_, l_, eps)
                vals.append(free_bits_loss(model, lg, x, m_, l_, zz, free_bits=free_bits)[0].item())
            fd = (vals[0] - vals[1]) / (2 * h)
            fd_err = max(fd_err, abs(fd - g[b, d].item()) / max(abs(g[b, d].item()), 1.0))
            g_max = max(g_max, abs(g[b, d].item()))
    check(f"3c flow={use_flow}: central finite differences vs autograd (12 entries of mu, logvar)",
          fd_err < 1e-6, f"max |fd - g| / max(|g|, 1) = {fd_err:.1e} (largest |g| checked {g_max:.2f})")

    # 4. free_bits = 0 is the plain loss; per-dim prior terms sum to forward()'s log p(z)
    torch.manual_seed(2)
    out = model(x)
    plain = loss_function(out[0], x, *out[1:])
    fb0, _ = free_bits_loss(model, out[0], x, out[1], out[2], out[3], free_bits=0.0)
    lp_sum = model.log_prior_terms(out[3]).sum(1)
    lp_fwd = -0.5 * (LOG2PI + out[3] ** 2).sum(1) + out[4]
    d0, d1 = abs(fb0 - plain).item(), (lp_sum - lp_fwd).abs().max().item()
    check(f"4 flow={use_flow}: free_bits=0 == loss_function; sum_d log_prior_terms == log N(z) + correction",
          d0 < 1e-9 and d1 < 1e-9, f"|d| = {d0:.1e}, {d1:.1e}")

# ---------------------------------------------------------------- 5. AF prior: conditional densities
torch.manual_seed(3)
m6 = VLAE(input_dim=1, latent_dim=6, hidden_dim=8, n_layers=1, use_flow=True)
zz = torch.randn(6)
J = torch.autograd.functional.jacobian(lambda v: m6.log_prior_terms(v[None])[0], zz)   # (6, 6): d term_d / d z_e
upper = torch.triu(J, diagonal=1).abs().max().item()
check("5a log_prior_terms[d] depends only on z_<=d (upper triangle of d term / d z)", upper == 0.0,
      f"max |upper| = {upper:.1e}")

m2 = VLAE(input_dim=1, latent_dim=2, hidden_dim=8, n_layers=1, use_flow=True)
grid = torch.linspace(-150, 150, 60001)          # step 0.005; the prior's scale is 1/prod(sigma) <= ~16
dz = (grid[1] - grid[0]).item()
with torch.no_grad():
    Z = torch.stack([grid, torch.zeros_like(grid)], 1)
    mass0 = (torch.exp(m2.log_prior_terms(Z)[:, 0]).sum() * dz).item()          # integral over z_0
    masses = []
    for z0 in (-3.0, 0.0, 2.5):
        Z = torch.stack([torch.full_like(grid, z0), grid], 1)
        masses.append((torch.exp(m2.log_prior_terms(Z)[:, 1]).sum() * dz).item())  # integral over z_1
err5 = max(abs(mass0 - 1), *[abs(v - 1) for v in masses])
check("5b 2-D AF prior: int exp(term_0) dz_0 and int exp(term_1(z_0, .)) dz_1 for z_0 in {-3, 0, 2.5}",
      err5 < 1e-6, f"{mass0:.8f}, " + ", ".join(f"{v:.8f}" for v in masses))

# ---------------------------------------------------------------- 6. flow_inverse
torch.manual_seed(4)
m8 = VLAE(input_dim=1, latent_dim=8, hidden_dim=8, n_layers=1, use_flow=True)
e = torch.randn(50, 8)
with torch.no_grad():
    zi = m8.flow_inverse(e)
    back = zi
    for f in m8.flow_steps:
        back, _ = f(back)
d6 = (back - e).abs().max().item()
check("6 flow(flow_inverse(eps)) == eps (50 draws, 8-d, 2 steps)", d6 < 1e-10, f"max |d| = {d6:.1e}")

# ---------------------------------------------------------------- 7. short training comparison
torch.set_default_dtype(torch.float32)
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402
import train_minimal as TM  # noqa: E402  (reuses its train loop)


class BroadcastZ(VLAE):
    """The repo's original conditioning: every pixel gets the same z vector (no feature map)."""
    def decode(self, x, z):
        h_latent = z.view(z.size(0), z.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))  # (B, D, H, W) view
        h = self.initial_conv(x)
        for layer in self.decoder_layers:
            h = layer(h, h_latent)
        return self.final_conv(h)


N_BATCHES = int(os.environ.get('DAY23_BATCHES', 150))
configs = [('free bits 0.01', 0.01, VLAE), ('plain ELBO', 0.0, VLAE)]
if os.environ.get('DAY23_BROADCAST') == '1':     # optional: free bits with the original broadcast z
    configs.append(('free bits, broadcast z', 0.01, BroadcastZ))
transform = transforms.Compose([transforms.ToTensor(), lambda t: (t > 0.5).float()])
data_dir = os.path.join(DAY, 'data')
train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
x_eval = TM.binarize(test_set.data[:1000])
results = {}
for label, fb, cls in configs:
    torch.manual_seed(42)
    model = cls(input_dim=1, latent_dim=32, n_layers=3, use_flow=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(train_set, batch_size=64, shuffle=True)
    hist = {'step': [], 'objective': [], 'recon': [], 'kl': []}
    t0 = time.time()
    print(f"\n--- {label}: {N_BATCHES} batches ---")
    TM.train(model, loader, opt, torch.device('cpu'), 1, free_bits=fb, max_batches=N_BATCHES,
             log_every=25, history=hist)
    torch.manual_seed(0)
    u = latent_usage(model, x_eval)
    u['seconds'] = time.time() - t0
    results[label] = u
    print(f"test (1000 images): KL {u['kl']:.2f} | -log p(x|z) own {u['nll_own']:.2f}, another image's z "
          f"{u['nll_shuffled']:.2f}, z~p(z) {u['nll_prior']:.2f} | gap {u['gap']:.2f} | I(x;z) {u['mi']:.2f} "
          f"| -ELBO {u['neg_elbo']:.2f} nats ({u['seconds']:.0f} s)")

budget = 0.01 * 784
fbr, pl = results['free bits 0.01'], results['plain ELBO']
check("7a free bits: the decoder uses z (gap > 2 nats, I(x;z) > 0.5 nat)", fbr['gap'] > 2.0 and fbr['mi'] > 0.5,
      f"KL {fbr['kl']:.2f} (budget {budget:.2f}), gap {fbr['gap']:.2f}, I {fbr['mi']:.2f}")
check("7b plain ELBO collapses (KL < 1 nat, I(x;z) < 0.5 nat)", pl['kl'] < 1.0 and pl['mi'] < 0.5,
      f"KL {pl['kl']:.3f}, gap {pl['gap']:.3f}, I {pl['mi']:.3f}")
check("7c free bits gains more from the right z than the plain ELBO (gap difference > 1 nat)",
      fbr['gap'] - pl['gap'] > 1.0, f"{fbr['gap']:.2f} vs {pl['gap']:.2f}")
if 'free bits, broadcast z' in results:
    bc = results['free bits, broadcast z']
    print(f"[info] free bits with broadcast z: KL {bc['kl']:.2f} but I(x;z) {bc['mi']:.2f}, gap {bc['gap']:.2f}: "
          f"the KL budget is not information")
print("PASS" if ok else "FAIL")
