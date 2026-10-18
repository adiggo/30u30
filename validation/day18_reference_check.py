"""Day 18 (Vinyals, Fortunato, Jaitly 2015, "Pointer Networks") -- reference checks.

  1. PointerAttention: u^i_j = v^T tanh(W1 e_j + W2 d_i), log_softmax over j (Eq. 3 / Sec. 2.3)
     vs an explicit double loop; masked positions -> probability exactly 0.
  2. PointerNetwork.forward decode loop:
     a. the re-selection mask is created but never updated -> repeated indices are possible;
     b. there is no teacher forcing: step t is conditioned on the model's own argmax at t-1,
        whereas Eq. (1)-(2) maximise log p(C_i | C_1..C_{i-1}, P) on the *true* prefix.
     Uses a trained checkpoint (model_weights.pth as train_minimal.py writes it; sorting n=5):
     greedy exact-sort accuracy, fraction of outputs that are permutations, accuracy when
     already-chosen indices are masked at inference, and repo loss vs teacher-forced NLL.
  3. Task data: solution_03 hull labels (scipy) vs an independent monotone-chain hull (vertex
     set, counter-clockwise order) and how often they start at the lowest index (the paper's
     convention, Sec. 3.1); solution_04 tour cost vs a vectorised closed-tour length.

Run: OMP_NUM_THREADS=1 python validation/day18_reference_check.py [path/to/model_weights.pth]
     (default: papers/18_pointer_networks/model_weights.pth, which train_minimal.py writes when run
     from that directory; without it check 2 reports numbers for an untrained model)
"""
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "18_pointer_networks")
sys.path.insert(0, DAY)
import implementation as imp  # noqa: E402

RESULTS = []


def report(name, err, tol, expect_pass=True):
    ok = err <= tol
    RESULTS.append(ok == expect_pass)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {err:.3e} (tol {tol:g})" + ("" if ok == expect_pass else "  <-- unexpected"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{DAY}/exercises/solutions/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


torch.manual_seed(0)
# ------------------------------------------------------------------ 1. pointer attention
H, B, N = 8, 2, 5
att = imp.PointerAttention(H).double()
d, e = torch.randn(B, H, dtype=torch.float64), torch.randn(B, N, H, dtype=torch.float64)
mask = torch.zeros(B, N, dtype=torch.bool)
mask[1, [0, 3]] = True
u = torch.empty(B, N, dtype=torch.float64)
for b in range(B):
    for j in range(N):
        u[b, j] = att.v.weight[0] @ torch.tanh(att.W1.weight @ e[b, j] + att.W2.weight @ d[b])
ref = F.log_softmax(u.masked_fill(mask, float("-inf")), -1)
with torch.no_grad():
    out = att(d, e, mask)
report("1 PointerAttention log-probs vs loop (unmasked entries)", (out[~mask] - ref[~mask]).abs().max().item(), 1e-12)
report("1 masked positions have probability exactly 0", out[mask].exp().max().item(), 0.0)

# ------------------------------------------------------------------ 2. decode loop on the trained model
ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DAY, "model_weights.pth")
model = imp.PointerNetwork(input_size=1, hidden_size=64)
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    print(f"[INFO] loaded checkpoint {ckpt}")
else:
    print(f"[WARN] no checkpoint at {ckpt}: numbers below are for an untrained model")
model.eval()
torch.manual_seed(1)
x = torch.rand(2000, 5, 1)
y = x[..., 0].argsort(1)
with torch.no_grad():
    logp = model(x)                                    # (B, 5, 5), greedy feedback, no mask updates
pred = logp.argmax(-1)
exact = (pred == y).all(1).float().mean().item()
is_perm = torch.tensor([len(set(r.tolist())) == 5 for r in pred]).float().mean().item()
repo_loss = F.nll_loss(logp.reshape(-1, 5), y.reshape(-1)).item()


def decode(model, x, targets=None, mask_chosen=False):
    """Same modules as PointerNetwork.forward; optionally teacher-forced and/or masked."""
    Bx, n, _ = x.shape
    enc, (h, c) = model.encoder(model.input_proj(x))
    inp, hid = torch.zeros(Bx, 1, model.hidden_size), (h, c)
    chosen = torch.zeros(Bx, n, dtype=torch.bool)
    out = []
    for t in range(n):
        o, hid = model.decoder(inp, hid)
        lp = model.pointer(o.squeeze(1), enc, chosen if mask_chosen else None)
        out.append(lp)
        nxt = targets[:, t] if targets is not None else lp.argmax(-1)
        chosen = chosen.scatter(1, nxt[:, None], True)
        inp = enc[torch.arange(Bx), nxt].unsqueeze(1)
    return torch.stack(out, 1)


with torch.no_grad():
    same = decode(model, x)
    tf = decode(model, x, targets=y)
    masked = decode(model, x, mask_chosen=True)
report("2 re-implementation reproduces PointerNetwork.forward", (same - logp).abs().max().item(), 1e-5)
tf_nll = F.nll_loss(tf.reshape(-1, 5), y.reshape(-1)).item()
exact_masked = (masked.argmax(-1) == y).all(1).float().mean().item()
print(f"[INFO] trained model, 2000 fresh sets of 5: greedy exact-sort {exact:.1%}, valid permutations {is_perm:.1%}, "
      f"exact-sort with chosen-index masking at inference {exact_masked:.1%}")
print(f"[INFO] repo loss (own-argmax prefixes) = {repo_loss:.4f}; teacher-forced NLL (true prefixes, Eq. 1-2) = {tf_nll:.4f}")
report("2a all greedy outputs are permutations (1 - fraction valid)", 1 - is_perm, 0.0, expect_pass=False)

# ------------------------------------------------------------------ 3. task data
sol3, sol4 = load("solution_03_convex_hull"), load("solution_04_tsp_cost")


def monotone_chain(p):
    idx = sorted(range(len(p)), key=lambda i: (p[i][0], p[i][1]))

    def cross(o, a, b):
        return (p[a][0] - p[o][0]) * (p[b][1] - p[o][1]) - (p[a][1] - p[o][1]) * (p[b][0] - p[o][0])
    lo, up = [], []
    for i in idx:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], i) <= 0:
            lo.pop()
        lo.append(i)
    for i in reversed(idx):
        while len(up) >= 2 and cross(up[-2], up[-1], i) <= 0:
            up.pop()
        up.append(i)
    return lo[:-1] + up[:-1]


rng = np.random.default_rng(0)
mism = ccw_bad = not_lowest = 0
for _ in range(300):
    P = rng.random((10, 2))
    h = sol3.get_convex_hull_pointers(P)
    mism += set(h) != set(monotone_chain(P.tolist()))
    area2 = sum(P[h[i], 0] * P[h[(i + 1) % len(h)], 1] - P[h[(i + 1) % len(h)], 0] * P[h[i], 1] for i in range(len(h)))
    ccw_bad += area2 <= 0
    not_lowest += h[0] != min(h)
report("3 hull vertex sets == monotone chain (300 sets of 10 points; mismatches)", float(mism), 0)
report("3 hull order counter-clockwise (violations)", float(ccw_bad), 0)
print(f"[INFO] scipy hull labels start at the lowest index in {300 - not_lowest}/300 cases "
      f"(paper Sec. 3.1 starts at the lowest index; rotate before using them as targets)")
pts = torch.rand(9, 2, dtype=torch.float64)
errs = []
for _ in range(50):
    tour = torch.randperm(9).tolist()
    q = pts[tour].numpy()
    ref_len = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1).sum()   # closed tour incl. last->first
    errs.append(abs(sol4.calculate_tour_cost(pts, tour) - ref_len))
report("3 calculate_tour_cost vs vectorised closed-tour length (50 random tours)", max(errs), 1e-12)
report("3 unit square tour 0-1-2-3 has cost 4", abs(sol4.calculate_tour_cost(
    torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]), [0, 1, 2, 3]) - 4.0), 1e-12)
print(f"\nSUMMARY: {sum(RESULTS)}/{len(RESULTS)} checks behaved as expected")
