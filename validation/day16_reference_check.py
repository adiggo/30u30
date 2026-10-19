"""Day 16 (Vinyals, Bengio, Kudlur 2015, "Order Matters") -- reference checks.

  1. Permutation tests (eval mode, float64):
     a. implementation.SetEncoder is permutation-EQUIVARIANT: enc(x[perm]) == enc(x)[perm]
        (its mean-pool is therefore invariant) -- passes for any self-attention w/o positions.
     b. The ReadProcessWrite of the first round was NOT permutation-invariant: the WRITE block's
        PointerNetwork re-encodes the set with an order-sensitive nn.LSTM whose final state
        seeds the decoder, so shuffling the set changes the pointer distribution
        (compared after mapping indices back). [2026-10-18: ReadProcessWrite was rewritten as the
        paper's model; this check now rebuilds the old one from SetEncoder + PointerNetwork, and
        a new line checks the new model. Full tests: day16_process_block_check.py]
     c. A paper-faithful PROCESS block (Eqs. 3-7: q_t = LSTM(q*_{t-1}) with no input,
        e_{i,t} = <m_i, q_t>, a = softmax, r_t = sum_i a_{i,t} m_i, q*_t = [q_t, r_t]) written
        here from the equations IS invariant -- the component the repo omitted until 2026-10-18
        (implementation.ProcessBlock now; day16_process_block_check.py tests it against Eqs. 3-7).
  2. Pointer scores u_j = v^T tanh(W1 e_j + W2 d) vs an explicit loop; already-chosen and padded
     positions get probability exactly 0; greedy decoding emits a permutation.
  3. compute_loss(log_probs, targets): log_probs already holds log p(target_t) per step under
     teacher forcing, so the NLL is -log_probs.mean(). The original gather(1, targets) re-indexes
     the *step* axis by pointer values: identical for permutation targets (sorting, TSP), wrong
     otherwise (e.g. padded convex-hull targets). [Patched 2026-10-18 -> -log_probs.mean().]
  4. ConvexHullDataset labels vs an independent monotone-chain hull; the padding scheme repeated an
     index that the decoder has already masked -> teacher-forced log-prob = -inf. [2026-10-18: labels
     are now canonical (leftmost first, CCW) + a STOP pointer + IGNORE padding -> finite.]
  5. train.py's train_epoch/evaluate (originally there was no main() to run them): 3 epochs of
     sorting N=5 with ReadProcessWrite; loss falls, eval loss is the NLL of the *targets*
     (teacher-forced), accuracy is computed from the model's own argmax (patched metrics).

Run: OMP_NUM_THREADS=1 python validation/day16_reference_check.py
"""
import contextlib
import io
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
# PointerNetwork builds its first decoder input with torch.zeros(...) in the *default* dtype,
# so float64 checks need float64 as the default dtype.
torch.set_default_dtype(torch.float64)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "16_order_matters")
sys.path.insert(0, DAY)
import implementation as imp  # noqa: E402
import train as tr  # noqa: E402

RESULTS = []


def report(name, err, tol, expect_pass=True):
    ok = err <= tol
    RESULTS.append((name, ok == expect_pass, err))
    tag = "PASS" if ok else "FAIL"
    note = "" if ok == expect_pass else "  <-- unexpected"
    print(f"[{tag}] {name}: {err:.3e} (tol {tol:g}){note}")


torch.manual_seed(0)
B, N, D_in, H = 2, 6, 2, 16
x = torch.rand(B, N, D_in, dtype=torch.float64)
perm = torch.randperm(N)
inv = torch.argsort(perm)
xp = x[:, perm]

# ---------------------------------------------------------------- 1a. SetEncoder equivariance
enc = imp.SetEncoder(D_in, H, num_heads=4, num_layers=2, dropout=0.0).double().eval()
with torch.no_grad():
    e, ep = enc(x), enc(xp)
report("1a SetEncoder equivariant: enc(x[perm]) == enc(x)[perm]", (ep - e[:, perm]).abs().max().item(), 1e-10)
report("1a mean-pooled SetEncoder output invariant", (ep.mean(1) - e.mean(1)).abs().max().item(), 1e-10)

# ---------------------------------------------------------------- 1b. ReadProcessWrite invariance


# [PATCH 2026-10-18] ReadProcessWrite is now the paper's model; rebuild the first-round one
# (ReadProcessWrite(use_set_encoder=True) = SetEncoder -> PointerNetwork) from the same classes.
class OldReadProcessWrite(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.encoder = imp.SetEncoder(input_dim, hidden_dim, dropout=dropout)
        self.pointer_net = imp.PointerNetwork(hidden_dim, hidden_dim, dropout=dropout)

    def forward(self, inputs, input_lengths, max_steps, **kwargs):
        return self.pointer_net(self.encoder(inputs), input_lengths, max_steps, **kwargs)


rpw = OldReadProcessWrite(D_in, H, dropout=0.0).double().eval()
L = torch.full((B,), N)
with torch.no_grad():
    _, _, att = rpw(x, L, max_steps=1)
    _, _, attp = rpw(xp, L, max_steps=1)
    # attp[0][:, k] is the prob of permuted element k = original element perm[k]
    diff = (attp[0][:, inv] - att[0]).abs().max().item()
    m = rpw.encoder(x)
    _, (h1, _) = rpw.pointer_net.encoder(m)
    _, (h2, _) = rpw.pointer_net.encoder(rpw.encoder(xp))
report("1b OLD ReadProcessWrite: first-step pointer distribution invariant to shuffling the set",
       diff, 1e-6, expect_pass=False)
report("1b OLD ReadProcessWrite: decoder initial state (LSTM final h) invariant to shuffling",
       (h1 - h2).abs().max().item(), 1e-6, expect_pass=False)
new_rpw = imp.ReadProcessWrite(D_in, H, process_steps=5, glimpses=1).double().eval()
with torch.no_grad():
    _, _, att = new_rpw(x, L, max_steps=1)
    _, _, attp = new_rpw(xp, L, max_steps=1)
report("1b NEW ReadProcessWrite (process block): first-step pointer distribution invariant to shuffling",
       (attp[0][:, inv] - att[0]).abs().max().item(), 1e-12)


# ---------------------------------------------------------------- 1c. paper-faithful process block
class ProcessBlock(nn.Module):
    """Eqs. (3)-(7) of Order Matters, written from the paper: an LSTM with no input that
    attends over the memories m_i for T steps; f(m, q) = dot product."""

    def __init__(self, d, steps):
        super().__init__()
        self.cell = nn.LSTMCell(2 * d, d)   # input q*_{t-1} = [q_{t-1}, r_{t-1}] (Eq. 3)
        self.steps = steps

    def forward(self, m):
        Bm, _, d = m.shape
        q_star = torch.zeros(Bm, 2 * d, dtype=m.dtype)
        h = c = torch.zeros(Bm, d, dtype=m.dtype)
        for _ in range(self.steps):
            h, c = self.cell(q_star, (h, c))                  # q_t = LSTM(q*_{t-1})       (3)
            e_ = torch.einsum("bnd,bd->bn", m, h)             # e_{i,t} = <m_i, q_t>       (4)
            a = F.softmax(e_, dim=-1)                         # a_{i,t}                    (5)
            r = torch.einsum("bn,bnd->bd", a, m)              # r_t = sum_i a_{i,t} m_i    (6)
            q_star = torch.cat([h, r], -1)                    # q*_t = [q_t, r_t]          (7)
        return q_star


read = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, H)).double()  # "small MLP" read block
proc = ProcessBlock(H, steps=5).double()
with torch.no_grad():
    q1, q2 = proc(read(x)), proc(read(xp))
report("1c paper process block (T=5): q*_T invariant to shuffling the set", (q1 - q2).abs().max().item(), 1e-12)

# ---------------------------------------------------------------- 2. pointer scores + masking
pn = imp.PointerNetwork(D_in, H, dropout=0.0).double().eval()
x2 = torch.rand(2, 5, D_in, dtype=torch.float64)
L2 = torch.tensor([5, 3])
with torch.no_grad():
    ptr, lp, atts = pn(x2, L2, max_steps=3)
    # recompute step 0 by hand
    enc_out, (h0, c0) = pn.encoder(x2)
    hd, _ = pn.decoder_cell(torch.zeros(2, D_in, dtype=torch.float64), (h0[-1], c0[-1]))
    u = torch.empty(2, 5, dtype=torch.float64)
    for b in range(2):
        for j in range(5):
            u[b, j] = pn.v.weight[0] @ torch.tanh(pn.W1.weight @ enc_out[b, j] + pn.W2.weight @ hd[b])
    u[1, 3:] = float("-inf")                   # padding of the length-3 example
    p_ref = F.softmax(u, -1)
report("2 pointer probs (step 0) vs loop u_j = v^T tanh(W1 e_j + W2 d)", (atts[0] - p_ref).abs().max().item(), 1e-12)
chosen_mass = max(atts[t][b, ptr[b, s]].item() for b in range(2) for t in range(3) for s in range(t))
pad_mass = max(atts[t][1, 3:].abs().max().item() for t in range(3))
report("2 already-chosen positions get probability 0 at later steps", chosen_mass, 0.0)
report("2 padded positions get probability 0", pad_mass, 0.0)
with torch.no_grad():
    ptr_full, _, _ = pn(torch.rand(4, 7, D_in, dtype=torch.float64), torch.full((4,), 7), max_steps=7)
report("2 greedy decode emits permutations (rows with repeats)",
       float(sum(len(set(r.tolist())) != 7 for r in ptr_full)), 0)

# ---------------------------------------------------------------- 3. compute_loss
torch.manual_seed(1)
lp_steps = -torch.rand(3, 5, dtype=torch.float64)                     # log p(target_t), (B, T)
perm_targets = torch.stack([torch.randperm(5) for _ in range(3)])
nll = -lp_steps.mean()
report("3 compute_loss == NLL for permutation targets (coincidence)",
       abs(tr.compute_loss(lp_steps, perm_targets).item() - nll.item()), 1e-12)
hull_like = torch.tensor([[3, 0, 4, 3, 3], [1, 2, 1, 1, 1], [0, 4, 2, 0, 0]])  # padded hull labels
report("3 compute_loss == NLL for non-permutation targets (padded hull labels) [fails before patch]",
       abs(tr.compute_loss(lp_steps, hull_like).item() - nll.item()), 1e-12)
old_loss = -lp_steps.gather(1, hull_like).mean()      # the pre-patch formula, for the record
print(f"[INFO] pre-patch gather formula on the same hull-like targets: {old_loss.item():.4f} vs NLL {nll.item():.4f}")

# ---------------------------------------------------------------- 4. convex hull labels


def monotone_chain(p):
    """Andrew's monotone chain; returns hull vertex indices in counter-clockwise order."""
    idx = sorted(range(len(p)), key=lambda i: (p[i][0], p[i][1]))

    def cross(o, a, b):
        return (p[a][0] - p[o][0]) * (p[b][1] - p[o][1]) - (p[a][1] - p[o][1]) * (p[b][0] - p[o][0])
    lower, upper = [], []
    for i in idx:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    for i in reversed(idx):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


torch.manual_seed(2)
ds = tr.ConvexHullDataset(200, 10)
set_mismatch = orient_bad = 0
for k in range(200):
    pts, lab = ds[k]
    P = pts.double().numpy()
    ref = monotone_chain(P.tolist())
    k_h = len(ref)
    hull = lab[:k_h].tolist()
    set_mismatch += set(hull) != set(ref)
    area2 = sum(P[hull[i], 0] * P[hull[(i + 1) % k_h], 1] - P[hull[(i + 1) % k_h], 0] * P[hull[i], 1] for i in range(k_h))
    orient_bad += area2 <= 0                       # >0 means counter-clockwise
report("4 hull vertex sets == monotone-chain hull (200 random 10-point sets, mismatches)", float(set_mismatch), 0)
report("4 hull order is counter-clockwise (shoelace area > 0; violations)", float(orient_bad), 0)
pts, lab = ds[0]
# [PATCH 2026-10-18] labels end with STOP (= 10) and IGNORE_INDEX padding; the model needs the STOP slot
print(f"[INFO] example hull label (10 points): {lab.tolist()} (STOP = 10, then IGNORE_INDEX = -1)")
with torch.no_grad():
    _, lp_h, _ = imp.ReadProcessWrite(2, H, stop_token=True).double().eval()(
        pts[None].double(), torch.tensor([10]), max_steps=11, teacher_forcing=lab[None])
print(f"[INFO] teacher-forced log-probs on the hull label: {[round(v, 2) for v in lp_h[0].tolist()]}")
report("4 hull labels give finite teacher-forced log-probs (was -inf with repeat padding)",
       float(torch.isinf(lp_h).sum()), 0)

# ---------------------------------------------------------------- 5. train.py functions end to end
torch.set_default_dtype(torch.float32)
torch.manual_seed(3)
from torch.utils.data import DataLoader  # noqa: E402
model = imp.ReadProcessWrite(1, 32)  # [PATCH 2026-10-18] new constructor (was use_set_encoder=True, dropout=0.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
tl = DataLoader(tr.SortingDataset(1024, 5), batch_size=32, shuffle=True)
vl = DataLoader(tr.SortingDataset(256, 5), batch_size=64)
hist = []
for ep in range(3):
    with contextlib.redirect_stderr(io.StringIO()):          # silence tqdm
        trn = tr.train_epoch(model, tl, opt, "cpu")
    val = tr.evaluate(model, vl, "cpu")
    hist.append((round(trn[0], 3), round(trn[1], 3), round(val[0], 3), round(val[1], 3)))
print(f"[INFO] train.py sorting N=5 (train loss, train acc, val loss, val greedy acc) per epoch: {hist}")
report("5 train.py: loss decreases over 3 epochs (last - first)", hist[-1][0] - hist[0][0], 0.0)
# [PATCH 2026-10-18] With masking, the new model can sort before any training (a random pointer
# score that happens to be monotone in the value sorts every set), so a fast 1.0 no longer shows
# the metric is wrong. Test the metric itself: a model that cannot learn (lr = 0) and is not
# masked must score below 1.0; the pre-fix metric compared the fed targets with themselves (1.0).
torch.manual_seed(4)
frozen = imp.ReadProcessWrite(1, 32, mask_chosen=False)
with contextlib.redirect_stderr(io.StringIO()):
    _, frozen_acc = tr.train_epoch(frozen, tl, torch.optim.Adam(frozen.parameters(), lr=0.0), "cpu")
print(f"[INFO] train_epoch accuracy of an untrained, unmasked model that cannot learn: {frozen_acc:.3f}")
report("5 train.py: train accuracy is the model's own argmax, not the fed targets (1 if it reports 1.0)",
       float(frozen_acc >= 1.0), 0)

n_expected = sum(ok for _, ok, _ in RESULTS)
print(f"\nSUMMARY: {n_expected}/{len(RESULTS)} checks behaved as expected "
      f"(expected FAIL lines document the first-round ReadProcessWrite, rebuilt for comparison)")
