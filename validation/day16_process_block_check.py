"""Day 16 (Vinyals, Bengio, Kudlur 2015, "Order Matters") -- Read-Process-Write invariance checks.

The model in papers/16_order_matters/implementation.py (patched 2026-10-18) is the paper's
Read-Process-Write (Sec. 4.2-4.3): per-element MLP read, process block of Eqs. 3-7, pointer-network
write started from q*_T. Checks (eval mode, float64 unless stated):

  1. Read block is permutation-EQUIVARIANT: read(x[perm]) == read(x)[perm].
  2. Process block == Eqs. 3-7 written out by hand (NumPy: LSTM gate equations, dot-product scores,
     softmax, weighted sum, concatenation) from the block's own weights, T = 5.
  3. Process block output q*_T is INVARIANT to shuffling the set: T = 1, 5, 10 in float64 (tol 1e-12)
     and float32 (tol 1e-6); also with padded sets of different sizes; the attention weights a_t
     permute with the set.
  4. Full model: after shuffling the set, the output distribution over input elements is the
     original one permuted the same way (teacher-forced, every step), log p(target) is unchanged and
     greedy pointers map through the permutation -- for P = 0/5 process steps, 0/1 glimpses, with
     and without the STOP slot, with and without masking of chosen elements, with padding.
  5. The OLD model (until 2026-10-18: SetEncoder -> PointerNetwork, whose LSTM re-reads the encoded
     set in order; rebuilt here from the two classes, which are unchanged) is NOT invariant at
     initialisation. [expected FAIL lines: they document the old model]
  6. Trained on sorting (N = 5, 800 Adam steps each): fraction of 2000 test sets whose greedy output
     (as a sequence of values) changes when the set is shuffled -- old model > 0 (expected FAIL),
     new model 0; mean |dh| of the old model's LSTM state (the decoder's initial state).
  7. STOP slot and hull labels: train.ConvexHullDataset targets == an independent monotone-chain hull
     (start at the leftmost point, counter-clockwise) + STOP + IGNORE padding; teacher-forced loss
     and gradients finite; compute_loss skips IGNORE steps; greedy decoding repeats STOP after it.
  8. [INFO] With chosen elements masked, UNTRAINED models (seeds 0-5, hidden 32) sort 2000 sets of 5
     numbers in [0, 10): a random pointer score that is monotone in the value already sorts (or
     reverse-sorts) every set -- why masked sorting says little about the process block.

Run: OMP_NUM_THREADS=1 python validation/day16_process_block_check.py      (about 1 minute)
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "16_order_matters")
sys.path.insert(0, DAY)
import implementation as imp  # noqa: E402
import train as tr  # noqa: E402

RESULTS = []


def report(name, err, tol, expect_pass=True):
    ok = err <= tol
    RESULTS.append((name, ok == expect_pass))
    tag = "PASS" if ok else "FAIL"
    note = "" if ok == expect_pass else "  <-- unexpected"
    print(f"[{tag}] {name}: {err:.3e} (tol {tol:g}){note}")


def shuffle_rows(x, lengths, gen):
    """Shuffle the first lengths[b] elements of each row b (padding stays at the end).
    Returns x_shuffled and slot_inv (B, N + 1): original slot i of row b sits at slot_inv[b, i]
    of the shuffled row; column N maps the STOP slot to itself."""
    B, N = x.shape[:2]
    perms = []
    for b in range(B):
        L = int(lengths[b])
        perms.append(torch.cat([torch.randperm(L, generator=gen), torch.arange(L, N)]))
    perm = torch.stack(perms)                                   # (B, N): new slot k holds old slot perm[b, k]
    # gather along axis 1 with the index broadcast over the feature axis: x_s[b, k] = x[b, perm[b, k]]
    x_s = torch.gather(x, 1, perm.unsqueeze(-1).expand_as(x))
    slot_inv = torch.cat([torch.argsort(perm, dim=1), torch.full((B, 1), N)], dim=1)  # argsort inverts a permutation
    return x_s, slot_inv


def map_targets(y, slot_inv):
    """Targets for the shuffled set: element / STOP indices go through slot_inv, IGNORE stays."""
    return torch.where(y >= 0, torch.gather(slot_inv, 1, y.clamp(min=0)), y)


# ---------------------------------------------------------------- 1. read block equivariance
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
B, N, D_in, H = 3, 7, 2, 16
x = torch.rand(B, N, D_in, dtype=torch.float64)
full = torch.full((B,), N)
x_s, slot_inv = shuffle_rows(x, full, gen)
inv = slot_inv[:, :N]                                         # (B, N) without the STOP column
model = imp.ReadProcessWrite(D_in, H, process_steps=5, glimpses=1).double().eval()
with torch.no_grad():
    m, m_s = model.read(x), model.read(x_s)
    # m_s[b, inv[b, i]] is the memory of original element i in the shuffled run
    m_s_back = torch.gather(m_s, 1, inv.unsqueeze(-1).expand_as(m_s))
report("1 read block equivariant: read(x[perm]) == read(x)[perm]", (m_s_back - m).abs().max().item(), 1e-12)

# ---------------------------------------------------------------- 2. process block == Eqs. 3-7 by hand
proc = model.process
with torch.no_grad():
    q_star_code, att_code = proc(m, torch.ones(B, N, dtype=torch.bool))
W_ih, W_hh = proc.cell.weight_ih.detach().numpy(), proc.cell.weight_hh.detach().numpy()  # (4H, 2H), (4H, H)
b_all = (proc.cell.bias_ih + proc.cell.bias_hh).detach().numpy()          # (4H,)
M = m.numpy()                                                            # (B, N, H)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


worst = 0.0
for b in range(B):
    q_star = np.zeros(2 * H)
    h, c = np.zeros(H), np.zeros(H)
    for t in range(proc.steps):
        gates = W_ih @ q_star + W_hh @ h + b_all                         # (4H,) PyTorch order: i, f, g, o
        i_g, f_g, g_g, o_g = np.split(gates, 4)                          # four (H,) blocks
        c = sigmoid(f_g) * c + sigmoid(i_g) * np.tanh(g_g)
        h = sigmoid(o_g) * np.tanh(c)                                    # (3) q_t = LSTM(q*_{t-1})
        e = np.array([M[b, i] @ h for i in range(N)])                    # (4) e_i = m_i . q_t
        a = np.exp(e - e.max()) / np.exp(e - e.max()).sum()              # (5) softmax (shifted by max: same value, no overflow)
        r = sum(a[i] * M[b, i] for i in range(N))                        # (6) r_t = sum_i a_i m_i
        q_star = np.concatenate([h, r])                                  # (7) q*_t = [q_t, r_t]
        worst = max(worst, np.abs(a - att_code[t][b].numpy()).max())
    worst = max(worst, np.abs(q_star - q_star_code[b].numpy()).max())
report("2 process block == Eqs. 3-7 written out in NumPy (q*_5 and every a_t)", worst, 1e-12)

# ---------------------------------------------------------------- 3. process block invariance
for T in (1, 5, 10):
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
        torch.manual_seed(T)
        mdl = imp.ReadProcessWrite(D_in, H, process_steps=T).to(dtype).eval()
        xx = x.to(dtype)
        xx_s = x_s.to(dtype)
        with torch.no_grad():
            _, _, q1, a1 = mdl.encode(xx, full)
            _, _, q2, a2 = mdl.encode(xx_s, full)
        err = (q1 - q2).abs().max().item()
        # a2[t][b, inv[b, i]] is the weight on original element i: gather it back to compare
        att_err = max((torch.gather(p2, 1, inv) - p1).abs().max().item() for p1, p2 in zip(a1, a2))
        report(f"3 process block T={T:2d} {str(dtype)[6:]}: q*_T invariant to shuffling "
               f"(a_t permute with the set: {att_err:.1e})", max(err, att_err), tol)
lengths = torch.tensor([N, N - 2, N - 4])                     # padded sets of sizes 7, 5, 3
x_pad = x.clone()
for b in range(B):
    x_pad[b, lengths[b]:] = 99.0                              # garbage in the padding must not matter
x_pad_s, slot_inv_pad = shuffle_rows(x_pad, lengths, gen)
with torch.no_grad():
    q1 = model.encode(x_pad, lengths)[2]
    q2 = model.encode(x_pad_s, lengths)[2]
    x_pad2 = x_pad.clone()
    for b in range(B):
        x_pad2[b, lengths[b]:] = -5.0
    q3 = model.encode(x_pad2, lengths)[2]
report("3 padded sets (sizes 7, 5, 3): q*_T invariant to shuffling the real elements",
       (q1 - q2).abs().max().item(), 1e-12)
report("3 padded sets: q*_T independent of the padding values", (q1 - q3).abs().max().item(), 0.0)


# ---------------------------------------------------------------- 4. full model equivariance
def full_model_errors(mdl, xx, xx_s, lens, s_inv, targets, steps):
    """Max errors between the run on xx and the run on its shuffle xx_s."""
    with torch.no_grad():
        _, lp1, att1 = mdl(xx, lens, steps, teacher_forcing=targets)
        _, lp2, att2 = mdl(xx_s, lens, steps, teacher_forcing=map_targets(targets, s_inv))
        g1, _, _ = mdl(xx, lens, steps)
        g2, _, _ = mdl(xx_s, lens, steps)
    S = att1[0].size(1)
    cols = s_inv[:, :S]                                       # STOP column only when the model has one
    dist = max((torch.gather(p2, 1, cols) - p1).abs().max().item() for p1, p2 in zip(att1, att2))
    lp = (lp1 - lp2).abs().max().item()
    greedy = float((map_targets(g1, s_inv) != g2).sum())      # pointers of run 1 mapped into run 2's slots
    return dist, lp, greedy


worst = {"dist": 0.0, "lp": 0.0, "greedy": 0.0}
configs = 0
for P in (0, 5):
    for G in (0, 1):
        for stop in (False, True):
            for mask_chosen in (True, False):
                torch.manual_seed(10 * P + 2 * G + stop)
                mdl = imp.ReadProcessWrite(D_in, H, process_steps=P, glimpses=G, stop_token=stop,
                                           mask_chosen=mask_chosen).double().eval()
                with torch.no_grad():                         # random nonzero STOP memory, not the zero init
                    if stop:
                        mdl.stop_memory.normal_()
                if stop:                                      # hull-like targets: 3-4 elements, STOP, IGNORE
                    y = torch.tensor([[2, 0, 5, N, -1, -1], [1, 3, 0, 4, N, -1], [0, 2, 1, N, -1, -1]])
                else:                                         # one valid ordering of each padded set
                    y = torch.stack([torch.cat([torch.randperm(int(L)), torch.zeros(N - int(L), dtype=torch.long)])
                                     for L in lengths])[:, :int(lengths.min())]
                errs = full_model_errors(mdl, x_pad, x_pad_s, lengths, slot_inv_pad, y, y.size(1))
                for k, e in zip(("dist", "lp", "greedy"), errs):
                    worst[k] = max(worst[k], e)
                configs += 1
report(f"4 full model ({configs} configs): pointer distributions permute with the set (every step)",
       worst["dist"], 1e-12)
report("4 full model: log p(target) unchanged by shuffling (targets mapped)", worst["lp"], 1e-12)
report("4 full model: greedy pointers map through the permutation (mismatches)", worst["greedy"], 0)
torch.manual_seed(5)
mdl32 = imp.ReadProcessWrite(D_in, 64, process_steps=5, glimpses=1).eval()
x32 = torch.rand(64, 10, D_in)
x32_s, s_inv32 = shuffle_rows(x32, torch.full((64,), 10), gen)
y32 = x32[..., 0].argsort(dim=1)
d32, lp32, _ = full_model_errors(mdl32, x32, x32_s, torch.full((64,), 10), s_inv32, y32, 10)
report("4 float32, H=64, N=10, 64 sets: pointer distributions permute with the set",
       max(d32, lp32), 1e-6)


# ---------------------------------------------------------------- 5. old model: not invariant
class OldReadProcessWrite(nn.Module):
    """ReadProcessWrite(use_set_encoder=True) as it was until 2026-10-18 (same classes, same order)."""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.encoder = imp.SetEncoder(input_dim, hidden_dim, dropout=dropout)
        self.pointer_net = imp.PointerNetwork(hidden_dim, hidden_dim, dropout=dropout)

    def forward(self, inputs, input_lengths, max_steps, **kwargs):
        return self.pointer_net(self.encoder(inputs), input_lengths, max_steps, **kwargs)

    def decoder_init_state(self, inputs):
        """The LSTM's final hidden state over the encoded set: the decoder's initial state."""
        _, (h, _) = self.pointer_net.encoder(self.encoder(inputs))
        return h[-1]


torch.set_default_dtype(torch.float64)   # PointerNetwork builds its first decoder input in the default dtype
torch.manual_seed(0)
old = OldReadProcessWrite(D_in, H, dropout=0.0).eval()
with torch.no_grad():
    dh = (old.decoder_init_state(x) - old.decoder_init_state(x_s)).abs()
    y_sort = x[..., 0].argsort(dim=1)
    _, _, a_old = old(x, full, N, teacher_forcing=y_sort)
    _, _, a_old_s = old(x_s, full, N, teacher_forcing=map_targets(y_sort, slot_inv))
dist_old = max((torch.gather(p2, 1, inv) - p1).abs().max().item() for p1, p2 in zip(a_old, a_old_s))
print(f"[INFO] old model at init: decoder initial state |dh| mean {dh.mean().item():.3e}, max {dh.max().item():.3e}")
report("5 OLD model: decoder initial state invariant to shuffling (max |dh|)", dh.max().item(), 1e-6,
       expect_pass=False)
report("5 OLD model: pointer distributions permute with the set (teacher-forced)", dist_old, 1e-6,
       expect_pass=False)
torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------- 6. trained models: does shuffling change answers?
def train_sorting(mdl, steps=800, batch=64, n=5, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
    lens = torch.full((batch,), n)
    mdl.train()
    for _ in range(steps):
        xb = torch.rand(batch, n, 1) * 10                     # train.SortingDataset's range [0, 10)
        yb = xb[..., 0].argsort(dim=1)
        _, lp, _ = mdl(xb, lens, n, teacher_forcing=yb)
        loss = imp.compute_loss(lp, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 2.0)
        opt.step()
    return mdl.eval()


t0 = time.time()
torch.manual_seed(1)
new_m = train_sorting(imp.ReadProcessWrite(1, 64, process_steps=5, glimpses=1))
torch.manual_seed(1)
old_m = train_sorting(OldReadProcessWrite(1, 64))
gen_t = torch.Generator().manual_seed(7)
xt = torch.rand(2000, 5, 1, generator=gen_t) * 10
xt_s, s_inv_t = shuffle_rows(xt, torch.full((2000,), 5), gen_t)
lt = torch.full((2000,), 5)
yt = xt[..., 0].argsort(dim=1)
changed = {}
for name, mdl in (("new", new_m), ("old", old_m)):
    with torch.no_grad():
        p1, _, _ = mdl(xt, lt, 5)
        p2, _, _ = mdl(xt_s, lt, 5)
    # compare the *values* each run outputs: gather picks x[b, p[b, t]] -> (2000, 5)
    v1 = torch.gather(xt[..., 0], 1, p1)
    v2 = torch.gather(xt_s[..., 0], 1, p2)
    changed[name] = (v1 != v2).any(dim=1).float().mean().item()
    acc = (p1 == yt).all(dim=1).float().mean().item()
    print(f"[INFO] trained {name} model (800 steps): greedy exact sorts {acc:.1%} on 2000 test sets; "
          f"answer changes after a shuffle on {changed[name]:.2%} of them")
with torch.no_grad():
    dh_trained = (old_m.decoder_init_state(xt) - old_m.decoder_init_state(xt_s)).abs()
print(f"[INFO] trained old model: decoder initial state |dh| under shuffling, mean {dh_trained.mean().item():.3f} "
      f"(max {dh_trained.max().item():.3f}); training both took {time.time() - t0:.0f}s")
report("6 NEW model trained: fraction of test sets whose greedy answer changes when shuffled", changed["new"], 0.0)
report("6 OLD model trained: fraction of test sets whose greedy answer changes when shuffled", changed["old"], 0.0,
       expect_pass=False)

# ---------------------------------------------------------------- 7. STOP slot and convex-hull labels


def monotone_chain(p):
    """Andrew's monotone chain: hull vertex indices, counter-clockwise from the lowest (x, y) point."""
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
NH = 10
ds = tr.ConvexHullDataset(300, NH)
bad = 0
for k in range(300):
    pts, lab = ds[k]
    ref = monotone_chain(pts.double().numpy().tolist())
    expected = ref + [NH] + [imp.IGNORE_INDEX] * (NH - len(ref))
    bad += lab.tolist() != expected
report("7 hull targets == monotone chain from the leftmost point, CCW, + STOP + IGNORE (300 sets, mismatches)",
       float(bad), 0)
pts_b, lab_b = zip(*[ds[k] for k in range(32)])
pts_b, lab_b = torch.stack(pts_b), torch.stack(lab_b)
torch.manual_seed(3)
hull_model = imp.ReadProcessWrite(2, 32, process_steps=5, glimpses=1, stop_token=True)
_, lp_h, _ = hull_model(pts_b, torch.full((32,), NH), NH + 1, teacher_forcing=lab_b)
loss_h = imp.compute_loss(lp_h, lab_b)
loss_h.backward()
grad_ok = all(torch.isfinite(p.grad).all() for p in hull_model.parameters() if p.grad is not None)
print(f"[INFO] hull batch (32 sets of 10 points): teacher-forced loss {loss_h.item():.4f} "
      f"(uniform guess ~ ln 11 = {np.log(NH + 1):.4f}); example target {lab_b[0].tolist()}")
report("7 hull: teacher-forced loss and all gradients finite (0 = yes)",
       float(not (torch.isfinite(loss_h).item() and grad_ok)), 0)
valid = lab_b >= 0
manual = -(lp_h.detach()[valid]).mean()                     # boolean indexing keeps only the valid steps
report("7 compute_loss skips IGNORE steps (== mean NLL over valid steps)",
       abs(loss_h.item() - manual.item()), 1e-6)
with torch.no_grad():
    g_h, _, _ = hull_model.eval()(pts_b, torch.full((32,), NH), NH + 1)
after_stop = 0
for row in g_h.tolist():
    if NH in row:
        after_stop += any(v != NH for v in row[row.index(NH):])
print(f"[INFO] untrained greedy hull decode: {sum(NH in r for r in g_h.tolist())}/32 rows emit STOP within {NH + 1} steps")
report("7 greedy decoding repeats STOP after the first STOP (rows violating)", float(after_stop), 0)

# ---------------------------------------------------------------- 8. untrained models with masking
gen_u = torch.Generator().manual_seed(11)
xu = torch.rand(2000, 5, 1, generator=gen_u) * 10
yu = xu[..., 0].argsort(dim=1)
rows = []
for seed in range(6):
    torch.manual_seed(seed)
    untrained = imp.ReadProcessWrite(1, 32).eval()
    with torch.no_grad():
        pu, _, _ = untrained(xu, torch.full((2000,), 5), 5)
    rows.append(f"seed {seed}: sorted {(pu == yu).all(dim=1).float().mean().item():.1%}, "
                f"reverse-sorted {(pu == yu.flip(dims=[1])).all(dim=1).float().mean().item():.1%}")
print("[INFO] untrained ReadProcessWrite, masking on, greedy decoding of 2000 sets of 5 numbers:\n       "
      + "\n       ".join(rows))

n_expected = sum(ok for _, ok in RESULTS)
print(f"\nSUMMARY: {n_expected}/{len(RESULTS)} checks behaved as expected "
      f"(expected FAIL lines document the pre-2026-10-18 model)")
