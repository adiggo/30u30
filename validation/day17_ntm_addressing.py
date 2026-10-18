"""Day 17 reference check: Neural Turing Machine addressing, read and write.

Independent float64 NumPy re-implementation written directly from Graves, Wayne & Danihelka (2014),
arXiv 1410.5401v2, Section 3:
  Eq 2  read      r = sum_i w(i) M(i)
  Eq 3  erase     M~(i) = M_{t-1}(i) * (1 - w(i) e)
  Eq 4  add       M(i)  = M~(i) + w(i) a
  Eq 5  content   w_c(i) = softmax_i(beta * K[k, M(i)])
  Eq 6  cosine    K[u, v] = u.v / (|u| |v|)
  Eq 7  gate      w_g = g w_c + (1 - g) w_{t-1}
  Eq 8  shift     w~(i) = sum_j w_g(j) s(i - j)   (indices mod N; s has entries for shifts -1, 0, +1)
  Eq 9  sharpen   w(i) = w~(i)^gamma / sum_j w~(j)^gamma
compared against implementation.py (NTMHead._addressing and one full NTM.forward step, LSTM controller
included). Also: a shift-direction probe and torch.autograd.gradcheck of one step.

Usage: OMP_NUM_THREADS=1 python validation/day17_ntm_addressing.py [path/to/17_neural_turing_machines]
"""
import os
import sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "17_neural_turing_machines")
sys.path.insert(0, DAY)
from implementation import NTM  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)


def softplus(x):
    return np.log1p(np.exp(x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def ref_address(raw, prev_w, mem, M, is_read, shift_labels=(-1, 0, 1)):
    """Eqs 5-9 for ONE batch element. raw = the head's linear output (same slicing order as the repo:
    k[M], beta[1], g[1], s[3], gamma[1], (erase[M], add[M]) ); squashing as in the repo
    (softplus, sigmoid, softmax, 1+softplus) -- the paper only requires beta>0, g in (0,1), s a
    distribution, gamma>=1."""
    N = mem.shape[0]
    p = 0
    k = raw[p:p + M]; p += M
    beta = softplus(raw[p]); p += 1
    g = sigmoid(raw[p]); p += 1
    s = softmax(raw[p:p + 3]); p += 3
    gamma = 1.0 + softplus(raw[p]); p += 1
    # Eq 6: cosine similarity of the key with every row, written as an explicit loop over rows
    K = np.array([mem[i] @ k / (np.linalg.norm(mem[i]) * np.linalg.norm(k)) for i in range(N)])
    wc = softmax(beta * K)                                   # Eq 5
    wg = g * wc + (1 - g) * prev_w                           # Eq 7
    s_full = np.zeros(N)                                     # s as a function on Z_N
    for idx, lab in enumerate(shift_labels):
        s_full[lab % N] += s[idx]
    wt = np.array([sum(wg[j] * s_full[(i - j) % N] for j in range(N)) for i in range(N)])  # Eq 8
    w = wt ** gamma / np.sum(wt ** gamma)                    # Eq 9
    out = {"w": w}
    if not is_read:
        out["erase"] = sigmoid(raw[p:p + M]); p += M
        out["add"] = raw[p:p + M]
    return out


def lstm_cell(x, h, c, cell):
    """Plain LSTM equations with PyTorch's gate order (i, f, g, o)."""
    W_ih = cell.weight_ih.detach().numpy(); W_hh = cell.weight_hh.detach().numpy()
    b = cell.bias_ih.detach().numpy() + cell.bias_hh.detach().numpy()
    z = W_ih @ x + W_hh @ h + b
    H = h.shape[0]
    i, f, gg, o = sigmoid(z[:H]), sigmoid(z[H:2 * H]), np.tanh(z[2 * H:3 * H]), sigmoid(z[3 * H:])
    c2 = f * c + i * gg
    return o * np.tanh(c2), c2


results = []

# ---------------------------------------------------------------- 1. one full NTM step, float64
for (B, N, M, H, X) in [(3, 16, 6, 12, 5), (2, 128, 20, 100, 9)]:
    ntm = NTM(input_size=X, output_size=X - 1, controller_size=H, N=N, M=M).double()
    with torch.no_grad():
        for prm in ntm.parameters():
            prm.copy_(torch.randn_like(prm) * 0.5)
    mem0 = torch.randn(B, N, M, dtype=torch.float64)
    h0 = torch.randn(B, H, dtype=torch.float64); c0 = torch.randn(B, H, dtype=torch.float64)
    rw0 = [torch.softmax(torch.randn(B, N, dtype=torch.float64), 1)]
    ww0 = [torch.softmax(torch.randn(B, N, dtype=torch.float64), 1)]
    r0 = [torch.randn(B, M, dtype=torch.float64)]
    x = torch.randn(B, X, dtype=torch.float64)
    with torch.no_grad():
        out, (mem1, (h1, c1), rw1, ww1, r1) = ntm(x, (mem0, (h0, c0), rw0, ww0, r0))
    errs = {k: 0.0 for k in ["h", "c", "read_w", "read_vec", "write_w", "memory", "output"]}
    errs_mirror = {"read_w": 0.0, "write_w": 0.0}
    rh, wh = ntm.read_heads[0], ntm.write_heads[0]
    for b in range(B):
        Mprev = mem0[b].numpy()
        hb, cb = lstm_cell(np.concatenate([x[b].numpy(), r0[0][b].numpy()]), h0[b].numpy(), c0[b].numpy(),
                           ntm.controller)
        raw_r = rh.fc.weight.detach().numpy() @ hb + rh.fc.bias.detach().numpy()
        raw_w = wh.fc.weight.detach().numpy() @ hb + wh.fc.bias.detach().numpy()
        R = ref_address(raw_r, rw0[0][b].numpy(), Mprev, M, True)
        W = ref_address(raw_w, ww0[0][b].numpy(), Mprev, M, False)
        R_m = ref_address(raw_r, rw0[0][b].numpy(), Mprev, M, True, shift_labels=(1, 0, -1))
        W_m = ref_address(raw_w, ww0[0][b].numpy(), Mprev, M, False, shift_labels=(1, 0, -1))
        r = sum(R["w"][i] * Mprev[i] for i in range(N))                       # Eq 2 (reads M_{t-1})
        Mnew = np.stack([Mprev[i] * (1 - W["w"][i] * W["erase"]) + W["w"][i] * W["add"]  # Eqs 3-4
                         for i in range(N)])
        o = ntm.fc_out.weight.detach().numpy() @ np.concatenate([hb, r]) + ntm.fc_out.bias.detach().numpy()
        cmp = {"h": (h1[b], hb), "c": (c1[b], cb), "read_w": (rw1[0][b], R["w"]), "read_vec": (r1[0][b], r),
               "write_w": (ww1[0][b], W["w"]), "memory": (mem1[b], Mnew), "output": (out[b], o)}
        for kk, (got, ref) in cmp.items():
            errs[kk] = max(errs[kk], float(np.abs(got.numpy() - ref).max()))
        errs_mirror["read_w"] = max(errs_mirror["read_w"], float(np.abs(rw1[0][b].numpy() - R_m["w"]).max()))
        errs_mirror["write_w"] = max(errs_mirror["write_w"], float(np.abs(ww1[0][b].numpy() - W_m["w"]).max()))
    print(f"[NTM step B={B} N={N} M={M} H={H}] max|delta| vs Eq 2-9 reference (shift labels -1,0,+1):")
    for kk, v in errs.items():
        print(f"    {kk:9s} {v:.2e}")
    print(f"    (same, with mirrored shift kernel +1,0,-1: read_w {errs_mirror['read_w']:.2e}, "
          f"write_w {errs_mirror['write_w']:.2e})")
    results.append(max(errs.values()))

# ---------------------------------------------------------------- 2. shift-direction probe
ntm = NTM(input_size=3, output_size=2, controller_size=4, N=10, M=4)
head = ntm.read_heads[0]
with torch.no_grad():
    head.fc.weight.zero_()
    bias = torch.zeros(head.addr_size)
    M = 4
    bias[M + 1] = -40.0                          # g ~ 0: ignore content, keep previous weighting
    bias[M + 2:M + 5] = torch.tensor([-40.0, -40.0, 40.0])   # s ~ [0, 0, 1] = "shift +1" per the comments
    bias[M + 5] = -40.0                          # gamma ~ 1
    head.fc.bias.copy_(bias)
prev = torch.zeros(1, 10); prev[0, 5] = 1.0
w, _, _ = head._addressing(torch.zeros(1, 4), prev, torch.randn(1, 10, 4))
dest = int(w[0].argmax())
print(f"[shift probe] s = [0,0,1] (documented as shift +1), focus at 5 -> {dest} "
      f"({'matches paper: +1 moves to next location' if dest == 6 else 'MIRRORED: moves to previous location'})")

# ---------------------------------------------------------------- 3. gradcheck of one step
ntm = NTM(input_size=3, output_size=2, controller_size=5, N=6, M=3).double()
B = 2
st = ntm.reset(B)
mem_in = torch.randn(B, 6, 3, dtype=torch.float64, requires_grad=True)
x_in = torch.randn(B, 3, dtype=torch.float64, requires_grad=True)


def step(xx, mm):
    out, (m2, _, rw, ww, _) = ntm(xx, (mm, st[1], st[2], st[3], st[4]))
    return out, m2


ok = torch.autograd.gradcheck(step, (x_in, mem_in), eps=1e-6, atol=1e-5)
print(f"[gradcheck] one NTM step w.r.t. input and memory: {ok}")

# ---------------------------------------------------------------- 4. Eq 1 under diffuse focus + large gamma
# beta ~ 0 makes the content weighting uniform over N=128; g ~ 1 uses it; s = no shift; gamma = 1+softplus(10) ~ 11.
ntm = NTM(input_size=3, output_size=2, controller_size=4, N=128, M=20)
head = ntm.read_heads[0]
with torch.no_grad():
    head.fc.weight.zero_()
    bias = torch.zeros(head.addr_size)
    M = 20
    bias[M] = -40.0                                   # beta = softplus(-40) ~ 0 -> uniform content weighting
    bias[M + 1] = 40.0                                # g ~ 1
    bias[M + 2:M + 5] = torch.tensor([-40.0, 40.0, -40.0])   # s = no shift
    bias[M + 5] = 10.0                                # gamma ~ 11
    head.fc.bias.copy_(bias)
w, _, _ = head._addressing(torch.zeros(1, 4), torch.zeros(1, 128), torch.randn(1, 128, 20))
wsum = float(w.detach().sum())
print(f"[Eq 1 probe] uniform focus over 128 slots, gamma ~ 11: sum(w) = {wsum:.6f} (must be 1)")

worst = max(results)
passed = worst < 1e-6 and dest == 6 and ok and abs(wsum - 1) < 1e-5
print(f"\nRESULT: {'PASS' if passed else 'FAIL'} "
      f"(worst max|delta| = {worst:.2e}; shift probe dest = {dest}; gradcheck = {ok}; sum(w) = {wsum:.6f})")
