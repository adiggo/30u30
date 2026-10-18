"""Day 20 reference check: Relational Memory Core (Santoro et al. 2018, arXiv 1806.01822v2).

What the paper specifies (Section 3):
  Eq 2   M~ = softmax( M W^q ([M; x] W^k)^T / sqrt(d_k) ) [M; x] W^v
         [M; x] = ROW-wise concatenation: the input is one extra row that memories can attend to;
         queries come from M only, so M~ has N rows (N x (N+1) attention matrix per head).
  Fig 1  MHDPA -> residual -> row-wise MLP g_psi -> residual (Sonnet: LayerNorm after each residual)
  Eq 7   m_t = sigma(f + b~_f) * m_{t-1} + sigma(i) * g_psi(m~_t)
What implementation.py does instead (checked numerically below):
  input projected and TILED onto every slot as extra FEATURES ([M_i ; x~] per row, N x N attention);
  residual + LayerNorm computed into `m_attended` but never used; no MLP; candidate = tanh(W_o attn).

Checks
  1. MHDPA inside RelationalMemory == F.scaled_dot_product_attention on the module's own q/k/v
     (head split, 1/sqrt(d) scaling, softmax, merge) and == independent NumPy float64 attention.
  2. One full RMC step == independent NumPy float64 re-implementation of the code's equations.
  3. Dead code: output and gradients do not depend on layer_norm_mem / layer_norm_output.
  4. Contrast with paper Eq 2: attention shape and the input-as-row formulation (tiny example).
  5. Nth-farthest generator labels vs independent recomputation + trivial baselines for the MSE target.

Usage: OMP_NUM_THREADS=1 python validation/day20_rmc.py [path/to/20_Relational_RNNs]
"""
import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "20_Relational_RNNs")
sys.path.insert(0, DAY)
from implementation import RelationalMemory, RelationalRNN  # noqa: E402
from train_minimal import generate_nth_farthest_batch  # noqa: E402

torch.manual_seed(0)
ok = True
B, N, Fm, X, H = 3, 4, 32, 16, 4
rmc = RelationalMemory(mem_slots=N, mem_size=Fm, input_size=X, num_heads=H).double()
x = torch.randn(B, X, dtype=torch.float64)
M0 = torch.randn(B, N, Fm, dtype=torch.float64)

# ---------------------------------------------------------------- 1. attention
with torch.no_grad():
    xt = rmc.input_projector(x).unsqueeze(1).expand(-1, N, -1)
    aug = torch.cat([M0, xt], -1)
    split = lambda t: t.view(B, N, H, Fm // H).transpose(1, 2)  # noqa: E731
    q, k, v = split(rmc.q_proj(aug)), split(rmc.k_proj(aug)), split(rmc.v_proj(aug))
    sdpa = F.scaled_dot_product_attention(q, k, v)                       # PyTorch built-in
    M1, probs = rmc(x, M0)
    # independent NumPy attention
    qn, kn, vn = q.numpy(), k.numpy(), v.numpy()
    sc = np.einsum("bhid,bhjd->bhij", qn, kn) / math.sqrt(Fm // H)      # (B,H,N,N) dot products
    pn = np.exp(sc - sc.max(-1, keepdims=True)); pn /= pn.sum(-1, keepdims=True)  # row softmax
    on = np.einsum("bhij,bhjd->bhid", pn, vn)                           # weighted sum of values
e_probs = np.abs(probs.numpy() - pn).max()
e_sdpa = (torch.matmul(probs, v) - sdpa).abs().max().item()
e_np = np.abs(sdpa.numpy() - on).max()
print(f"[MHDPA] attention probs vs NumPy softmax: {e_probs:.2e}; probs@V vs F.scaled_dot_product_attention: "
      f"{e_sdpa:.2e}; SDPA vs NumPy: {e_np:.2e}; probs shape {tuple(probs.shape)} (N x N, input not a key)")
ok &= max(e_probs, e_sdpa, e_np) < 1e-12


# ---------------------------------------------------------------- 2. full step vs NumPy
def W(lin):
    return lin.weight.detach().numpy(), lin.bias.detach().numpy()


def sig(z):
    return 1 / (1 + np.exp(-z))


Mn = M0.numpy()
Wp, bp = W(rmc.input_projector)
xt_n = x.numpy() @ Wp.T + bp                                  # (B, F) projected input
aug_n = np.concatenate([Mn, np.repeat(xt_n[:, None, :], N, axis=1)], -1)   # (B, N, 2F): tile over slots
heads = []
for name in ["q_proj", "k_proj", "v_proj"]:
    Wl, bl = W(getattr(rmc, name))
    heads.append((aug_n @ Wl.T + bl).reshape(B, N, H, Fm // H).transpose(0, 2, 1, 3))  # (B,H,N,d)
qq, kk, vv = heads
s = qq @ kk.transpose(0, 1, 3, 2) / math.sqrt(Fm // H)
p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
att = (p @ vv).transpose(0, 2, 1, 3).reshape(B, N, Fm)          # merge heads back to (B, N, F)
Wo, bo = W(rmc.output_proj)
out = att @ Wo.T + bo
Wf, bf = W(rmc.gate_forget); Wi, bi = W(rmc.gate_input)
f = sig(aug_n @ Wf.T + bf); i = sig(aug_n @ Wi.T + bi)
M1_ref = f * Mn + i * np.tanh(out)
e_step = np.abs(M1.detach().numpy() - M1_ref).max()
print(f"[RMC step] next memory vs NumPy re-implementation of the code's equations: max|delta| = {e_step:.2e}")
ok &= e_step < 1e-12
print(f"[RMC step] forget-gate bias initialised to 1.0: {bool(torch.all(rmc.gate_forget.bias == 1.0))}")

# ---------------------------------------------------------------- 3. dead residual + LayerNorm
rnn = RelationalRNN(input_size=X, mem_slots=N, mem_size=Fm, output_size=X, num_heads=H)
seq = torch.randn(2, 5, X)
y = rnn(seq).sum()
y.backward()
g_ln = [prm.grad for prm in rnn.rmc.layer_norm_mem.parameters()] + \
       [prm.grad for prm in rnn.rmc.layer_norm_output.parameters()]
dead = all(g is None or float(g.abs().max()) == 0.0 for g in g_ln)
with torch.no_grad():
    before = rnn(seq)
    rnn.rmc.layer_norm_mem.weight.mul_(7.0); rnn.rmc.layer_norm_mem.bias.add_(3.0)
    after = rnn(seq)
print(f"[dead code] layer_norm_mem/_output receive no gradient: {dead}; output change after scaling LN params: "
      f"{(after - before).abs().max().item():.1e}  -> the residual+LayerNorm `m_attended` is computed and discarded")

# ---------------------------------------------------------------- 4. paper Eq 2 on a tiny example
# 2 memory rows + 1 input row, 1 head, identity projections, F = 2: keys/values include the input row.
Mt = np.array([[1.0, 0.0], [0.0, 1.0]]); xr = np.array([[2.0, 0.0]])
KV = np.vstack([Mt, xr])                                    # [M; x] row-wise: 3 x 2
S = Mt @ KV.T / math.sqrt(2)                                 # queries from M only: 2 x 3
P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
Mtil = P @ KV
print(f"[paper Eq 2 example] attention matrix {P.shape} (N x (N+1)):\n{np.round(P, 4)}\n  M~ =\n{np.round(Mtil, 4)}")

# ---------------------------------------------------------------- 5. data + baselines
torch.manual_seed(1)
inp, tgt = generate_nth_farthest_batch(2000, 15, 16)
ref = torch.stack([s[torch.argmax(((s - s[0]) ** 2).sum(1))] for s in inp])
lab_err = (ref - tgt).abs().max().item()
mse_zero = (tgt ** 2).mean().item()
# best "-c * first vector" predictor (closed form least squares over c)
v0 = inp[:, 0]; c = -(tgt * v0).sum() / (v0 * v0).sum()
mse_neg = ((tgt + c * v0) ** 2).mean().item()
print(f"[data] targets vs independent argmax-distance recomputation: max|delta| = {lab_err:.1e}; "
      f"MSE baselines: predict 0 -> {mse_zero:.3f}; predict -{c:.2f}*v0 -> {mse_neg:.3f}")
ok &= lab_err == 0.0

print(f"\nRESULT: {'PASS' if ok else 'FAIL'} (code matches its own equations and PyTorch SDPA; "
      f"architecture deviates from paper Eq 2 / Fig 1 as listed above)")
