"""Day 20 check: the paper-faithful Relational Memory Core and the labelled Nth Farthest task.

Reference: Santoro et al. 2018 (arXiv 1806.01822v2) Sec. 3 / Fig. 1 / Eqs. 1-9 and App. A.1, plus DeepMind
Sonnet v1 relational_memory.py and examples/dataset_nth_farthest.py for details the paper leaves out.

Checks
  1. Shapes: attention is (B, H, N, N+1) - queries from the N memory rows, keys/values from [M; x~] (Eq. 2) -
     with rows summing to 1; RelationalRNN returns (B, T, H, N, N+1); every parameter receives a nonzero
     gradient (the old cell's residual + LayerNorm received none).
  2. One step vs an independent float64 re-implementation that loops explicitly over batch, heads, query
     slots and key rows (NumPy only for the small linear maps; no torch.nn, no module methods), for the
     module in float64 and in float32.
  3. Paper Eq. 2 worked example (2 slots + 1 input row, identity projections, 1 head, no qkv LayerNorm).
  4. Slot-permutation equivariance: step(P M) = P step(M); the attention permutes the same way and the
     input column stays last.
  5. Gate limits: forget gate forced to 1 and input gate to 0 -> M_t = M_{t-1}; the reverse -> M_t = tanh(G)
     (G = attention + MLP block output); 'memory' gating gives one scalar gate per row.
  6. Gradients in float64: autograd vs central finite differences for every parameter tensor of a small
     RelationalRNN (max relative error per tensor), and torch.autograd.gradcheck on the step's inputs.
  7. Nth Farthest generator vs an independent decode-and-sort of its own inputs (paper size 8 x 16, the
     reduced default 3 x 2, and the curriculum's fixed_n=0 batches); label / question structure; accuracy of
     "answer m when n = K-1" = 2/K.

Usage: OMP_NUM_THREADS=1 python validation/day20_rmc_paper_check.py [path/to/20_Relational_RNNs]
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


def report(name, value, tol, fmt="{:.1e}"):
    global ok
    passed = value <= tol
    ok &= passed
    print(f"  {'PASS' if passed else 'FAIL'}  {name}: {fmt.format(value)} (tol {tol:g})")


def perturbed(module, scale=0.3):
    """Shift every parameter (LayerNorm gains/biases and zero-initialised gate biases included) so that
    no term of the update can hide behind an identity or a zero."""
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))
    return module


B, N, Fm, H, X = 2, 3, 8, 2, 5
rmc = perturbed(RelationalMemory(N, Fm, X, num_heads=H).double())
x = torch.randn(B, X, dtype=torch.float64)
M = torch.randn(B, N, Fm, dtype=torch.float64)

# ---------------------------------------------------------------- 1. shapes, no dead parameters
print("1. shapes and gradient flow")
M1, W = rmc(x, M)
print(f"  attention {tuple(W.shape)} = (B, H, N, N+1); next memory {tuple(M1.shape)}")
report("attention rows sum to 1, max |sum - 1|", (W.sum(-1) - 1).abs().max().item(), 1e-12)
ok &= W.shape == (B, H, N, N + 1) and M1.shape == (B, N, Fm)
rnn = RelationalRNN(X, N, Fm, 4, num_heads=H, head_hidden=(6,))
out, att = rnn(torch.randn(B, 3, X), return_attention=True)
print(f"  RelationalRNN output {tuple(out.shape)}, attention history {tuple(att.shape)} = (B, T, H, N, N+1)")
ok &= att.shape == (B, 3, H, N, N + 1)
F.cross_entropy(out, torch.tensor([0, 1])).backward()
dead = [n for n, p in rnn.named_parameters() if p.grad is None or p.grad.abs().max().item() == 0.0]
print(f"  parameters without gradient: {dead or 'none'} ({sum(p.numel() for p in rnn.parameters())} parameters)")
ok &= not dead


# ---------------------------------------------------------------- 2. explicit-loop re-implementation
def lin(layer, v):
    """v (..., in) @ W^T (in, out) + b -> (..., out): one nn.Linear, written out."""
    return v @ layer.weight.detach().double().numpy().T + layer.bias.detach().double().numpy()


def layer_norm(layer, v, eps=1e-5):
    """LayerNorm over the last axis: (v - mean) / sqrt(var + eps) * gain + bias (biased variance)."""
    mu = v.mean(-1, keepdims=True)                  # keepdims keeps the axis so (..., F) - (..., 1) broadcasts
    var = ((v - mu) ** 2).mean(-1, keepdims=True)
    return (v - mu) / np.sqrt(var + eps) * layer.weight.detach().double().numpy() + layer.bias.detach().double().numpy()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def reference_step(core, x_in, mem):
    """Paper Eq. 2 + Fig. 1a + Sonnet gates, looping over batch b, head h, query slot i and key row j."""
    xn, Mn = x_in.detach().double().numpy(), mem.detach().double().numpy()
    Bn, Nn, Fn = Mn.shape
    Hn, dk, dv = core.num_heads, core.key_size, core.head_size
    qkv_ln = core.qkv_norm[0]
    mlp1, mlp2 = core.mlp[0], core.mlp[2]
    new_M = np.zeros_like(Mn)
    probs = np.zeros((Bn, Hn, Nn, Nn + 1))
    for b in range(Bn):
        x_t = lin(core.input_projector, xn[b])                       # (F,) projected input row x~
        rows = [Mn[b, i] for i in range(Nn)] + [x_t]                  # [M; x~]: N + 1 rows
        qkv = [layer_norm(qkv_ln, lin(core.qkv[0], r)) for r in rows]   # per row: [q_h | k_h | v_h] for each head
        attended = np.zeros((Nn, Hn * dv))
        for h in range(Hn):
            o = h * (2 * dk + dv)                                      # start of head h's block
            for i in range(Nn):                                        # queries: memory rows only (Eq. 2)
                q = qkv[i][o:o + dk]
                scores = [float(q @ qkv[j][o + dk:o + 2 * dk]) / math.sqrt(dk) for j in range(Nn + 1)]
                top = max(scores)
                e = [math.exp(s - top) for s in scores]
                w = [v / sum(e) for v in e]                            # softmax over the N + 1 keys
                probs[b, h, i] = w
                attended[i, h * dv:(h + 1) * dv] = sum(w[j] * qkv[j][o + 2 * dk:o + 2 * dk + dv] for j in range(Nn + 1))
        for i in range(Nn):
            m1 = layer_norm(core.norm_attention[0], Mn[b, i] + attended[i])        # residual + LN
            g = layer_norm(core.norm_mlp[0], m1 + lin(mlp2, np.maximum(lin(mlp1, m1), 0.0)))   # MLP + residual + LN
            gates = lin(core.gate_memory_proj, np.tanh(Mn[b, i])) + lin(core.gate_input_proj, x_t)
            i_gate = sigmoid(gates[:Fn] + core.input_bias)            # first half: input gate
            f_gate = sigmoid(gates[Fn:] + core.forget_bias)           # second half: forget gate (+1)
            new_M[b, i] = i_gate * np.tanh(g) + f_gate * Mn[b, i]     # Eq. 7 with Sonnet's tanh
    return new_M, probs


print("2. one step vs an explicit-loop float64 re-implementation")
ref_M, ref_P = reference_step(rmc, x, M)
report("float64 module: next memory max|delta|", np.abs(M1.detach().numpy() - ref_M).max(), 1e-12)
report("float64 module: attention max|delta|", np.abs(W.detach().numpy() - ref_P).max(), 1e-12)
rmc32 = RelationalMemory(N, Fm, X, num_heads=H)
rmc32.load_state_dict({k: v.float() for k, v in rmc.state_dict().items()})
M1_32, W_32 = rmc32(x.float(), M.float())
report("float32 module: next memory max|delta|", np.abs(M1_32.detach().double().numpy() - ref_M).max(), 1e-5)
report("float32 module: attention max|delta|", np.abs(W_32.detach().double().numpy() - ref_P).max(), 1e-5)

# ---------------------------------------------------------------- 3. worked example (paper Eq. 2)
print("3. paper Eq. 2 worked example: M = [[1,0],[0,1]], x = [2,0], identity projections, 1 head")
ex = RelationalMemory(2, 2, 2, num_heads=1, qkv_layer_norm=False).double()
with torch.no_grad():
    ex.input_projector.weight.copy_(torch.eye(2)); ex.input_projector.bias.zero_()
    ex.qkv[0].weight.copy_(torch.cat([torch.eye(2)] * 3)); ex.qkv[0].bias.zero_()    # W^q = W^k = W^v = I
    Mex = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    attended_ex, P_ex = ex.multihead_attention(Mex, ex.project_input(torch.tensor([[2.0, 0.0]], dtype=torch.float64)))
KV = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])                    # [M; x] row-wise
S = KV[:2] @ KV.T / math.sqrt(2)                                         # (2, 3) scores, queries from M only
P_hand = np.exp(S) / np.exp(S).sum(1, keepdims=True)
print(f"  attention (2 x 3):\n{np.round(P_ex[0, 0].numpy(), 4)}\n  M~ = \n{np.round(attended_ex[0].numpy(), 4)}")
report("vs closed form, attention and M~ max|delta|",
       max(np.abs(P_ex[0, 0].numpy() - P_hand).max(), np.abs(attended_ex[0].numpy() - P_hand @ KV).max()), 1e-12)

# ---------------------------------------------------------------- 4. slot permutation equivariance
print("4. permuting memory slots permutes the step's outputs")
perm = torch.tensor([2, 0, 1])
Mp, Wp = rmc(x, M[:, perm])
cols = torch.cat([perm, torch.tensor([N])])                              # memory columns permute, input column stays last
report("next memory: step(M[perm]) vs step(M)[perm]", (Mp - M1[:, perm]).abs().max().item(), 1e-12)
report("attention: rows and memory columns permute, input column fixed", (Wp - W[:, :, perm][..., cols]).abs().max().item(), 1e-12)

# ---------------------------------------------------------------- 5. gate limits
print("5. gate limits")
G, _ = rmc.attend_over_memory(M, rmc.project_input(x))
rmc.forget_bias, rmc.input_bias = 1e4, -1e4                              # sigmoid -> exactly 1 and 0 in float64
report("forget = 1, input = 0: M_t vs M_{t-1}", (rmc(x, M)[0] - M).abs().max().item(), 0.0)
rmc.forget_bias, rmc.input_bias = -1e4, 1e4
report("forget = 0, input = 1: M_t vs tanh(G)", (rmc(x, M)[0] - torch.tanh(G)).abs().max().item(), 0.0)
rmc.forget_bias, rmc.input_bias = 1.0, 0.0
mem_core = RelationalMemory(N, Fm, X, num_heads=H, gate_style='memory').double()
gi, gf = mem_core.create_gates(mem_core.project_input(x), M)
print(f"  'memory' gating: gate shapes {tuple(gi.shape)}, {tuple(gf.shape)} (one scalar per row)")
ok &= gi.shape == (B, N, 1) and gf.shape == (B, N, 1)
fresh = RelationalMemory(N, Fm, X, num_heads=H)
print(f"  fresh core: gate biases zero, forget gate = sigmoid(1 + ...) -> mean "
      f"{fresh.create_gates(fresh.project_input(x.float()), M.float())[1].mean().item():.3f} (sigmoid(1) = 0.731)")

# ---------------------------------------------------------------- 6. gradients
print("6. autograd vs central finite differences (float64), every parameter tensor")
torch.manual_seed(1)
net = perturbed(RelationalRNN(X, N, Fm, 4, num_heads=H, head_hidden=(6,)).double(), 0.2)
seq = torch.randn(B, 3, X, dtype=torch.float64)
target = torch.tensor([1, 3])


def loss_fn():
    return F.cross_entropy(net(seq), target)


net.zero_grad()
loss_fn().backward()
eps, worst, worst_name = 1e-6, 0.0, ""
with torch.no_grad():
    for name, p in net.named_parameters():
        fd = torch.zeros_like(p).view(-1)
        flat = p.view(-1)
        for k in range(flat.numel()):
            old = flat[k].item()
            flat[k] = old + eps; lp = loss_fn().item()
            flat[k] = old - eps; lm = loss_fn().item()
            flat[k] = old
            fd[k] = (lp - lm) / (2 * eps)                                  # central difference
        rel = ((p.grad.view(-1) - fd).norm() / max(p.grad.norm().item(), fd.norm().item(), 1e-30)).item()
        if rel > worst:
            worst, worst_name = rel, name
n_par = sum(p.numel() for p in net.parameters())
report(f"max relative error over {n_par} parameters ({worst_name})", worst, 1e-6)
step_fn = lambda xx, mm: rmc(xx, mm)[0]                                    # noqa: E731
gc = torch.autograd.gradcheck(step_fn, (x.clone().requires_grad_(), M.clone().requires_grad_()), eps=1e-6, atol=1e-8)
print(f"  {'PASS' if gc else 'FAIL'}  torch.autograd.gradcheck of one step w.r.t. (x, M): {gc}")
ok &= gc

# ---------------------------------------------------------------- 7. task generator
print("7. Nth Farthest generator vs decode-and-sort of its own inputs")
for K, D, fixed_n in ((8, 16, None), (3, 2, None), (3, 2, 0)):
    g = torch.Generator().manual_seed(K * 100 + D)
    inp, tgt = generate_nth_farthest_batch(2000, K, D, g, fixed_n=fixed_n)
    bad = 0
    structure = True
    for b in range(inp.size(0)):
        seq_b = inp[b].double().numpy()
        vecs, lab = seq_b[:, :D], seq_b[:, D:D + K].argmax(1)
        n_b, m_b = seq_b[0, D + K:D + 2 * K].argmax(), seq_b[0, D + 2 * K:].argmax()
        structure &= sorted(lab.tolist()) == list(range(K))                         # labels: a permutation
        structure &= bool((seq_b[:, D + K:] == seq_b[0, D + K:]).all())             # n, m repeated every step
        structure &= bool((np.abs(seq_b[:, D:]).sum(1) == 3).all())                 # three one-hots per step
        ref = vecs[list(lab).index(m_b)]
        dists = [math.sqrt(sum((a - c) ** 2 for a, c in zip(v, ref))) for v in vecs]
        ranked = sorted(range(K), key=lambda t: -dists[t])                           # farthest first
        bad += int(lab[ranked[n_b]] != tgt[b].item())
    n_all = inp[:, 0, D + K:D + 2 * K].argmax(-1)
    m_all = inp[:, 0, D + 2 * K:].argmax(-1)
    copy_m = (n_all == K - 1)
    shortcut_acc = (copy_m.float() + (~copy_m).float() / (K - 1)).mean().item()    # expected accuracy, copy m when n = K-1
    if fixed_n is None:
        print(f"  K={K}, D={D}: input width {inp.size(-1)}; target mismatches {bad}/2000; structure ok {structure}; "
              f"n = K-1 -> target = m in {(tgt[copy_m] == m_all[copy_m]).float().mean().item():.0%} of cases; "
              f"'copy m' shortcut accuracy {shortcut_acc:.3f} (2/K = {2 / K:.3f})")
    else:
        structure &= bool((n_all == fixed_n).all())
        print(f"  K={K}, D={D}, fixed_n={fixed_n} (curriculum batches): target mismatches {bad}/2000; "
              f"every question asks for rank {fixed_n}: {structure}")
    ok &= bad == 0 and structure and bool((tgt[copy_m] == m_all[copy_m]).all())

print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
