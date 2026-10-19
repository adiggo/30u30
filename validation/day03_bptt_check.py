"""Day 03: check LSTMLanguageModel's hand-written BPTT (papers/03_RNN_Regularization/implementation.py).

The model is the paper's (Zaremba et al. 2014, Sec. 3.2 and 4.1): word embedding -> 2 LSTM layers ->
softmax, inverted dropout on every non-recurrent connection (embedding -> layer 1, layer 1 -> layer 2,
layer 2 -> softmax) and never on h_{t-1} -> h_t; loss = (1/B) * summed cross-entropy.

Checks (float64, tiny config, the SAME dropout masks given to NumPy and PyTorch):
 1. Forward: loss and final (h, c) of both layers vs an independent PyTorch model built from one
    nn.LSTM per layer (so the masks can be applied between layers). Weights are copied with the gate
    blocks reordered: ours (i, f, o, g) as in the paper, torch's (i, f, g, o).
 2. Backward: every parameter's gradient, and dL/d(h0, c0) of each layer, vs torch.autograd.
    Relative error = max|numpy - torch| / max|torch| per tensor; tolerance 1e-6.
 3. Fourth-order central finite differences (eps = 1e-3) on the 3 largest-gradient and 3 random entries of every
    parameter; error relative to max(|a|, |n|, 1% of the tensor's largest gradient); tolerance 1e-6.
 4. Dropout placement: a layer's recurrent trajectory does not depend on the masks above it, and the
    state carried to the next window is never dropped.
 5. Truncated-BPTT bookkeeping: two windows with the state carried across give the same loss as
    one window of twice the length.
 6. clip_grad_norm is global-norm clipping (Sec. 4.1): norm 10 -> 5 with the direction kept; norm 3
    unchanged.
 7. Smoke test: 30 passes of SGD (lr 1, clip 5) over a short repeating word sequence drive the loss
    toward 0.

Run:  OMP_NUM_THREADS=1 python validation/day03_bptt_check.py      (about 5 s)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '03_RNN_Regularization')
sys.path.insert(0, DAY)
from implementation import LSTMLanguageModel, clip_grad_norm  # noqa: E402

torch.set_default_dtype(torch.float64)
TOL = 1e-6
results = []


def report(name, ok, detail):
    results.append(ok)
    print('%-4s %s: %s' % ('PASS' if ok else 'FAIL', name, detail))


# ---- tiny config ------------------------------------------------------------------------------
V, H, L, T, B, KEEP = 13, 6, 2, 7, 3, 0.6
model = LSTMLanguageModel(V, H, num_layers=L, keep_prob=KEEP, init_scale=0.5, dtype=np.float64, seed=1)
rng = np.random.default_rng(2)
x = rng.integers(0, V, size=(T, B))
y = rng.integers(0, V, size=(T, B))
x[3, 1] = x[0, 1]                              # a repeated word id: the embedding gradient must accumulate
state0 = [(rng.normal(0, 0.5, (B, H)), rng.normal(0, 0.5, (B, H))) for _ in range(L)]  # non-zero carried state
masks = model.sample_masks(T, B)               # L+1 masks, entries 0 or 1/KEEP

loss_np, state_np, cache = model.forward(x, y, state0, masks=masks, training=True)
grads_np, dstate_np = model.backward(cache)

# ---- PyTorch reference -------------------------------------------------------------------------
# torch.nn.LSTM stacks its gates as (i, f, g, o); ours are (i, f, o, g). Block k of torch = block PERM[k]
# of ours. PERM swaps blocks 2 and 3, so it is its own inverse (the same map converts gradients back).
PERM = [0, 1, 3, 2]


def to_torch_gates(W):       # ours (H_in, 4H) or (4H,) -> torch (4H, H_in) or (4H,)
    blocks = [W[..., k * H:(k + 1) * H] for k in PERM]
    return np.concatenate(blocks, axis=-1).T


def from_torch_gates(Wt):    # torch (4H, H_in) or (4H,) -> ours
    Wt = Wt.T
    return np.concatenate([Wt[..., k * H:(k + 1) * H] for k in PERM], axis=-1)


P = model.params
emb = nn.Embedding(V, H)
lstms = [nn.LSTM(H, H) for _ in range(L)]      # one single-layer LSTM per layer; input (T, B, H)
out_layer = nn.Linear(H, V)
with torch.no_grad():
    emb.weight.copy_(torch.from_numpy(P['embed']))
    for l, lstm in enumerate(lstms, start=1):
        lstm.weight_ih_l0.copy_(torch.from_numpy(to_torch_gates(P['W_x%d' % l])))
        lstm.weight_hh_l0.copy_(torch.from_numpy(to_torch_gates(P['W_h%d' % l])))
        lstm.bias_ih_l0.copy_(torch.from_numpy(to_torch_gates(P['b%d' % l])))
        lstm.bias_hh_l0.zero_()                # torch has two biases per layer; ours is their sum
    out_layer.weight.copy_(torch.from_numpy(P['W_y'].T))
    out_layer.bias.copy_(torch.from_numpy(P['b_y']))
h0s = [torch.tensor(h, requires_grad=True) for h, _ in state0]
c0s = [torch.tensor(c, requires_grad=True) for _, c in state0]
tm = [torch.from_numpy(m) for m in masks]

inp = emb(torch.from_numpy(x)) * tm[0]                 # D(embedding)
final_t = []
for l, lstm in enumerate(lstms):
    out, (hT, cT) = lstm(inp, (h0s[l][None], c0s[l][None]))   # recurrent h, c pass through un-dropped
    final_t.append((hT[0], cT[0]))
    inp = out * tm[l + 1]                              # D(h^l) into the next layer / the softmax
logits = out_layer(inp.reshape(T * B, H))
loss_t = F.cross_entropy(logits, torch.from_numpy(y.reshape(-1)), reduction='sum') / B
loss_t.backward()

# 1. forward
d_state = max(max(np.abs(state_np[l][0] - final_t[l][0].detach().numpy()).max(),
                  np.abs(state_np[l][1] - final_t[l][1].detach().numpy()).max()) for l in range(L))
d_loss = abs(loss_np - loss_t.item())
report('1. forward vs nn.LSTM reference', d_loss < 1e-12 and d_state < 1e-12,
       'loss %.12f vs %.12f (|d| %.1e); final (h, c) max|d| %.1e' % (loss_np, loss_t.item(), d_loss, d_state))

# 2. gradients vs autograd
ref = {'embed': emb.weight.grad.numpy(), 'W_y': out_layer.weight.grad.numpy().T,
       'b_y': out_layer.bias.grad.numpy()}
for l, lstm in enumerate(lstms, start=1):
    ref['W_x%d' % l] = from_torch_gates(lstm.weight_ih_l0.grad.numpy())
    ref['W_h%d' % l] = from_torch_gates(lstm.weight_hh_l0.grad.numpy())
    ref['b%d' % l] = from_torch_gates(lstm.bias_ih_l0.grad.numpy())
for l in range(L):
    ref['dh0_layer%d' % (l + 1)] = h0s[l].grad.numpy()
    ref['dc0_layer%d' % (l + 1)] = c0s[l].grad.numpy()
    grads_np['dh0_layer%d' % (l + 1)] = dstate_np[l][0]
    grads_np['dc0_layer%d' % (l + 1)] = dstate_np[l][1]
worst = 0.0
for k in ref:
    rel = np.abs(grads_np[k] - ref[k]).max() / max(np.abs(ref[k]).max(), 1e-300)
    worst = max(worst, rel)
    print('       %-11s shape %-8s max|grad| %.3e   rel. error vs autograd %.1e'
          % (k, 'x'.join(map(str, ref[k].shape)), np.abs(ref[k]).max(), rel))
report('2. BPTT gradients vs torch.autograd (%d tensors)' % len(ref), worst < TOL,
       'max relative error %.1e (tol %.0e)' % (worst, TOL))

# 3. finite differences: per parameter, its 3 largest-gradient entries and 3 random ones.
# The plain 2-point difference (L(+e) - L(-e)) / 2e has float64 round-off ~ 1e-16 * |loss| / e ~ 2e-10
# at e = 1e-5 here, too close to a 1e-6 check on small entries. The 4th-order central difference
#   L'(p) ~ [8 (L(p+e) - L(p-e)) - (L(p+2e) - L(p-2e))] / (12 e),   error O(e^4),
# allows e = 1e-3, cutting the round-off 100x with negligible truncation error. Entries whose own
# gradient is tiny are judged against 1% of that tensor's largest gradient (the denominator's floor).
eps, worst_fd, worst_abs, n_fd = 1e-3, 0.0, 0.0, 0


def loss_at(k, idx, value):
    old = model.params[k][idx]
    model.params[k][idx] = value
    out = model.forward(x, y, state0, masks=masks, training=True)[0]
    model.params[k][idx] = old
    return out


for k, p in model.params.items():
    g = grads_np[k]
    top3 = np.argsort(np.abs(g).ravel())[-3:]                  # flat indices of the 3 largest |grad|
    picks = np.concatenate([top3, rng.choice(p.size, size=min(3, p.size), replace=False)])
    for flat in picks:
        idx = np.unravel_index(flat, p.shape)                  # flat index -> (row, col) tuple
        v = p[idx]
        num = (8 * (loss_at(k, idx, v + eps) - loss_at(k, idx, v - eps))
               - (loss_at(k, idx, v + 2 * eps) - loss_at(k, idx, v - 2 * eps))) / (12 * eps)
        ana = g[idx]
        worst_fd = max(worst_fd, abs(num - ana) / max(abs(num), abs(ana), 0.01 * np.abs(g).max()))
        worst_abs = max(worst_abs, abs(num - ana))
        n_fd += 1
report('3. finite differences (%d entries over all %d parameters)' % (n_fd, len(model.params)), worst_fd < TOL,
       'max relative error %.1e (|a - n| / max(|a|, |n|, 1%% of the tensor max); tol %.0e); max |a - n| %.1e'
       % (worst_fd, TOL, worst_abs))

# 4. dropout placement
other = model.sample_masks(T, B)


def final_states(ms):
    return model.forward(x, y, state0, masks=ms, training=True)[1]


sA = final_states(masks)
sB = final_states([masks[0], other[1], other[2]])          # change the masks above layer 1
sC = final_states([masks[0], masks[1], other[2]])          # change only the top -> softmax mask
d1 = max(np.abs(sA[0][0] - sB[0][0]).max(), np.abs(sA[0][1] - sB[0][1]).max())
d2 = max(np.abs(sA[1][0] - sC[1][0]).max(), np.abs(sA[1][1] - sC[1][1]).max())
_, _, cA = model.forward(x, y, state0, masks=masks, training=True)
carried_is_clean = np.array_equal(sA[1][0], cA['layers'][1]['Hs'][-1])  # carried h = undropped h_T
report('4. dropout placement (recurrent paths untouched)', d1 == 0 and d2 == 0 and carried_is_clean,
       'layer-1 (h_T, c_T) vs masks above it: max|d| = %g; layer-2 vs top mask: %g; '
       'carried h_T == undropped h_T: %s' % (d1, d2, carried_is_clean))

# 5. truncated BPTT bookkeeping: carrying the state across windows = one long window (forward)
x2 = rng.integers(0, V, size=(2 * T, B))
y2 = rng.integers(0, V, size=(2 * T, B))
m2 = model.sample_masks(2 * T, B)
l_long = model.forward(x2, y2, state0, masks=m2, training=True)[0]
la, st, _ = model.forward(x2[:T], y2[:T], state0, masks=[m[:T] for m in m2], training=True)
lb, _, _ = model.forward(x2[T:], y2[T:], st, masks=[m[T:] for m in m2], training=True)
d5 = abs(l_long - (la + lb)) / abs(l_long)
report('5. state carried across windows', d5 < 1e-13,
       'loss(2T window) %.10f vs loss(T) + loss(T) %.10f, relative |d| %.1e' % (l_long, la + lb, d5))

# 6. global-norm clipping
g = {'a': np.array([3.0, 4.0]), 'b': np.array([[0.0, 0.0], [0.0, 0.0]])}
g['b'][0, 0] = np.sqrt(100 - 25)                            # total norm = sqrt(25 + 75) = 10
before = {k: v.copy() for k, v in g.items()}
n = clip_grad_norm(g, 5.0)
after = np.sqrt(sum(np.sum(v * v) for v in g.values()))
same_dir = all(np.allclose(g[k], before[k] * 0.5) for k in g)
g2 = {'a': np.array([3.0, 0.0])}
n2 = clip_grad_norm(g2, 5.0)
report('6. clip_grad_norm (global norm, Sec. 4.1)',
       abs(n - 10) < 1e-12 and abs(after - 5) < 1e-12 and same_dir and np.array_equal(g2['a'], [3.0, 0.0]),
       'norm 10 -> %.6f (all tensors scaled by 0.5: %s); norm %.1f left unchanged' % (after, same_dir, n2))

# 7. smoke test: the model can fit a short repeating sequence
lm = LSTMLanguageModel(V, 16, num_layers=2, keep_prob=1.0, init_scale=0.1, dtype=np.float64, seed=3)
stream = np.tile(np.arange(5), 40)                          # 0 1 2 3 4 0 1 2 ... (perfectly predictable)
data = stream[:len(stream) // 2 * 2].reshape(2, -1).T      # 2 streams side by side, (rows, 2)
first = last = None
for epoch in range(30):
    s = lm.init_state(2)
    tot = cnt = 0
    for i in range(0, data.shape[0] - 1, 10):              # 10-step windows, state carried across them
        yy = data[i + 1:i + 11]
        xx = data[i:i + len(yy)]
        loss, s, c = lm.forward(xx, yy, s, training=True)
        gr, _ = lm.backward(c)
        clip_grad_norm(gr, 5.0)
        lm.sgd_update(gr, 1.0)
        tot += loss * 2                                     # loss is (1/B) * sum, B = 2 -> summed NLL
        cnt += xx.size
    first = tot / cnt if first is None else first
    last = tot / cnt                                        # mean cross-entropy per word in this pass
report('7. training smoke test (repeating sequence, lr 1, clip 5)', last < 0.05 * first,
       'mean cross-entropy %.3f -> %.4f nats/word over 30 passes (uniform guess: ln 13 = %.3f)'
       % (first, last, np.log(V)))

print('\n%d/%d checks pass' % (sum(results), len(results)))
sys.exit(0)
