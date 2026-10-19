"""Day 13 -- the NumPy Transformer's hand-written backward pass vs torch.autograd and finite differences.

Everything in float64. "rel err" of a tensor = max|ours - reference| / max|reference|.

  1. softmax_backward                          vs autograd
  2. ScaledDotProductAttention.backward        vs autograd (padding mask, causal mask): dQ, dK, dV
  3. MultiHeadAttention.backward               vs autograd, cross-attention (Lq != Lk) with key padding:
                                                  d_query, d_key, d_value, dW_Q/K/V/O
  4. LayerNorm.backward, FeedForward.backward  vs autograd (random gamma/beta and biases)
  5. cross_entropy_loss_and_grad               vs F.cross_entropy(ignore_index, label_smoothing): loss, dlogits
  6. Full Transformer (2+2 layers, d_model 16, 2 heads, d_ff 32, vocab 12, T 7, batch 3; padded source,
     padded targets, causal mask, dropout off): EVERY parameter gradient vs the same weights loaded into
     nn.TransformerEncoderLayer / nn.TransformerDecoderLayer + autograd. Untied, and tied (Sec. 3.4).
  7. Central finite differences on sampled entries of every parameter tensor: dropout off, and dropout ON
     with the masks held fixed by reseeding (checks that backward reuses the forward masks).
  8. train_minimal.Adam (0.9, 0.98, 1e-9, Eq. 3 schedule) vs torch.optim.Adam over 6 steps.

Run:  OMP_NUM_THREADS=1 python validation/day13_backprop_check.py
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "13_attention")
sys.path.insert(0, DAY)
import implementation as imp  # noqa: E402
import train_minimal as tm  # noqa: E402

TOL = 1e-6          # max relative error accepted (the spec); float64 agreement is typically ~1e-14
RESULTS = []


def rel_err(ours, ref):
    ours, ref = np.asarray(ours, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    return float(np.abs(ours - ref).max() / max(np.abs(ref).max(), 1e-300))


def report(name, err, tol=TOL, extra=""):
    ok = err <= tol
    RESULTS.append((name, ok, err))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: rel err {err:.2e} (tol {tol:g}){extra}")
    return ok


def t(x, grad=False):
    return torch.tensor(np.asarray(x, dtype=np.float64), requires_grad=grad)


rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. softmax backward
s = rng.normal(size=(2, 3, 5, 6)) * 2
g = rng.normal(size=s.shape)
s_t = t(s, True)
(torch.softmax(s_t, -1) * t(g)).sum().backward()
report("softmax_backward vs autograd", rel_err(imp.softmax_backward(imp.softmax(s), g), s_t.grad))

# ---------------------------------------------------------------- 2. scaled dot-product attention
B, H, Lq, Lk, dk = 2, 3, 5, 6, 4
attn = imp.ScaledDotProductAttention(dropout_p=0.0)
for label, Lq_, mask in [
    ("padding mask (B,1,1,Lk)", Lq, np.array([[False] * 6, [False] * 4 + [True] * 2])[:, None, None, :]),
    ("causal mask (1,1,L,L)", Lk, imp.create_causal_mask(Lk)),
]:
    Q, K, V = rng.normal(size=(B, H, Lq_, dk)), rng.normal(size=(B, H, Lk, dk)), rng.normal(size=(B, H, Lk, dk))
    G = rng.normal(size=(B, H, Lq_, dk))
    out = attn.forward(Q, K, V, mask)
    dQ, dK, dV = attn.backward(G)
    Qt, Kt, Vt = t(Q, True), t(K, True), t(V, True)
    ref = F.scaled_dot_product_attention(Qt, Kt, Vt, attn_mask=torch.from_numpy(~np.broadcast_to(mask, (B, H, Lq_, Lk))))
    (ref * t(G)).sum().backward()
    report(f"attention backward, {label}: dQ, dK, dV",
           max(rel_err(dQ, Qt.grad), rel_err(dK, Kt.grad), rel_err(dV, Vt.grad)),
           extra=f"  (forward {rel_err(out, ref.detach()):.1e})")

# ---------------------------------------------------------------- 3. multi-head attention (cross-attention)
D, h = 12, 3
np.random.seed(1)
mha = imp.MultiHeadAttention(D, h, dropout_p=0.0)
xq, xkv = rng.normal(size=(2, 5, D)), rng.normal(size=(2, 7, D))
kpm = np.zeros((2, 7), bool)
kpm[1, 5:] = True
G = rng.normal(size=(2, 5, D))
mha.forward(xq, xkv, xkv * 0.5 + 0.1, mask=kpm[:, None, None, :])     # value != key on purpose
d_q, d_k, d_v = mha.backward(G)
tq, tk, tv = t(xq, True), t(xkv, True), t(xkv * 0.5 + 0.1, True)
W = {n: t(getattr(mha, n), True) for n in ("W_Q", "W_K", "W_V", "W_O")}


def torch_mha(q, k, v, W, mask, n_heads):
    Bq, Lq_, D_ = q.shape
    Lk_ = k.shape[1]
    split = lambda x, L: x.reshape(Bq, L, n_heads, D_ // n_heads).transpose(1, 2)  # noqa: E731
    o = F.scaled_dot_product_attention(split(q @ W["W_Q"], Lq_), split(k @ W["W_K"], Lk_), split(v @ W["W_V"], Lk_),
                                       attn_mask=torch.from_numpy(~mask))
    return o.transpose(1, 2).reshape(Bq, Lq_, D_) @ W["W_O"]


(torch_mha(tq, tk, tv, W, kpm[:, None, None, :], h) * t(G)).sum().backward()
errs = {"d_query": rel_err(d_q, tq.grad), "d_key": rel_err(d_k, tk.grad), "d_value": rel_err(d_v, tv.grad)}
errs.update({n: rel_err(mha.grads[n], W[n].grad) for n in W})
report("MultiHeadAttention backward (cross-attn, key padding): inputs + W_Q/W_K/W_V/W_O", max(errs.values()),
       extra="  (" + ", ".join(f"{k} {v:.0e}" for k, v in errs.items()) + ")")

# ---------------------------------------------------------------- 4. LayerNorm and FeedForward
ln = imp.LayerNorm(D)
ln.gamma, ln.beta = 1 + 0.3 * rng.normal(size=D), 0.3 * rng.normal(size=D)
x = rng.normal(size=(3, 4, D)) * 2 + 0.5
G = rng.normal(size=x.shape)
ln.forward(x)
dx = ln.backward(G)
xt, gt, bt = t(x, True), t(ln.gamma, True), t(ln.beta, True)
(F.layer_norm(xt, (D,), gt, bt, 1e-5) * t(G)).sum().backward()
report("LayerNorm backward: dx, dgamma, dbeta",
       max(rel_err(dx, xt.grad), rel_err(ln.grads["gamma"], gt.grad), rel_err(ln.grads["beta"], bt.grad)))

np.random.seed(2)
ff = imp.FeedForward(D, 20, dropout_p=0.0)
ff.b1, ff.b2 = 0.3 * rng.normal(size=20), 0.3 * rng.normal(size=D)
ff.forward(x)
dx = ff.backward(G)
P = {n: t(getattr(ff, n), True) for n in ("W1", "b1", "W2", "b2")}
xt = t(x, True)
((torch.relu(xt @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]) * t(G)).sum().backward()
report("FeedForward backward: dx, dW1, db1, dW2, db2",
       max([rel_err(dx, xt.grad)] + [rel_err(ff.grads[n], P[n].grad) for n in P]))

# ---------------------------------------------------------------- 5. cross-entropy with padding + label smoothing
logits = rng.normal(size=(3, 7, 12)) * 2
targets = rng.integers(1, 12, size=(3, 7))
targets[1, 5:] = 0
targets[2, 3:] = 0                                   # pad = 0 = ignore_index
for eps in (0.0, 0.1):
    loss, dlogits = tm.cross_entropy_loss_and_grad(logits, targets, ignore_index=0, label_smoothing=eps)
    lt = t(logits, True)
    ref = F.cross_entropy(lt.reshape(-1, 12), torch.from_numpy(targets.reshape(-1)), ignore_index=0, label_smoothing=eps)
    ref.backward()
    report(f"cross_entropy_loss_and_grad (ignore pad, eps_ls={eps}) vs F.cross_entropy: loss and dlogits",
           max(abs(loss - ref.item()) / abs(ref.item()), rel_err(dlogits, lt.grad)))


# ---------------------------------------------------------------- 6. full model vs nn.Transformer*Layer + autograd
V, D, h, dff, N, S, T, Bsz = 12, 16, 2, 32, 2, 7, 7, 3
src = rng.integers(3, V, size=(Bsz, S))
src[1, 5:] = 0
src[2, 4:] = 0                                       # padded source rows
tgt_full = np.concatenate([np.ones((Bsz, 1), int), rng.integers(3, V, size=(Bsz, T))], axis=1)
tgt_full[1, 6:] = 0
tgt_full[2, 4:] = 0                                  # padded target rows
tgt_in, tgt_out = tgt_full[:, :-1], tgt_full[:, 1:]  # teacher forcing; pads in tgt_out are ignored
src_mask, tgt_mask = imp.create_padding_mask(src), imp.create_causal_mask(T)


def make_model(seed, tied, dropout_p=0.0):
    np.random.seed(seed)
    m = imp.Transformer(V, V, d_model=D, n_heads=h, n_encoder_layers=N, n_decoder_layers=N, d_ff=dff,
                        dropout_p=dropout_p, max_len=32, share_embeddings=tied)
    for blk in m.encoder_layers + m.decoder_layers:  # non-trivial biases / norms so their gradients are tested
        blk.feed_forward.b1 = 0.1 * rng.normal(size=dff)
        blk.feed_forward.b2 = 0.1 * rng.normal(size=D)
        for nm in ("norm1", "norm2", "norm3"):
            if hasattr(blk, nm):
                getattr(blk, nm).gamma = 1 + 0.1 * rng.normal(size=D)
                getattr(blk, nm).beta = 0.1 * rng.normal(size=D)
    if not tied:                                     # the 0.02-scale init gives tiny logits; use larger ones
        m.output_projection = rng.normal(size=(D, V)) * 0.3
    return m


def numpy_loss_and_grads(m, eps_ls, train=False, seed=None):
    if train:
        m.train()
    else:
        m.eval()
    if seed is not None:
        np.random.seed(seed)                         # same seed -> same dropout masks
    logits = m.forward(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
    loss, dlogits = tm.cross_entropy_loss_and_grad(logits, tgt_out, ignore_index=0, label_smoothing=eps_ls)
    m.backward(dlogits)
    return loss, {n: g.copy() for n, _, g in m.named_params_and_grads()}


PE = np.array([[math.sin(p / 10000 ** (2 * (j // 2) / D)) if j % 2 == 0 else math.cos(p / 10000 ** (2 * (j // 2) / D))
                for j in range(D)] for p in range(32)])  # Sec. 3.5, computed independently


def copy_mha(dst, src_mha):
    with torch.no_grad():  # repo computes x @ W (W is (in, out)); torch computes x @ W^T, so copy W.T
        dst.in_proj_weight.copy_(torch.cat([t(src_mha.W_Q).T, t(src_mha.W_K).T, t(src_mha.W_V).T], 0))
        dst.in_proj_bias.zero_()
        dst.out_proj.weight.copy_(t(src_mha.W_O).T)
        dst.out_proj.bias.zero_()


def mha_grads(tmha, prefix):
    gw = tmha.in_proj_weight.grad.numpy()            # (3D, D): rows = [W_Q^T; W_K^T; W_V^T]
    Dm = gw.shape[1]
    return {prefix + "W_Q": gw[:Dm].T, prefix + "W_K": gw[Dm:2 * Dm].T, prefix + "W_V": gw[2 * Dm:].T,
            prefix + "W_O": tmha.out_proj.weight.grad.numpy().T}


def copy_ff_ln(L, blk, norms):
    with torch.no_grad():
        L.linear1.weight.copy_(t(blk.feed_forward.W1).T)
        L.linear1.bias.copy_(t(blk.feed_forward.b1))
        L.linear2.weight.copy_(t(blk.feed_forward.W2).T)
        L.linear2.bias.copy_(t(blk.feed_forward.b2))
        for nm in norms:
            getattr(L, nm).weight.copy_(t(getattr(blk, nm).gamma))
            getattr(L, nm).bias.copy_(t(getattr(blk, nm).beta))


def ff_ln_grads(L, prefix, norms):
    out = {prefix + "feed_forward.W1": L.linear1.weight.grad.numpy().T, prefix + "feed_forward.b1": L.linear1.bias.grad.numpy(),
           prefix + "feed_forward.W2": L.linear2.weight.grad.numpy().T, prefix + "feed_forward.b2": L.linear2.bias.grad.numpy()}
    for nm in norms:
        out[prefix + nm + ".gamma"] = getattr(L, nm).weight.grad.numpy()
        out[prefix + nm + ".beta"] = getattr(L, nm).bias.grad.numpy()
    return out


def torch_reference(m, eps_ls):
    """Same weights in torch's own layers + autograd; returns loss and {numpy name: grad in numpy layout}."""
    enc, dec = [], []
    for blk in m.encoder_layers:
        L = nn.TransformerEncoderLayer(D, h, dff, dropout=0.0, batch_first=True, dtype=torch.float64)
        copy_mha(L.self_attn, blk.self_attention)
        copy_ff_ln(L, blk, ("norm1", "norm2"))
        enc.append(L)
    for blk in m.decoder_layers:
        L = nn.TransformerDecoderLayer(D, h, dff, dropout=0.0, batch_first=True, dtype=torch.float64)
        copy_mha(L.self_attn, blk.self_attention)
        copy_mha(L.multihead_attn, blk.cross_attention)
        copy_ff_ln(L, blk, ("norm1", "norm2", "norm3"))
        dec.append(L)
    if m.share_embeddings:                           # one leaf tensor used three times
        E = t(m.src_embedding.embedding, True)
        E_src, E_tgt, W_out = E, E, E.T
    else:
        E_src, E_tgt = t(m.src_embedding.embedding, True), t(m.tgt_embedding.embedding, True)
        W_out = t(m.output_projection, True)

    kpm = torch.from_numpy(src == 0)
    causal = torch.from_numpy(np.triu(np.ones((T, T), bool), 1))
    x = E_src[torch.from_numpy(src)] * math.sqrt(D) + t(PE[:S])
    for L in enc:
        x = L(x, src_key_padding_mask=kpm)
    y = E_tgt[torch.from_numpy(tgt_in)] * math.sqrt(D) + t(PE[:T])
    for L in dec:
        y = L(y, x, tgt_mask=causal, memory_key_padding_mask=kpm)
    logits = y @ W_out
    loss = F.cross_entropy(logits.reshape(-1, V), torch.from_numpy(tgt_out.reshape(-1)),
                           ignore_index=0, label_smoothing=eps_ls)
    loss.backward()

    grads = {}
    if m.share_embeddings:
        grads["shared_embedding"] = E.grad.numpy()
    else:
        grads.update({"src_embedding.embedding": E_src.grad.numpy(), "tgt_embedding.embedding": E_tgt.grad.numpy(),
                      "output_projection": W_out.grad.numpy()})
    for i, L in enumerate(enc):
        grads.update(mha_grads(L.self_attn, f"encoder_layers.{i}.self_attention."))
        grads.update(ff_ln_grads(L, f"encoder_layers.{i}.", ("norm1", "norm2")))
    for i, L in enumerate(dec):
        grads.update(mha_grads(L.self_attn, f"decoder_layers.{i}.self_attention."))
        grads.update(mha_grads(L.multihead_attn, f"decoder_layers.{i}.cross_attention."))
        grads.update(ff_ln_grads(L, f"decoder_layers.{i}.", ("norm1", "norm2", "norm3")))
    return loss.item(), grads


for tied, eps_ls in [(False, 0.1), (True, 0.0)]:
    model = make_model(10 + tied, tied)
    loss_np, g_np = numpy_loss_and_grads(model, eps_ls)
    loss_t, g_t = torch_reference(model, eps_ls)
    assert set(g_np) == set(g_t), sorted(set(g_np) ^ set(g_t))
    per = {n: rel_err(g_np[n], g_t[n]) for n in g_np}
    worst = max(per, key=per.get)
    zero = [n for n in g_t if np.abs(g_t[n]).max() == 0]
    label = "tied embeddings (Sec. 3.4)" if tied else "untied"
    print(f"       full model, {label}, eps_ls={eps_ls}: loss numpy {loss_np:.12f} vs torch {loss_t:.12f}; "
          f"{len(per)} parameter tensors, {sum(v.size for v in g_np.values())} values")
    report(f"full Transformer, {label}: every parameter gradient vs autograd", per[worst],
           extra=f"  (worst: {worst}; all-zero reference grads: {zero or 'none'})")
    report(f"full Transformer, {label}: loss vs torch", abs(loss_np - loss_t) / loss_t, tol=1e-12)
    if not tied:
        pad_rows = max(np.abs(g_np["src_embedding.embedding"][0]).max(), np.abs(g_np["tgt_embedding.embedding"][0]).max())
        print(f"       gradient of the padding token's embedding rows: max |.| = {pad_rows:.1e} "
              "(padded positions are masked/ignored everywhere, so 0 is expected)")


# ---------------------------------------------------------------- 7. central finite differences
def fd_check(m, grads, train, seed, n_random, label, h_fd=1e-5):
    """Largest-|g| entry of every tensor + n_random random entries with |g| > 1e-4."""
    params = {n: p for n, p, _ in m.named_params_and_grads()}
    picks = [(n, np.unravel_index(np.abs(gr).argmax(), gr.shape)) for n, gr in grads.items()]
    big = [(n, idx) for n, gr in grads.items() for idx in zip(*np.nonzero(np.abs(gr) > 1e-4))]
    for k in rng.choice(len(big), size=min(n_random, len(big)), replace=False):
        picks.append(big[k])
    worst = (0.0, None)
    for n, idx in picks:
        p = params[n]
        old = p[idx]
        p[idx] = old + h_fd                          # perturb the model's own array in place
        lp = numpy_loss_and_grads(m, 0.0, train=train, seed=seed)[0]
        p[idx] = old - h_fd
        lm = numpy_loss_and_grads(m, 0.0, train=train, seed=seed)[0]
        p[idx] = old
        num, ana = (lp - lm) / (2 * h_fd), grads[n][idx]
        err = abs(num - ana) / max(abs(num), abs(ana))
        if err > worst[0]:
            worst = (err, f"{n}{list(map(int, idx))}: numeric {num:.6e} vs analytic {ana:.6e}")
    report(f"central differences (h={h_fd:g}), {label}: {len(picks)} entries", worst[0],
           extra=f"  (worst: {worst[1]})")


model = make_model(10, False)
_, grads = numpy_loss_and_grads(model, 0.0)
fd_check(model, grads, train=False, seed=None, n_random=40, label="dropout off")

p_drop = 0.2
model_do = make_model(12, False, dropout_p=p_drop)
_, grads_do = numpy_loss_and_grads(model_do, 0.0, train=True, seed=123)
fd_check(model_do, grads_do, train=True, seed=123, n_random=30,
         label=f"dropout ON (p={p_drop}, masks fixed by reseeding)")

# ---------------------------------------------------------------- 8. Adam vs torch.optim.Adam
d_model, warmup = 16, 3
shapes = [(4, 5), (5,), (3, 2, 2)]
ps = [rng.normal(size=s_) for s_ in shapes]
ps_t = [t(p_, True) for p_ in ps]
opt = tm.Adam([p_ for p_ in ps], d_model=d_model, warmup=warmup)
opt_t = torch.optim.Adam(ps_t, lr=1.0, betas=(0.9, 0.98), eps=1e-9)
worst = 0.0
lrs = []
for step in range(1, 7):
    gs = [rng.normal(size=s_) for s_ in shapes]
    lrs.append(opt.step(gs))
    for group in opt_t.param_groups:
        group["lr"] = tm.get_lr(step, d_model, warmup)
    for p_t, g_ in zip(ps_t, gs):
        p_t.grad = t(g_)
    opt_t.step()
    worst = max(worst, max(rel_err(p_, p_t.detach()) for p_, p_t in zip(ps, ps_t)))
report("Adam(0.9, 0.98, 1e-9) + Eq. 3 schedule vs torch.optim.Adam, 6 steps", worst,
       extra=f"  (lr per step: {', '.join(f'{v:.4f}' for v in lrs)}; peak at step {warmup} = "
             f"(d_model*warmup)^-0.5 = {(d_model * warmup) ** -0.5:.4f})")

n_ok = sum(ok for _, ok, _ in RESULTS)
print(f"\nSUMMARY: {n_ok}/{len(RESULTS)} checks pass; worst relative error = {max(e for _, _, e in RESULTS):.2e} "
      f"(tolerance {TOL:g})")
print("RESULT:", "PASS" if n_ok == len(RESULTS) else "FAIL")
