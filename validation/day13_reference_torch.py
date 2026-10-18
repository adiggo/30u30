"""Day 13 (Attention Is All You Need) -- reference checks for the NumPy Transformer.

Verifies papers/13_attention/implementation.py (and the exercise solutions that
re-implement the same blocks) against trusted references, all in float64:

  1. ScaledDotProductAttention  vs torch F.scaled_dot_product_attention (padding + causal masks)
  2. MultiHeadAttention         vs nn.MultiheadAttention(bias=False) with copied weights,
                                   incl. per-head attention weights (head split/merge order)
  3. PositionalEncoding         vs the paper's formula (Sec. 3.5) evaluated with explicit loops,
                                   plus the "PE(pos+k) is a linear function of PE(pos)" property
  4. layer_norm                 vs F.layer_norm (random gamma/beta)
  5. EncoderBlock / DecoderBlock vs nn.TransformerEncoderLayer / nn.TransformerDecoderLayer
                                   (norm_first=False = post-norm, as in the paper), copied weights
  6. Transformer.forward        vs an independent torch composition (embed*sqrt(d) + PE -> layers -> proj)
  7. Mask semantics (perturbation): future target tokens cannot change earlier logits; padded
                                   source tokens cannot change any logit
  8. sqrt(d_k) variance argument (footnote 4): Var(q.k) = d_k for unit-variance components

Run:  OMP_NUM_THREADS=1 python validation/day13_reference_torch.py
"""
import importlib.util
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

RESULTS = []


def report(name, err, tol=1e-9):
    ok = err <= tol
    RESULTS.append((name, ok, err))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max|diff| = {err:.3e} (tol {tol:g})")


def t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).double()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. SDPA
B, H, Lq, Lk, dk = 2, 3, 5, 7, 8
Q, K, V = rng.normal(size=(B, H, Lq, dk)), rng.normal(size=(B, H, Lk, dk)), rng.normal(size=(B, H, Lk, dk))
attn = imp.ScaledDotProductAttention(dropout_p=0.0)
attn.eval()
out = attn.forward(Q, K, V)
ref = F.scaled_dot_product_attention(t(Q), t(K), t(V))
report("SDPA (no mask) vs F.scaled_dot_product_attention", (t(out) - ref).abs().max().item())

pad = rng.random((B, 1, 1, Lk)) < 0.4          # True = masked (repo convention)
pad[..., 0] = False                             # keep >=1 visible key per row
out = attn.forward(Q, K, V, mask=pad)
ref = F.scaled_dot_product_attention(t(Q), t(K), t(V), attn_mask=torch.from_numpy(~pad))  # torch: True = attend
report("SDPA (padding mask) vs torch", (t(out) - ref).abs().max().item())

Qs, Ks, Vs = rng.normal(size=(B, H, 6, dk)), rng.normal(size=(B, H, 6, dk)), rng.normal(size=(B, H, 6, dk))
out = attn.forward(Qs, Ks, Vs, mask=imp.create_causal_mask(6))
ref = F.scaled_dot_product_attention(t(Qs), t(Ks), t(Vs), is_causal=True)
report("SDPA (causal mask) vs torch is_causal=True", (t(out) - ref).abs().max().item())
w = attn.attention_weights
report("causal weights are exactly 0 above the diagonal (|w| max)", float(np.abs(np.triu(w[0, 0], 1)).max()), tol=1e-300)
report("attention rows sum to 1", float(np.abs(w.sum(-1) - 1).max()), tol=1e-12)


# ---------------------------------------------------------------- 2. MHA
def torch_mha_from(W_Q, W_K, W_V, W_O, n_heads):
    d = W_Q.shape[0]
    m = nn.MultiheadAttention(d, n_heads, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        # repo computes x @ W (W is (in, out)); torch Linear computes x @ W^T, so copy W.T
        m.in_proj_weight.copy_(torch.cat([t(W_Q).T, t(W_K).T, t(W_V).T], 0))
        m.out_proj.weight.copy_(t(W_O).T)
    return m.eval()


d_model, n_heads = 24, 4
np.random.seed(1)
mha = imp.MultiHeadAttention(d_model, n_heads, dropout_p=0.0)
mha.eval()
tm = torch_mha_from(mha.W_Q, mha.W_K, mha.W_V, mha.W_O, n_heads)
xq, xkv = rng.normal(size=(2, 5, d_model)), rng.normal(size=(2, 9, d_model))
with torch.no_grad():
    ref, ref_w = tm(t(xq), t(xkv), t(xkv), need_weights=True, average_attn_weights=False)
out = mha.forward(xq, xkv, xkv)
report("MHA cross-attention output vs nn.MultiheadAttention", (t(out) - ref).abs().max().item())
report("MHA per-head weights vs nn.MultiheadAttention (head order)", (t(mha.get_attention_weights()) - ref_w).abs().max().item())

kpm = np.zeros((2, 9), bool)
kpm[0, 6:] = True
kpm[1, 3:] = True
causal = np.triu(np.ones((5, 5), bool), 1)
with torch.no_grad():
    ref, _ = tm(t(xq), t(xkv), t(xkv), key_padding_mask=torch.from_numpy(kpm))
    ref2, _ = tm(t(xq), t(xq), t(xq), attn_mask=torch.from_numpy(causal))
out = mha.forward(xq, xkv, xkv, mask=kpm[:, None, None, :])
out2 = mha.forward(xq, xq, xq, mask=imp.create_causal_mask(5))
report("MHA with key-padding mask vs torch", (t(out) - ref).abs().max().item())
report("MHA self-attn with causal mask vs torch", (t(out2) - ref2).abs().max().item())

# Exercise solution 2 MHA (same weights layout) -- learners compare against it
sol2 = load(f"{DAY}/exercises/solutions/solution_02_multi_head_attention.py", "sol2")
np.random.seed(2)
smha = sol2.MultiHeadAttention(d_model, n_heads)
tm2 = torch_mha_from(smha.W_Q, smha.W_K, smha.W_V, smha.W_O, n_heads)
with torch.no_grad():
    ref, ref_w = tm2(t(xq), t(xkv), t(xkv), average_attn_weights=False)
report("solution_02 MHA vs nn.MultiheadAttention", (t(smha.forward(xq, xkv, xkv)) - ref).abs().max().item())

# ---------------------------------------------------------------- 3. Positional encoding
d_pe, L_pe = 16, 60
pe = imp.PositionalEncoding(d_pe, max_len=L_pe, dropout_p=0.0).pe
ref = np.zeros((L_pe, d_pe))
for pos in range(L_pe):
    for i in range(d_pe // 2):                       # paper: PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...)
        ref[pos, 2 * i] = math.sin(pos / 10000 ** (2 * i / d_pe))
        ref[pos, 2 * i + 1] = math.cos(pos / 10000 ** (2 * i / d_pe))
report("PositionalEncoding vs paper formula (loops)", float(np.abs(pe - ref).max()), tol=1e-12)
sol3 = load(f"{DAY}/exercises/solutions/solution_03_positional_encoding.py", "sol3")
report("solution_03 PE vs paper formula", float(np.abs(sol3.PositionalEncoding(d_pe, L_pe).pe - ref).max()), tol=1e-12)

# linear-in-offset property: PE[pos+k] = M_k PE[pos] with M_k block-diagonal 2x2 rotations
k = 7
omega = 1.0 / 10000 ** (np.arange(0, d_pe, 2) / d_pe)
M = np.zeros((d_pe, d_pe))
for i, w_ in enumerate(omega):
    c, s = math.cos(w_ * k), math.sin(w_ * k)
    # [sin(w(p+k)), cos(w(p+k))] = [[c, s], [-s, c]] @ [sin(wp), cos(wp)]
    M[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[c, s], [-s, c]]
report("PE(pos+7) == M_7 @ PE(pos) for all pos (fixed matrix, no pos dependence)",
       float(np.abs(pe[k:] - pe[:-k] @ M.T).max()), tol=1e-12)

# ---------------------------------------------------------------- 4. Layer norm
x = rng.normal(size=(3, 4, 10)) * 3 + 1
g, b = rng.normal(size=10), rng.normal(size=10)
report("layer_norm vs F.layer_norm", (t(imp.layer_norm(x, g, b)) - F.layer_norm(t(x), (10,), t(g), t(b), 1e-5)).abs().max().item())


# ---------------------------------------------------------------- 5. Encoder / decoder blocks
def copy_mha(dst, src):
    with torch.no_grad():
        dst.in_proj_weight.copy_(torch.cat([t(src.W_Q).T, t(src.W_K).T, t(src.W_V).T], 0))
        dst.in_proj_bias.zero_()
        dst.out_proj.weight.copy_(t(src.W_O).T)
        dst.out_proj.bias.zero_()


def copy_ff(layer, ff):
    with torch.no_grad():
        layer.linear1.weight.copy_(t(ff.W1).T)
        layer.linear1.bias.copy_(t(ff.b1))
        layer.linear2.weight.copy_(t(ff.W2).T)
        layer.linear2.bias.copy_(t(ff.b2))


def copy_ln(dst, src):
    with torch.no_grad():
        dst.weight.copy_(t(src.gamma))
        dst.bias.copy_(t(src.beta))


def randomize_affine(block, d, d_ff):
    """Make biases / LN params non-trivial so the comparison actually tests them."""
    block.feed_forward.b1 = rng.normal(size=d_ff) * 0.1
    block.feed_forward.b2 = rng.normal(size=d) * 0.1
    for nm in ("norm1", "norm2", "norm3"):
        if hasattr(block, nm):
            getattr(block, nm).gamma = 1 + 0.1 * rng.normal(size=d)
            getattr(block, nm).beta = 0.1 * rng.normal(size=d)


d, h, dff = 16, 4, 32
np.random.seed(3)
enc = imp.EncoderBlock(d, h, dff, dropout_p=0.0)
enc.eval()
randomize_affine(enc, d, dff)
tenc = nn.TransformerEncoderLayer(d, h, dff, dropout=0.0, activation="relu", batch_first=True,
                                  norm_first=False, dtype=torch.float64).eval()
copy_mha(tenc.self_attn, enc.self_attention)
copy_ff(tenc, enc.feed_forward)
copy_ln(tenc.norm1, enc.norm1)
copy_ln(tenc.norm2, enc.norm2)
xs = rng.normal(size=(2, 6, d))
src_kpm = np.zeros((2, 6), bool)
src_kpm[1, 4:] = True
with torch.no_grad():
    ref = tenc(t(xs))
    ref_m = tenc(t(xs), src_key_padding_mask=torch.from_numpy(src_kpm))
report("EncoderBlock vs nn.TransformerEncoderLayer(norm_first=False)", (t(enc.forward(xs)) - ref).abs().max().item())
out_m = enc.forward(xs, src_kpm[:, None, None, :])
# torch's fast path zero-fills padded *query* rows; compare only real positions
report("EncoderBlock + padding mask vs torch (non-pad rows)",
       max((t(out_m[0]) - ref_m[0]).abs().max().item(), (t(out_m[1, :4]) - ref_m[1, :4]).abs().max().item()))

np.random.seed(4)
dec = imp.DecoderBlock(d, h, dff, dropout_p=0.0)
dec.eval()
randomize_affine(dec, d, dff)
tdec = nn.TransformerDecoderLayer(d, h, dff, dropout=0.0, activation="relu", batch_first=True,
                                  norm_first=False, dtype=torch.float64).eval()
copy_mha(tdec.self_attn, dec.self_attention)
copy_mha(tdec.multihead_attn, dec.cross_attention)
copy_ff(tdec, dec.feed_forward)
for nm in ("norm1", "norm2", "norm3"):
    copy_ln(getattr(tdec, nm), getattr(dec, nm))
xt, mem = rng.normal(size=(2, 5, d)), rng.normal(size=(2, 6, d))
with torch.no_grad():
    ref = tdec(t(xt), t(mem), tgt_mask=torch.from_numpy(np.triu(np.ones((5, 5), bool), 1)),
               memory_key_padding_mask=torch.from_numpy(src_kpm))
out = dec.forward(xt, mem, self_mask=imp.create_causal_mask(5), cross_mask=src_kpm[:, None, None, :])
report("DecoderBlock (causal + memory padding) vs nn.TransformerDecoderLayer", (t(out) - ref).abs().max().item())

# exercise solutions 4 (encoder) and 5 (decoder) against the same torch layers
sol4 = load(f"{DAY}/exercises/solutions/solution_04_encoder_block.py", "sol4")
np.random.seed(5)
senc = sol4.EncoderBlock(d, h, dff)
t4 = nn.TransformerEncoderLayer(d, h, dff, dropout=0.0, batch_first=True, dtype=torch.float64).eval()
with torch.no_grad():
    t4.self_attn.in_proj_weight.copy_(torch.cat([t(senc.W_Q).T, t(senc.W_K).T, t(senc.W_V).T], 0))
    t4.self_attn.in_proj_bias.zero_()
    t4.self_attn.out_proj.weight.copy_(t(senc.W_O).T)
    t4.self_attn.out_proj.bias.zero_()
copy_ff(t4, senc.ff)
copy_ln(t4.norm1, senc.norm1)
copy_ln(t4.norm2, senc.norm2)
with torch.no_grad():
    report("solution_04 EncoderBlock vs nn.TransformerEncoderLayer", (t(senc.forward(xs)) - t4(t(xs))).abs().max().item())

sol5 = load(f"{DAY}/exercises/solutions/solution_05_full_transformer.py", "sol5")
np.random.seed(6)
sdec = sol5.DecoderBlock(d, h, dff)
t5 = nn.TransformerDecoderLayer(d, h, dff, dropout=0.0, batch_first=True, dtype=torch.float64).eval()
copy_mha(t5.self_attn, sdec.self_attn)
copy_mha(t5.multihead_attn, sdec.cross_attn)
copy_ff(t5, sdec.ff)
for nm in ("norm1", "norm2", "norm3"):
    copy_ln(getattr(t5, nm), getattr(sdec, nm))
with torch.no_grad():
    ref = t5(t(xt), t(mem), tgt_mask=torch.from_numpy(np.triu(np.ones((5, 5), bool), 1)))
report("solution_05 DecoderBlock vs nn.TransformerDecoderLayer",
       (t(sdec.forward(xt, mem, self_mask=sol5.create_causal_mask(5))) - ref).abs().max().item())

# ---------------------------------------------------------------- 6. Full model composition
V_src, V_tgt, d, h, dff, N = 13, 11, 16, 4, 32, 2
np.random.seed(7)
model = imp.Transformer(V_src, V_tgt, d_model=d, n_heads=h, n_encoder_layers=N, n_decoder_layers=N,
                        d_ff=dff, dropout_p=0.0, max_len=50)
model.eval()
src = np.array([[3, 4, 5, 6, 0, 0], [7, 8, 9, 10, 11, 12]])
tgt = np.array([[1, 3, 4, 5, 6], [1, 7, 8, 9, 10]])
src_mask = imp.create_padding_mask(src)                # (B,1,1,S), True = pad
tgt_mask = imp.create_causal_mask(tgt.shape[1])
logits = model.forward(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

enc_layers, dec_layers = [], []
for blk in model.encoder_layers:
    L = nn.TransformerEncoderLayer(d, h, dff, dropout=0.0, batch_first=True, dtype=torch.float64).eval()
    copy_mha(L.self_attn, blk.self_attention); copy_ff(L, blk.feed_forward)
    copy_ln(L.norm1, blk.norm1); copy_ln(L.norm2, blk.norm2)
    enc_layers.append(L)
for blk in model.decoder_layers:
    L = nn.TransformerDecoderLayer(d, h, dff, dropout=0.0, batch_first=True, dtype=torch.float64).eval()
    copy_mha(L.self_attn, blk.self_attention); copy_mha(L.multihead_attn, blk.cross_attention)
    copy_ff(L, blk.feed_forward)
    for nm in ("norm1", "norm2", "norm3"):
        copy_ln(getattr(L, nm), getattr(blk, nm))
    dec_layers.append(L)
PE = t(ref_pe := np.array([[math.sin(p / 10000 ** (2 * (j // 2) / d)) if j % 2 == 0 else
                            math.cos(p / 10000 ** (2 * (j // 2) / d)) for j in range(d)] for p in range(50)]))
with torch.no_grad():
    kpm_t = torch.from_numpy(src == 0)
    x = t(model.src_embedding.embedding)[torch.from_numpy(src)] * math.sqrt(d) + PE[:src.shape[1]]
    for L in enc_layers:
        # slow path (grad disabled + mask) keeps padded rows; real rows are what matter
        x = L(x, src_key_padding_mask=kpm_t)
    y = t(model.tgt_embedding.embedding)[torch.from_numpy(tgt)] * math.sqrt(d) + PE[:tgt.shape[1]]
    causal_t = torch.from_numpy(np.triu(np.ones((5, 5), bool), 1))
    for L in dec_layers:
        y = L(y, x, tgt_mask=causal_t, memory_key_padding_mask=kpm_t)
    ref_logits = y @ t(model.output_projection)
report("Transformer.forward vs independent torch composition", (t(logits) - ref_logits).abs().max().item(), tol=1e-8)

# ---------------------------------------------------------------- 7. Mask semantics by perturbation
tgt2 = tgt.copy()
tgt2[:, 3] = 2                                          # change target token at position 3
logits2 = model.forward(src, tgt2, src_mask=src_mask, tgt_mask=tgt_mask)
report("causal: changing tgt[3] leaves logits[:, :3] unchanged", float(np.abs(logits2[:, :3] - logits[:, :3]).max()), tol=1e-12)
print(f"       (and it does change logits[:, 3:]: max|diff| = {np.abs(logits2[:, 3:] - logits[:, 3:]).max():.3e})")
src3 = src.copy()
src3[0, 4:] = [9, 12]                                   # overwrite PAD slots, keep the original mask
logits3 = model.forward(src3, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
report("padding: changing tokens under the source pad mask leaves all logits unchanged",
       float(np.abs(logits3 - logits).max()), tol=1e-12)

# ---------------------------------------------------------------- 8. sqrt(d_k) variance argument
for dk_ in (16, 64, 256):
    q = rng.normal(size=(200_000, dk_))
    k_ = rng.normal(size=(200_000, dk_))
    dots = (q * k_).sum(1)
    print(f"[INFO] d_k={dk_:4d}: Var(q.k) = {dots.var():8.2f} (theory {dk_}),  Var(q.k/sqrt(d_k)) = {(dots / math.sqrt(dk_)).var():.3f} (theory 1)")

n_ok = sum(ok for _, ok, _ in RESULTS)
print(f"\nSUMMARY: {n_ok}/{len(RESULTS)} checks pass; worst max|diff| = {max(e for _, _, e in RESULTS):.3e}")
