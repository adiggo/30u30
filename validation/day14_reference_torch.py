"""Day 14 (The Annotated Transformer) -- reference checks.

Checks papers/14_annotated_transformer/implementation.py and train_minimal.py (float64):
  1. attention()               vs F.scaled_dot_product_attention (mask semantics: 1/True = keep)
  2. MultiHeadedAttention      vs nn.MultiheadAttention with copied weights + biases
  3. LayerNorm                 vs its own formula (unbiased std, eps outside the sqrt) and vs
                                  nn.LayerNorm (quantifies the deviation from Ba et al.)
  4. EncoderLayer/DecoderLayer vs an independent pre-norm composition x + Sublayer(LN(x))
  5. PositionalEncoding        vs the paper's formula
  6. LabelSmoothing            vs closed-form KL(q || p) incl. padding rows, and the 1-pad edge case
  7. rate()                    vs the paper's Eq. (3)
  8. PR #1 fixes in train_minimal.py:
       - NLLLoss(generator(h)) == cross-entropy on the raw projection (no double log-softmax)
       - (also: CrossEntropyLoss(log_probs) would give the same value: log_softmax is idempotent)
       - greedy_decode output is a fixed point of teacher-forced argmax (decode loop + masks agree)

Run: OMP_NUM_THREADS=1 python validation/day14_reference_torch.py
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "14_annotated_transformer")
sys.path.insert(0, DAY)
import implementation as imp  # noqa: E402
import train_minimal as tmin  # noqa: E402

RESULTS = []


def report(name, err, tol=1e-10):
    ok = err <= tol
    RESULTS.append((name, ok, err))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max|diff| = {err:.3e} (tol {tol:g})")


torch.manual_seed(0)
# ------------------------------------------------------------------ 1. attention()
q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
keep = torch.rand(2, 1, 1, 7) > 0.3
keep[..., 0] = True
out, p = imp.attention(q, k, v, mask=keep)
ref = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
report("attention() with padding mask vs F.scaled_dot_product_attention", (out - ref).abs().max().item())
qs = torch.randn(2, 3, 6, 8)
out, p = imp.attention(qs, qs, qs, mask=imp.subsequent_mask(6).unsqueeze(1))
ref = F.scaled_dot_product_attention(qs, qs, qs, is_causal=True)
report("attention() with subsequent_mask vs is_causal=True", (out - ref).abs().max().item())

# ------------------------------------------------------------------ 2. MultiHeadedAttention
d, h = 32, 4
mha = imp.MultiHeadedAttention(h, d, dropout=0.0).double().eval()
tm = nn.MultiheadAttention(d, h, bias=True, batch_first=True).eval()
with torch.no_grad():
    tm.in_proj_weight.copy_(torch.cat([mha.linears[i].weight for i in range(3)], 0))
    tm.in_proj_bias.copy_(torch.cat([mha.linears[i].bias for i in range(3)], 0))
    tm.out_proj.weight.copy_(mha.linears[3].weight)
    tm.out_proj.bias.copy_(mha.linears[3].bias)
xq, xkv = torch.randn(2, 5, d), torch.randn(2, 9, d)
src_keep = torch.ones(2, 1, 9, dtype=torch.bool)
src_keep[1, 0, 6:] = False
with torch.no_grad():
    ref, ref_w = tm(xq, xkv, xkv, key_padding_mask=~src_keep[:, 0, :], average_attn_weights=False)
    out = mha(xq, xkv, xkv, mask=src_keep)
report("MultiHeadedAttention (+src mask) vs nn.MultiheadAttention", (out - ref).abs().max().item())
report("MultiHeadedAttention per-head weights vs torch", (mha.attn - ref_w).abs().max().item())

# ------------------------------------------------------------------ 3. LayerNorm
ln = imp.LayerNorm(d).double()
with torch.no_grad():
    ln.a_2.copy_(1 + 0.1 * torch.randn(d))
    ln.b_2.copy_(0.1 * torch.randn(d))
x = torch.randn(3, 4, d) * 2 + 0.5
mean = x.mean(-1, keepdim=True)
std_unbiased = ((x - mean) ** 2).sum(-1, keepdim=True).div(d - 1).sqrt()
ref = ln.a_2 * (x - mean) / (std_unbiased + 1e-6) + ln.b_2
with torch.no_grad():
    report("LayerNorm vs formula a*(x-mean)/(std_unbiased+eps)+b", (ln(x) - ref).abs().max().item())
    dev = (ln(x) - F.layer_norm(x, (d,), ln.a_2, ln.b_2, eps=1e-6)).abs().max().item()
print(f"[INFO] Annotated LayerNorm vs nn.LayerNorm (biased var, eps inside sqrt), d={d}: max|diff| = {dev:.3e} "
      f"(relative scale factor sqrt((d-1)/d) = {math.sqrt((d - 1) / d):.5f})")


# ------------------------------------------------------------------ 4. Pre-norm layers vs independent composition
def ln_ref(mod, x):
    m = x.mean(-1, keepdim=True)
    s = ((x - m) ** 2).sum(-1, keepdim=True).div(x.size(-1) - 1).sqrt()
    return mod.a_2 * (x - m) / (s + mod.eps) + mod.b_2


def mha_ref(mod, qx, kx, vx, keep_mask):
    """Independent MHA from the paper's equations using F.scaled_dot_product_attention."""
    B, Lq, D = qx.shape
    H, dk = mod.h, mod.d_k

    def proj(lin, z):
        return F.linear(z, lin.weight, lin.bias).view(B, -1, H, dk).transpose(1, 2)
    o = F.scaled_dot_product_attention(proj(mod.linears[0], qx), proj(mod.linears[1], kx),
                                       proj(mod.linears[2], vx),
                                       attn_mask=None if keep_mask is None else keep_mask.unsqueeze(1).bool())
    return F.linear(o.transpose(1, 2).reshape(B, Lq, D), mod.linears[3].weight, mod.linears[3].bias)


def ffn_ref(mod, z):
    return F.linear(F.relu(F.linear(z, mod.w_1.weight, mod.w_1.bias)), mod.w_2.weight, mod.w_2.bias)


model = imp.make_model(13, 11, N=2, d_model=d, d_ff=64, h=h, dropout=0.0).double().eval()
for mod in model.modules():               # make LN affine params non-trivial
    if isinstance(mod, imp.LayerNorm):
        with torch.no_grad():
            mod.a_2.add_(0.1 * torch.randn(d))
            mod.b_2.add_(0.1 * torch.randn(d))
src = torch.tensor([[1, 4, 5, 6, 0, 0], [1, 7, 8, 9, 10, 12]])
tgt = torch.tensor([[1, 4, 5, 6, 0], [1, 7, 8, 9, 10]])
b = imp.Batch(src, torch.cat([tgt, torch.tensor([[0], [3]])], 1), pad=0)
with torch.no_grad():
    hidden = model(b.src, b.tgt, b.src_mask, b.tgt_mask)
    x = model.src_embed[0].lut(src) * math.sqrt(d) + model.src_embed[1].pe[:, :src.size(1)]
    for L in model.encoder.layers:
        x = x + mha_ref(L.self_attn, *(3 * [ln_ref(L.sublayer[0].norm, x)]), b.src_mask)
        x = x + ffn_ref(L.feed_forward, ln_ref(L.sublayer[1].norm, x))
    mem = ln_ref(model.encoder.norm, x)
    y = model.tgt_embed[0].lut(b.tgt) * math.sqrt(d) + model.tgt_embed[1].pe[:, :b.tgt.size(1)]
    for L in model.decoder.layers:
        y = y + mha_ref(L.self_attn, *(3 * [ln_ref(L.sublayer[0].norm, y)]), b.tgt_mask)
        z = ln_ref(L.sublayer[1].norm, y)
        y = y + mha_ref(L.src_attn, z, mem, mem, b.src_mask)
        y = y + ffn_ref(L.feed_forward, ln_ref(L.sublayer[2].norm, y))
    ref = ln_ref(model.decoder.norm, y)
report("EncoderDecoder.forward vs independent pre-norm composition", (hidden - ref).abs().max().item())
print(f"[INFO] EncoderDecoder.forward returns hidden states of shape {tuple(hidden.shape)} "
      f"(d_model={d}), not vocab probabilities -- the generator must be applied separately")

# ------------------------------------------------------------------ 5. Positional encoding
pe = imp.PositionalEncoding(16, 0.0, max_len=40).pe[0]
ref = torch.tensor([[math.sin(p_ / 10000 ** (2 * (j // 2) / 16)) if j % 2 == 0 else
                     math.cos(p_ / 10000 ** (2 * (j // 2) / 16)) for j in range(16)] for p_ in range(40)])
report("PositionalEncoding (float32 buffer) vs paper formula", (pe.double() - ref).abs().max().item(), tol=1e-5)

# ------------------------------------------------------------------ 6. Label smoothing
V, pad, eps_ls = 7, 0, 0.1
crit = imp.LabelSmoothing(V, pad, eps_ls)
logp = F.log_softmax(torch.randn(5, V), -1)
target = torch.tensor([3, 0, 5, 1, 0])          # two padding rows
q_ref = torch.full((5, V), eps_ls / (V - 2))
q_ref[torch.arange(5), target] = 1 - eps_ls
q_ref[:, pad] = 0
q_ref[target == pad] = 0
kl_ref = (q_ref[q_ref > 0] * (q_ref[q_ref > 0].log() - logp[q_ref > 0])).sum()
loss_ls = crit(logp, target)
report("LabelSmoothing true_dist vs closed form", (crit.true_dist - q_ref).abs().max().item())
report("LabelSmoothing loss vs closed-form KL(q||p)", abs(loss_ls.item() - kl_ref.item()))
print(f"[INFO] each real row of q sums to {q_ref[0].sum():.4f} (= 1); true-class mass {1 - eps_ls}, "
      f"others {eps_ls / (V - 2):.4f} (eps/(V-2): pad and true class excluded)")
try:
    one_pad = crit(logp[:3], torch.tensor([3, 0, 5]))
    print(f"[INFO] exactly-one-pad-row edge case runs (loss {one_pad.item():.4f})")
except Exception as e:  # noqa: BLE001
    print(f"[WARN] exactly-one-pad-row edge case fails: {type(e).__name__}: {e}")

# ------------------------------------------------------------------ 7. Noam schedule
dm, wu = 512, 4000
errs = [abs(imp.rate(s, dm, 1.0, wu) - dm ** -0.5 * min(s ** -0.5, s * wu ** -1.5)) for s in (1, 100, 3999, 4000, 4001, 10 ** 5)]
report("rate() vs paper Eq. (3)", max(errs), tol=1e-15)
print(f"[INFO] peak lr at step=warmup: {imp.rate(wu, dm, 1.0, wu):.3e} = 512^-0.5 * 4000^-0.5")

# ------------------------------------------------------------------ 8. PR #1 fixes in train_minimal.py
torch.manual_seed(1)
m = tmin.make_model(11, N=1, d_model=32, d_ff=64, h=4, dropout=0.0).double().eval()
data = torch.randint(1, 11, (4, 10))
data[:, 0] = 1
bt = tmin.Batch(data.clone(), data.clone())
hs = m(bt.src, bt.tgt, bt.src_mask, bt.tgt_mask)
logits = m.generator.proj(hs)
nll = nn.NLLLoss(ignore_index=0)(m.generator(hs).reshape(-1, 11), bt.tgt_y.reshape(-1))
ce = F.cross_entropy(logits.reshape(-1, 11), bt.tgt_y.reshape(-1), ignore_index=0)
ce_twice = nn.CrossEntropyLoss(ignore_index=0)(m.generator(hs).reshape(-1, 11), bt.tgt_y.reshape(-1))
report("train_minimal: NLLLoss(generator(h)) == cross_entropy(proj(h))", abs(nll.item() - ce.item()), tol=1e-12)
report("CrossEntropyLoss on log-probs gives the same value (log_softmax idempotent)", abs(ce_twice.item() - nll.item()), tol=1e-12)

src1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
mask1 = torch.ones(1, 1, 10)
with torch.no_grad():
    ys = tmin.greedy_decode(m, src1, mask1, max_len=10, start_symbol=1)
    tf = m.generator(m(src1, ys[:, :-1], mask1, tmin.subsequent_mask(ys.size(1) - 1))).argmax(-1)
mism = (tf != ys[:, 1:]).sum().item()
report("greedy_decode output == teacher-forced argmax on its own prefix (mismatched tokens)", float(mism), tol=0)
print(f"[INFO] greedy_decode returns shape {tuple(ys.shape)} (dtype {ys.dtype})")

n_ok = sum(ok for _, ok, _ in RESULTS)
print(f"\nSUMMARY: {n_ok}/{len(RESULTS)} checks pass")
