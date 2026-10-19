"""Day 15 (Bahdanau, Cho, Bengio 2014) -- reference checks.

  1. BahdanauAttention vs an independent loop implementation of the paper's Appendix A.2.2:
       e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j),  alpha_ij = softmax_j(e_ij),  c_i = sum_j alpha_ij h_j
     incl. padding mask (masked weights exactly 0, rows sum to 1, softmax over *source* axis).
  2. Encoder annotations: h_j = [fwd_j ; bwd_j] with packed variable-length batches == each
     sequence run alone (implementation.Encoder and exercises/solutions/solution_2.py).
  3. PR #1 fix: AttentionDecoder.forward (teacher forcing) runs exactly the same per-step
     computation as Seq2SeqWithAttention.translate(): feeding translate()'s own tokens back in
     reproduces its attention maps bit-for-bit and its argmax tokens.
  4. translate() start/end tokens vs the training data convention (PAD=0, SOS=1, EOS=2): uses the
     checkpoint best_model.pt that train_minimal.py writes (if present) and measures reversal
     accuracy with translate()'s defaults vs with sos_idx=1, eos_idx=2.

Run: OMP_NUM_THREADS=1 python validation/day15_reference_check.py [path/to/best_model.pt]
     (default checkpoint for check 4: papers/15_bahdanau_attention/best_model.pt, which
     train_minimal.py writes when run from that directory)
"""
import importlib.util
import os
import random
import sys

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "15_bahdanau_attention")
sys.path.insert(0, DAY)
sys.path.insert(0, os.path.join(DAY, "exercises", "solutions"))
import implementation as imp  # noqa: E402

RESULTS = []


def report(name, err, tol=1e-12):
    ok = err <= tol
    RESULTS.append((name, ok, err))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {err:.3e} (tol {tol:g})")


torch.manual_seed(0)
# ------------------------------------------------------------------ 1. additive attention
B, S, n, two_n, n_att = 3, 6, 5, 8, 7
att = imp.BahdanauAttention(hidden_size=n_att, key_size=two_n, query_size=n).double()
s_prev = torch.randn(B, n, dtype=torch.float64)          # s_{i-1}
h = torch.randn(B, S, two_n, dtype=torch.float64)        # annotations h_j
mask = torch.zeros(B, S, dtype=torch.bool)
mask[1, 4:] = True
mask[2, 2:] = True
W_a = att.query_layer.weight        # (n', n)
U_a = att.key_layer.weight          # (n', 2n)
v_a = att.energy_layer.weight[0]    # (n',)
e = torch.empty(B, S, dtype=torch.float64)
for b in range(B):
    for j in range(S):
        e[b, j] = v_a @ torch.tanh(W_a @ s_prev[b] + U_a @ h[b, j])
e_m = e.masked_fill(mask, float("-inf"))
alpha_ref = torch.exp(e_m - e_m.max(1, keepdim=True).values)
alpha_ref = alpha_ref / alpha_ref.sum(1, keepdim=True)
c_ref = torch.stack([(alpha_ref[b, :, None] * h[b]).sum(0) for b in range(B)])
with torch.no_grad():
    c, alpha = att(s_prev, h, mask)
report("alpha vs paper loop (A.2.2)", (alpha - alpha_ref).abs().max().item())
report("context c_i vs paper loop", (c - c_ref).abs().max().item())
report("masked positions get exactly zero weight", alpha[mask].abs().max().item(), tol=0.0)
report("weights sum to 1 over source positions", (alpha.sum(-1) - 1).abs().max().item())
with torch.no_grad():   # per-row independence: batch rows must not mix
    c0, _ = att(s_prev[2:3], h[2:3], mask[2:3])
report("batch rows independent (row 2 alone == row 2 in batch)", (c0[0] - c[2]).abs().max().item())


# ------------------------------------------------------------------ 2. packed bidirectional encoder
def per_example_check(enc, outputs_fn, name, H_out):
    torch.manual_seed(1)
    lengths = torch.tensor([3, 7, 5, 2])                   # deliberately NOT sorted
    src = torch.zeros(4, 7, dtype=torch.long)
    for b, L in enumerate(lengths):
        src[b, :L] = torch.randint(3, 20, (L,))
    enc.eval()
    with torch.no_grad():
        out_b, hid_b = outputs_fn(enc, src, lengths)
        err_out = err_hid = 0.0
        for b, L in enumerate(lengths):
            out_1, hid_1 = outputs_fn(enc, src[b:b + 1, :L], lengths[b:b + 1])
            err_out = max(err_out, (out_b[b, :L] - out_1[0, :L]).abs().max().item())
            err_hid = max(err_hid, (hid_b[:, b] - hid_1[:, 0]).abs().max().item())
    report(f"{name}: batched outputs == per-sequence outputs (valid positions)", err_out, tol=1e-6)
    report(f"{name}: batched final hidden == per-sequence final hidden", err_hid, tol=1e-6)
    return out_b, hid_b, src, lengths


enc = imp.Encoder(vocab_size=20, embed_size=6, hidden_size=4, num_layers=1, dropout=0.0)
out_b, hid_b, src, lengths = per_example_check(enc, lambda m, s, l: m(s, l), "implementation.Encoder", 8)
with torch.no_grad():
    # unpacked run of example 1 (length 7 = full width): out (1, 7, 2H), hid (2, 1, H)
    raw_out, raw_hid = enc.gru(enc.embedding(src[1:2]))
H = 4
# h_j = [fwd_j ; bwd_j]: the forward half at the LAST token and the backward half at the FIRST
# token are the two directions' final states (the paper's s_0 uses the backward one, bwd_1)
report("annotation layout: fwd half @ last token / bwd half @ first token == final states",
       max((raw_out[0, -1, :H] - raw_hid[0, 0]).abs().max().item(),
           (raw_out[0, 0, H:] - raw_hid[1, 0]).abs().max().item()), tol=1e-6)

spec = importlib.util.spec_from_file_location("sol2", os.path.join(DAY, "exercises", "solutions", "solution_2.py"))
sol2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol2)
enc2 = sol2.BidirectionalEncoder(vocab_size=20, embed_size=6, hidden_size=4, num_layers=1, dropout=0.0)
per_example_check(enc2, lambda m, s, l: m(s, l), "solution_2.BidirectionalEncoder", 4)

# ------------------------------------------------------------------ 3. forward() vs translate()
torch.manual_seed(2)
model = imp.create_model(30, 30, embed_dim=8, hidden_dim=12, layers=2, dropout=0.1).double().eval()
lengths = torch.tensor([5, 3, 4])
src = torch.zeros(3, 5, dtype=torch.long)
for b, L in enumerate(lengths):
    src[b, :L] = torch.randint(3, 30, (L,))
with torch.no_grad():
    toks, attn_tr = model.translate(src, lengths, max_len=6, sos_idx=1, eos_idx=2)
    trg = torch.cat([torch.full((3, 1), 1), toks], 1)     # [SOS, y_1, ..., y_T]
    logits_tf, attn_tf = model(src, lengths, trg)          # teacher-forced on translate()'s tokens
report("forward(): attention maps == translate() attention maps", (attn_tf - attn_tr).abs().max().item(), tol=0.0)
report("forward(): argmax logits == translate() tokens (mismatches)", float((logits_tf.argmax(-1) != toks).sum()), tol=0)
report("forward() output length == T-1, aligned with trg[:, 1:]", float(abs(logits_tf.size(1) - (trg.size(1) - 1))), tol=0)

# ------------------------------------------------------------------ 4. SOS/EOS convention of translate()
# checkpoint written by train_minimal.py (run from the day directory); another path can be passed as argv[1]
ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DAY, "best_model.pt")
sig = __import__("inspect").signature(imp.Seq2SeqWithAttention.translate)
d_sos, d_eos = sig.parameters['sos_idx'].default, sig.parameters['eos_idx'].default
print(f"[INFO] translate() defaults: sos_idx={d_sos}, eos_idx={d_eos}; "
      f"train_minimal.ReversalDataset uses PAD=0, SOS=1, EOS=2")
report("translate() default sos/eos match the training data convention (1, 2)",
       float((d_sos, d_eos) != (1, 2)), tol=0)
if os.path.exists(ckpt):
    import train_minimal as tm
    m = imp.create_model(100, 100, hidden_dim=256)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    random.seed(0)
    data = tm.ReversalDataset(num_samples=200, vocab_size=100)

    def acc(**kw):
        ok = 0
        for s_, t_ in data:
            p, _ = m.translate(s_[None], torch.tensor([len(s_)]), **kw)
            p = p[0].tolist()
            p = p[:p.index(2)] if 2 in p else p
            ok += p == t_[1:-1].tolist()
        return ok / len(data)
    a_old = acc(sos_idx=2, eos_idx=3)        # the pre-patch defaults
    a_fixed = acc(sos_idx=1, eos_idx=2)
    print(f"[INFO] exact-sequence accuracy on 200 reversal samples with checkpoint {ckpt}:")
    print(f"       old defaults (sos=2, eos=3) -> {a_old:.0%};  training convention (sos=1, eos=2) -> {a_fixed:.0%}")
else:
    print("[SKIP] no best_model.pt checkpoint; run train_minimal.py first for check 4")

# ------------------------------------------------------------------ 5. GRU cell form (paper A.1.1 vs nn.GRU)
# Paper: s~ = tanh(W x + U [r o s_prev] + C c)  (reset gate applied BEFORE the recurrent matmul).
# nn.GRU: n = tanh(W_in x + b_in + r o (W_hn h + b_hn))  (reset gate applied AFTER it).
torch.manual_seed(5)
gru = torch.nn.GRU(3, 4, batch_first=True).double()
xg, hg = torch.randn(2, 1, 3, dtype=torch.float64), torch.randn(1, 2, 4, dtype=torch.float64)
with torch.no_grad():
    ref_h = gru(xg, hg)[1][0]
    Wi, Wh, bi, bh = gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0
    gi, gh = xg[:, 0] @ Wi.T + bi, hg[0] @ Wh.T + bh
    r = torch.sigmoid(gi[:, :4] + gh[:, :4]); z = torch.sigmoid(gi[:, 4:8] + gh[:, 4:8])
    n_after = torch.tanh(gi[:, 8:] + r * gh[:, 8:])                                # torch form
    n_before = torch.tanh(gi[:, 8:] + (r * hg[0]) @ Wh[8:].T + bh[8:])             # paper form
    h_after = (1 - z) * n_after + z * hg[0]
    h_before = (1 - z) * n_before + z * hg[0]
report("nn.GRU uses the reset-after-matmul form", (h_after - ref_h).abs().max().item(), 1e-12)
print(f"[INFO] paper's reset-before-matmul form differs from nn.GRU by {(h_before - ref_h).abs().max().item():.3e} "
      "(same weights): the repo's GRU is a variant of the paper's gated unit")

n_ok = sum(ok for _, ok, _ in RESULTS)
print(f"\nSUMMARY: {n_ok}/{len(RESULTS)} checks pass")
