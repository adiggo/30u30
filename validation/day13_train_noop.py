"""Day 13 -- does train_minimal.py's training loop change the parameters (and lower the loss)?

History: until 2026-10-18 train_epoch() computed forward + loss + accuracy only (there was no
backward pass, the SGD class was never used and `lr` was ignored), so every weight stayed
bit-identical and the loss stayed at chance level: this script reported FAIL (max change 0.0).
Since the open-item fix, implementation.py has a backward pass and train_epoch() takes an Adam
optimizer (Sec. 5.3 schedule), so the expected result is PASS.

The parameter list below walks the model's attributes directly (not model.parameters()), so it
also checks that the optimizer updates, in place, the very arrays the layers compute with.
Imports the training functions (does not run the script's main) on a tiny model/data: ~2 s.

Run: OMP_NUM_THREADS=1 python validation/day13_train_noop.py
"""
import contextlib
import io
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "13_attention")
sys.path.insert(0, DAY)
import train_minimal as tm  # noqa: E402


def all_params(model):
    t = model.model
    ps = [t.src_embedding.embedding, t.tgt_embedding.embedding, t.output_projection]
    for blk in t.encoder_layers + t.decoder_layers:
        for att in [blk.self_attention] + ([blk.cross_attention] if hasattr(blk, "cross_attention") else []):
            ps += [att.W_Q, att.W_K, att.W_V, att.W_O]
        ff = blk.feed_forward
        ps += [ff.W1, ff.b1, ff.W2, ff.b2]
        for nm in ("norm1", "norm2", "norm3"):
            if hasattr(blk, nm):
                ps += [getattr(blk, nm).gamma, getattr(blk, nm).beta]
    return ps


np.random.seed(0)
src, tgt = tm.generate_copy_task_data(64, 6, vocab_size=12)
model = tm.SimpleTransformer(vocab_size=12, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout_p=0.1)
optimizer = tm.Adam(model.parameters(), d_model=16, warmup=20)
before = [p.copy() for p in all_params(model)]
loss0 = tm.evaluate(model, src, tgt, batch_size=32)["loss"]
epochs = 30
with contextlib.redirect_stdout(io.StringIO()):
    for _ in range(epochs):
        tm.train_epoch(model, src, tgt, optimizer, batch_size=32)
after = all_params(model)
loss1 = tm.evaluate(model, src, tgt, batch_size=32)["loss"]
changes = [float(np.abs(a - b).max()) for a, b in zip(after, before)]
print(f"parameters checked: {len(before)} arrays, {sum(p.size for p in before)} values "
      f"(model.parameters() lists {len(model.parameters())} arrays)")
print(f"after {epochs} epochs ({optimizer.t} Adam steps): max |param change| = {max(changes):.3e}, "
      f"arrays that did not change: {sum(c == 0.0 for c in changes)}")
print(f"eval loss before = {loss0:.4f}, after = {loss1:.4f}; ln(9 token values) = {math.log(9):.4f}, "
      f"ln(12) = {math.log(12):.4f}")
# loss must fall by >= 10% within 60 Adam steps (it falls ~20% here; a no-op loop changes nothing)
ok = min(changes) > 0.0 and loss1 < 0.9 * loss0 and len(model.parameters()) == len(before)
print("RESULT:", "PASS - every parameter array changes and the loss falls" if ok
      else "FAIL - training loop does not update every parameter or does not lower the loss")
