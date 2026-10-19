"""Day 13 -- does train_minimal.py's training loop ever change a parameter?

train_epoch() computes forward + loss + accuracy only; the SGD class is never
instantiated and `lr` is never used, so every weight is bit-identical after an
"epoch" and the loss stays at chance level (~ln(#token values)).
This script imports the training functions (does not run the script's main) on a
tiny model/data so it costs < 1 s.

Run: OMP_NUM_THREADS=1 python validation/day13_train_noop.py
"""
import copy
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
    return ps


np.random.seed(0)
src, tgt = tm.generate_copy_task_data(64, 6, vocab_size=12)
model = tm.SimpleTransformer(vocab_size=12, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout_p=0.1)
before = copy.deepcopy(all_params(model))
loss0 = tm.evaluate(model, src, tgt, batch_size=32)["loss"]
with contextlib.redirect_stdout(io.StringIO()):
    for _ in range(3):
        tm.train_epoch(model, src, tgt, batch_size=32, lr=0.1)
after = all_params(model)
loss1 = tm.evaluate(model, src, tgt, batch_size=32)["loss"]
max_change = max(float(np.abs(a - b).max()) for a, b in zip(after, before))
print(f"parameters checked: {len(before)} arrays, {sum(p.size for p in before)} values")
print(f"max |param change| after 3 'training' epochs at lr=0.1: {max_change}")
print(f"eval loss before = {loss0:.4f}, after = {loss1:.4f}; ln(9 token values) = {math.log(9):.4f}, ln(12) = {math.log(12):.4f}")
print("RESULT:", "FAIL - training loop is a no-op" if max_change == 0.0 else "parameters do change")
