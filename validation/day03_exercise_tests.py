"""Day 03: run every exercise file's own test functions with the reference solution swapped in
(and, for exercises 1-2, with the repo's implementation.py functions too).

[2026-10-18] Exercise 5's pipeline now trains the paper's LSTM language model on a Penn Treebank
slice, so this script needs PTB: set $PTB_DIR to a folder with ptb.{train,valid,test}.txt, or
train_minimal.load_ptb downloads it (~6 MB) into the day's data/ptb/.

Run:  OMP_NUM_THREADS=1 python validation/day03_exercise_tests.py
"""
import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '03_RNN_Regularization')
EX = DAY + '/exercises'
sys.path.insert(0, DAY)
os.environ.setdefault('MPLBACKEND', 'Agg')
import implementation as impl  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def run_tests(label, mod, names):
    ok = 0
    for n in names:
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                getattr(mod, n)()
            ok += 1
        except Exception as e:  # noqa: BLE001
            print('    FAIL %s.%s: %s: %s' % (label, n, type(e).__name__, str(e).splitlines()[0][:90]))
    print('  %-44s %d/%d tests pass' % (label, ok, len(names)))
    return ok, len(names)


# ---- exercise 1: dropout --------------------------------------------------------------------
e1 = load('e1', EX + '/exercise_01_dropout.py')
s1 = load('s1', EX + '/solutions/01_dropout.py')
T1 = ['test_dropout_forward', 'test_dropout_backward', 'test_dropout_integration']


def sol_dropout_forward(x, keep_prob, training=True):
    if not training or keep_prob == 1.0:
        return x, None
    d = s1.Dropout(rate=1 - keep_prob)
    out = d.forward(x, training=True)
    return out, d.mask / keep_prob              # exercise API: mask already scaled by 1/keep


def sol_dropout_backward(dout, mask):
    if mask is None:
        return dout
    nz = mask[mask > 0]
    keep = 1.0 / nz[0] if nz.size else 1.0
    d = s1.Dropout(rate=1 - keep)
    d.mask = (mask > 0).astype(float)
    return d.backward(dout)


e1.dropout_forward, e1.dropout_backward = sol_dropout_forward, sol_dropout_backward
run_tests('ex01 tests + solution Dropout class', e1, T1)
e1.dropout_forward, e1.dropout_backward = impl.dropout_forward, impl.dropout_backward
run_tests('ex01 tests + implementation.py functions', e1, T1)

# ---- exercise 2: layer norm -----------------------------------------------------------------
e2 = load('e2', EX + '/exercise_02_normalization.py')
s2 = load('s2', EX + '/solutions/02_normalization.py')
T2 = ['test_layer_norm_forward', 'test_layer_norm_backward']


def sol_ln_forward(x, gamma, beta, eps=1e-5):
    ln = s2.LayerNorm(x.shape[-1], epsilon=eps)
    ln.gamma, ln.beta = gamma, beta
    return ln.forward(x), ln


def sol_ln_backward(dout, cache):
    return cache.backward(dout)


e2.layer_norm_forward, e2.layer_norm_backward = sol_ln_forward, sol_ln_backward
run_tests('ex02 tests + solution LayerNorm class', e2, T2)
e2.layer_norm_forward, e2.layer_norm_backward = impl.layer_norm_forward, impl.layer_norm_backward
run_tests('ex02 tests + implementation.py functions', e2, T2)

# ---- exercise 3: weight decay ---------------------------------------------------------------
e3 = load('e3', EX + '/exercise_03_weight_decay.py')
s3 = load('s3', EX + '/solutions/03_weight_decay.py')
for n in ['compute_l2_penalty', 'compute_l2_gradient', 'regularized_loss', 'apply_weight_decay_to_gradient']:
    setattr(e3, n, getattr(s3, n))
run_tests('ex03 tests + solution functions', e3,
          ['test_compute_l2_penalty', 'test_compute_l2_gradient', 'test_regularized_loss',
           'test_apply_weight_decay_to_gradient'])

# ---- exercise 4: early stopping -------------------------------------------------------------
e4 = load('e4', EX + '/exercise_04_early_stopping.py')
s4 = load('s4', EX + '/solutions/04_early_stopping.py')
e4.EarlyStopping = s4.EarlyStopping
e4.simulate_training_with_early_stopping = s4.simulate_training_with_early_stopping
run_tests('ex04 tests + solution EarlyStopping', e4,
          ['test_early_stopping_basic', 'test_min_delta', 'test_weight_saving', 'test_reset',
           'test_simulation'])

# ---- exercise 5: full pipeline --------------------------------------------------------------
e5 = load('e5', EX + '/exercise_05_full_pipeline.py')
s5 = load('s5', EX + '/solutions/05_full_pipeline.py')
for n in ['RegularizedRNNCell', 'EarlyStopping', 'train_regularized_rnn']:
    setattr(e5, n, getattr(s5, n))
run_tests('ex05 tests + solution pipeline', e5,
          ['test_rnn_cell_initialization', 'test_layer_norm', 'test_dropout', 'test_forward_pass',
           'test_l2_penalty', 'test_weight_save_restore', 'test_full_pipeline'])
# [PATCH 2026-10-18] the solution's loop now trains LSTMLanguageModel (it used to compute losses only)
tr5, va5, v5 = s5.load_ptb_slice(n_train=6000, n_valid=3000)
lm5 = s5.LSTMLanguageModel(v5, hidden_size=32, num_layers=2, keep_prob=0.8, init_scale=0.1, seed=0)
w0 = lm5.get_state()
ppl0 = s5.evaluate(lm5, va5, 10, 20)
with redirect_stdout(io.StringIO()):
    h5 = s5.train_regularized_rnn(lm5, s5.batchify(tr5, 10), va5, max_epochs=2, weight_decay=1e-5)
n_changed = sum(not np.array_equal(w0[k], lm5.params[k]) for k in w0)
print('  ex05 solution train_regularized_rnn updates weights: %s (%d/%d arrays changed; validation '
      'perplexity %.0f -> %.0f on a 6k-word PTB slice)' % ('yes' if n_changed == len(w0) else 'NO', n_changed,
                                                         len(w0), ppl0, np.exp(min(h5['val_loss']))))
