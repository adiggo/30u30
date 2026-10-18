"""Day 03: run every exercise file's own test functions with the reference solution swapped in
(and, for exercises 1-2, with the repo's implementation.py functions too).

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
np.random.seed(0)
cell = s5.RegularizedRNNCell(3, 8, dropout_rate=0.2)
w0 = {k: v.copy() for k, v in cell.get_weights().items()}
with redirect_stdout(io.StringIO()):
    s5.train_regularized_rnn(cell, [np.random.randn(5, 3)] * 4, [np.random.randn(1, 8)] * 4,
                             [np.random.randn(5, 3)] * 2, [np.random.randn(1, 8)] * 2, max_epochs=3)
same = all(np.array_equal(w0[k], v) for k, v in cell.get_weights().items())
print('  ex05 solution train_regularized_rnn updates weights: %s' % ('NO (forward + loss only)' if same else 'yes'))
