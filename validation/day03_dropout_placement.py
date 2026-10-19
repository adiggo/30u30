"""Day 03 reference checks: Zaremba et al. (2014) dropout placement, plus layer-norm/dropout math.

Paper (Sec. 3.2, Eq. for the regularized LSTM): gates = T_{2n,4n}[ D(h_t^{l-1}) ; h_{t-1}^l ].
Dropout D touches only the non-recurrent (vertical) path; the recurrent h_{t-1} is never dropped.

Checks
 1. Placement property: with identical input-dropout masks, the recurrent trajectory (final c_T, h_T)
    must not depend on the masks used on the h -> output path. Run on the pre-fix implementation
    (loaded from git at ORIG_REF) and on the patched files in the working tree.
 2. Patched forward == an independent re-implementation of the paper's equations (same masks).
 3. layer_norm_backward vs torch.autograd through F.layer_norm.
 4. dropout_forward/backward: inverted scaling, E[out] = x, backward == autograd.
 5. Does train_minimal.train_epoch change any parameter? (it should, if it trained)
 6. Exercise-5 solution cell: fraction of exact zeros in the returned recurrent state (pre-fix vs patched).

Run:  OMP_NUM_THREADS=1 python validation/day03_dropout_placement.py
"""
import importlib.util
import io
import os
import subprocess
import sys
import types
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '03_RNN_Regularization')
# Pre-fix code for the original-vs-patched comparison: the commit the fixes on this branch were
# made on (read with `git show`, no checkout; needs the repo's git history).
ORIG_REF = '195dcf2'
sys.path.insert(0, DAY)
torch.set_default_dtype(torch.float64)


def module_from_source(name, src):
    mod = types.ModuleType(name)
    exec(compile(src, name, 'exec'), mod.__dict__)
    return mod


def head_source(rel):
    return subprocess.check_output(['git', '-C', REPO, 'show', ORIG_REF + ':' + rel], text=True)


impl_new = module_from_source('impl_new', open(DAY + '/implementation.py').read())
impl_head = module_from_source('impl_head', head_source('papers/03_RNN_Regularization/implementation.py'))

V, H, T, KEEP = 7, 6, 9, 0.5
rng = np.random.RandomState(0)
inputs = [int(i) for i in rng.randint(0, V, T)]
targets = [int(i) for i in rng.randint(0, V, T)]
in_masks = [(rng.rand(V, 1) < KEEP) / KEEP for _ in range(T)]
outA = [np.ones((H, 1)) for _ in range(T)]                         # output path untouched
outB = [(rng.rand(H, 1) < KEEP) / KEEP for _ in range(T)]          # output path dropped


def run(mod, out_masks, use_ln):
    """Run mod.RegularizedLSTM.forward with deterministic masks injected into dropout_forward."""
    calls = {'n': 0}

    def fake_dropout(x, keep_prob, training=True):
        t, which = divmod(calls['n'], 2)          # per step: call 0 = input x_t, call 1 = hidden h_t
        calls['n'] += 1
        mask = in_masks[t] if which == 0 else out_masks[t]
        return x * mask, mask

    np.random.seed(3)
    model = mod.RegularizedLSTM(V, H, V, dropout_keep_prob=KEEP, use_layer_norm=use_ln)
    for n in ['Wf', 'Wi', 'Wc', 'Wo']:
        setattr(model, n, np.random.RandomState(4).randn(H, H + V) * 0.6)
    model.Why = np.random.RandomState(5).randn(V, H) * 0.6
    saved = mod.dropout_forward
    mod.dropout_forward = fake_dropout
    try:
        loss, h, c = model.forward(inputs, targets, np.zeros((H, 1)), np.zeros((H, 1)), training=True)
    finally:
        mod.dropout_forward = saved
    return loss, h, c, model


print('1. Recurrent trajectory vs h->output dropout masks (identical input masks):')
for label, mod in [('pre-fix', impl_head), ('patched', impl_new)]:
    for use_ln in (True, False):
        _, hA, cA, _ = run(mod, outA, use_ln)
        _, hB, cB, _ = run(mod, outB, use_ln)
        d = max(np.abs(cA - cB).max(), np.abs(hA - hB).max())
        verdict = 'recurrent path untouched (PASS)' if d == 0 else 'recurrent path depends on dropout (FAIL)'
        print('   %s use_layer_norm=%-5s max|d(c_T,h_T)| = %.3e  %s' % (label, use_ln, d, verdict))
        lA, lB = run(mod, outA, use_ln)[0], run(mod, outB, use_ln)[0]
        print('   %s use_layer_norm=%-5s loss changes with output-path masks: %s'
              % (label, use_ln, 'yes (dropout reaches h->output)' if lA != lB
                 else 'NO -> h->output dropout is computed but discarded (FAIL)'))

# ---- 2. patched forward vs independent re-implementation of the paper's equations -------------
def sigm(z):
    return 1 / (1 + np.exp(-z))


def reference(model, out_masks, use_ln):
    h = np.zeros((H, 1))
    c = np.zeros((H, 1))
    loss = 0.0
    for t in range(T):
        x = np.zeros((V, 1))
        x[inputs[t]] = 1
        z = np.vstack([h, x * in_masks[t]])                 # [h_{t-1} ; D(x_t)]  (recurrent h clean)
        f, i = sigm(model.Wf @ z + model.bf), sigm(model.Wi @ z + model.bi)
        g, o = np.tanh(model.Wc @ z + model.bc), sigm(model.Wo @ z + model.bo)
        c = f * c + i * g
        h = o * np.tanh(c)
        u = h * out_masks[t]                                 # D(h_t) on the h -> softmax path only
        if use_ln:
            u = (u - u.mean()) / np.sqrt(u.var() + 1e-5) * model.gamma_h + model.beta_h
        y = model.Why @ u + model.by
        p = np.exp(y - y.max())
        p /= p.sum()
        loss += -np.log(p[targets[t], 0] + 1e-8)
    return loss, h, c


print('2. Patched forward vs independent paper-equation reference (same masks):')
for use_ln in (True, False):
    loss, h, c, model = run(impl_new, outB, use_ln)
    rl, rh, rc = reference(model, outB, use_ln)
    d = max(abs(loss - rl), np.abs(h - rh).max(), np.abs(c - rc).max())
    print('   use_layer_norm=%-5s max|d(loss,h_T,c_T)| = %.2e  %s' % (use_ln, d, 'PASS' if d < 1e-12 else 'FAIL'))

# ---- 3. layer-norm backward vs autograd --------------------------------------------------------
N, D = 4, 7
x = np.random.RandomState(6).randn(N, D) * 2 + 1
gamma = np.random.RandomState(7).randn(D)
beta = np.random.RandomState(8).randn(D)
dout = np.random.RandomState(9).randn(N, D)
out, cache = impl_new.layer_norm_forward(x, gamma, beta)
dx, dg, db = impl_new.layer_norm_backward(dout, cache)
xt, gt, bt = (torch.tensor(a, requires_grad=True) for a in (x, gamma, beta))
yt = F.layer_norm(xt, (D,), gt, bt, eps=1e-5)
yt.backward(torch.from_numpy(dout))
d_fwd = np.abs(yt.detach().numpy() - out).max()
d_bwd = max(np.abs(xt.grad.numpy() - dx).max(), np.abs(gt.grad.numpy() - dg).max(),
            np.abs(bt.grad.numpy() - db).max())
print('3. layer_norm forward vs F.layer_norm   max|d| = %.2e  %s' % (d_fwd, 'PASS' if d_fwd < 1e-12 else 'FAIL'))
print('   layer_norm backward vs autograd      max|d| = %.2e  %s' % (d_bwd, 'PASS' if d_bwd < 1e-12 else 'FAIL'))

# ---- 4. dropout forward/backward ---------------------------------------------------------------
np.random.seed(10)
xs = np.random.RandomState(11).randn(200000)
o, mask = impl_new.dropout_forward(xs, 0.8, training=True)
vals = np.unique(np.round(mask, 12))
rel_mean = abs(o.mean() - xs.mean()) / np.abs(xs).mean()
mt = torch.from_numpy(mask)
xt = torch.tensor(xs, requires_grad=True)
g = torch.from_numpy(np.random.RandomState(12).randn(xs.size))
(xt * mt).backward(g)
d_bw = np.abs(impl_new.dropout_backward(g.numpy(), mask) - xt.grad.numpy()).max()
o_eval, m_eval = impl_new.dropout_forward(xs, 0.8, training=False)
print('4. dropout mask values = %s (expect {0, 1/0.8=1.25})' % vals)
print('   keep fraction = %.4f (expect 0.8); |E[out]-E[x]|/E|x| = %.4f' % ((mask > 0).mean(), rel_mean))
print('   dropout_backward vs autograd max|d| = %.2e  %s; eval mode identity: %s'
      % (d_bw, 'PASS' if d_bw == 0 else 'FAIL', np.array_equal(o_eval, xs) and m_eval is None))

# ---- 5. does the training loop train? ----------------------------------------------------------
spec = importlib.util.spec_from_file_location('tm03', DAY + '/train_minimal.py')
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
np.random.seed(0)
model = tm.RegularizedLSTM(vocab_size=26, hidden_size=16, output_size=26, dropout_keep_prob=0.8)
before = {k: v.copy() for k, v in vars(model).items() if isinstance(v, np.ndarray)}
data, tgts = tm.generate_simple_dataset(seq_length=20, vocab_size=26, num_samples=10)
cfg = tm.RegularizationConfig()
with redirect_stdout(io.StringIO()):
    l1 = tm.train_epoch(model, data, tgts, cfg, training=True)
    l2 = tm.train_epoch(model, data, tgts, cfg, training=True)
changed = [k for k, v in before.items() if not np.array_equal(v, getattr(model, k))]
print('5. train_epoch: parameters changed after 2 epochs: %s  (losses %.2f, %.2f; ln(26)*19 = %.2f)'
      % (changed if changed else 'NONE -> no backward pass, nothing is learned (FAIL)', l1, l2,
         np.log(26) * 19))
print('   RegularizedLSTM defines a backward method:', hasattr(model, 'backward'))

# ---- 6. exercise-5 solution cell ---------------------------------------------------------------
sol_head = module_from_source('sol5_head', head_source(
    'papers/03_RNN_Regularization/exercises/solutions/05_full_pipeline.py'))
spec = importlib.util.spec_from_file_location('sol5_new', DAY + '/exercises/solutions/05_full_pipeline.py')
sol_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol_new)
print('6. Exercise-5 solution cell, dropout_rate=0.5, training=True:')
for label, mod in [('pre-fix', sol_head), ('patched', sol_new)]:
    np.random.seed(13)
    cell = mod.RegularizedRNNCell(input_size=4, hidden_size=50, dropout_rate=0.5)
    h = np.zeros((1, 50))
    zeros = []
    for t in range(20):
        h = cell.forward(np.random.randn(1, 4), h, training=True)  # returned h is the next h_prev
        zeros.append(np.mean(h == 0))
    print('   %s fraction of exact zeros in the recurrent state fed to the next step: %.2f  %s'
          % (label, np.mean(zeros), 'recurrent dropout (FAIL)' if np.mean(zeros) > 0.1 else 'PASS'))

# ---- 7. train_minimal.py status column (unit-level; train_minimal.py itself is not run here)
mon = impl_new.EarlyStoppingMonitor(patience=5, verbose=False)
old_col, new_col = [], []
for epoch, v in enumerate([3.0, 2.5, 2.6, 2.4, 2.45]):
    mon.check(v, epoch)
    old_col.append('IMPROVED' if v < mon.best_loss else 'No improve')         # pre-fix expression
    new_col.append('IMPROVED' if mon.no_improve_count == 0 else 'No improve')  # patched expression
print('7. status column for val losses [3.0, 2.5, 2.6, 2.4, 2.45]:')
print('   pre-fix:', old_col)
print('   patched:', new_col)
