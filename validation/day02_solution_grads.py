"""Day 02: gradient checks for the backward passes that the exercise solutions train with.

1. exercise_03_solution.VanillaRNN.backward (new BPTT).
2. exercise_03_solution.LSTMNoForget / LSTMNoInput / LSTMNoOutput: implementation.LSTM.backward reused on
   a forward pass with one gate fixed to 1. Correct only because sigma' is computed as g * (1 - g) from
   the stored gate value, which is exactly 0 for a constant 1; this checks it, and that the fixed gate's
   weights get exactly zero gradient.
3. exercise_05_solution.GRU.backward (reset gate applied before the matmul, as in Colah's post).

Each is compared, in float64, with torch.autograd on an independent re-implementation of the same
equations, and with central finite differences of the summed cross-entropy. The tolerance is a
relative error (max |analytic - reference| / max |reference|, per parameter) of 1e-6 vs autograd
and 1e-6 vs finite differences. Weights are drawn large enough for the nonlinearities to matter but
gradients stay below the +/-5 clip inside backward(), so the clip is inactive (printed).

Run:  OMP_NUM_THREADS=1 python validation/day02_solution_grads.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '02_Understanding_LSTM')
sys.path.insert(0, DAY)
torch.set_default_dtype(torch.float64)


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(DAY, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s3 = load('s3', 'exercises/solutions/exercise_03_solution.py')
s5 = load('s5', 'exercises/solutions/exercise_05_solution.py')

V, H, T = 5, 4, 7
rng = np.random.RandomState(3)
inputs = [int(i) for i in rng.randint(0, V, T)]
targets = [int(i) for i in rng.randint(0, V, T)]
h0 = rng.randn(H, 1) * 0.3
c0 = rng.randn(H, 1) * 0.3
TOL = 1e-6
rows = []


def rel_err(a, ref):
    return np.abs(a - ref).max() / max(np.abs(ref).max(), 1e-12)


def randomize(model, names, scale=0.6):
    for n in names:
        P = getattr(model, n)
        P[...] = rng.randn(*P.shape) * scale                       # in-place, keeps the array object


def central_diff(loss_fn, model, names, eps=1e-6):
    out = {}
    for n in names:
        P = getattr(model, n)
        num = np.zeros_like(P)
        for idx in np.ndindex(P.shape):                          # every entry of the parameter
            old = P[idx]
            P[idx] = old + eps
            lp = loss_fn()
            P[idx] = old - eps
            lm = loss_fn()
            P[idx] = old
            num[idx] = (lp - lm) / (2 * eps)
        out[n] = num
    return out


def onehot(i):
    x = torch.zeros(V, 1)
    x[i] = 1.0
    return x


def report(label, grads, ag, fd, names):
    worst_ag = max(rel_err(grads[n], ag[n]) for n in names)
    worst_fd = max(rel_err(grads[n], fd[n]) for n in names)
    gmax = max(np.abs(grads[n]).max() for n in names)
    rows.append((label + ' vs autograd', worst_ag, worst_ag < TOL))
    rows.append((label + ' vs central differences', worst_fd, worst_fd < TOL))
    return gmax


# ---- 1. VanillaRNN ---------------------------------------------------------------------------
rnn = s3.VanillaRNN(V, H, T)
RN = ['Wh', 'bh', 'Why', 'by']
randomize(rnn, RN)
_, _, _, cache = rnn.forward(inputs, targets, h0)
g_rnn = rnn.backward(inputs, targets, cache)
tp = {n: torch.tensor(getattr(rnn, n), requires_grad=True) for n in RN}
h = torch.from_numpy(h0)
loss = torch.zeros(())
for t in range(T):
    h = torch.tanh(tp['Wh'] @ torch.cat([h, onehot(inputs[t])]) + tp['bh'])
    loss = loss - torch.log_softmax((tp['Why'] @ h + tp['by']).ravel(), dim=0)[targets[t]]
loss.backward()
ag = {n: tp[n].grad.numpy() for n in RN}
fd = central_diff(lambda: rnn.forward(inputs, targets, h0)[0], rnn, RN)
gmax = [report('VanillaRNN.backward', g_rnn, ag, fd, RN)]

# ---- 2. gate-ablated LSTMs (inherited LSTM.backward) -------------------------------------------
LN = ['Wf', 'Wi', 'Wc', 'Wo', 'Why', 'bf', 'bi', 'bc', 'bo', 'by']
zero_ok = True
for cls, gate in ((s3.LSTMNoForget, 'f'), (s3.LSTMNoInput, 'i'), (s3.LSTMNoOutput, 'o')):
    m = cls(V, H, T)
    randomize(m, LN)
    _, _, _, cache = m.forward(inputs, targets, h0, c0)
    g = m.backward(inputs, targets, cache)
    zero_ok &= (np.abs(g['W' + gate]).max() == 0.0) and (np.abs(g['b' + gate]).max() == 0.0)
    tp = {n: torch.tensor(getattr(m, n), requires_grad=True) for n in LN}
    h, c = torch.from_numpy(h0), torch.from_numpy(c0)
    loss = torch.zeros(())
    for t in range(T):
        z = torch.cat([h, onehot(inputs[t])])
        gates = {k: (torch.ones(H, 1) if k == gate else torch.sigmoid(tp['W' + k] @ z + tp['b' + k]))
                 for k in ('f', 'i', 'o')}
        c = gates['f'] * c + gates['i'] * torch.tanh(tp['Wc'] @ z + tp['bc'])
        h = gates['o'] * torch.tanh(c)
        loss = loss - torch.log_softmax((tp['Why'] @ h + tp['by']).ravel(), dim=0)[targets[t]]
    loss.backward()
    # the fixed gate's weights are not used by torch at all, so their .grad stays None (= zero)
    ag = {n: (tp[n].grad.numpy() if tp[n].grad is not None else np.zeros_like(getattr(m, n))) for n in LN}
    fd = central_diff(lambda: m.forward(inputs, targets, h0, c0)[0], m, LN)
    gmax.append(report(f'{cls.__name__} (LSTM.backward reused)', g, ag, fd, LN))
rows.append(('fixed gate weights get exactly zero gradient (3 variants)', 0.0 if zero_ok else 1.0,
             zero_ok))

# ---- 3. GRU ----------------------------------------------------------------------------------
gru = s5.GRU(V, H, V)
GN = ['Wz', 'bz', 'Wr', 'br', 'Wh', 'bh', 'Why', 'by']
randomize(gru, GN)
gru.forward(inputs, targets, h0)
g_gru = gru.backward()
tp = {n: torch.tensor(getattr(gru, n), requires_grad=True) for n in GN}
h = torch.from_numpy(h0)
loss = torch.zeros(())
for t in range(T):
    x = onehot(inputs[t])
    hx = torch.cat([h, x])
    zg = torch.sigmoid(tp['Wz'] @ hx + tp['bz'])
    rg = torch.sigmoid(tp['Wr'] @ hx + tp['br'])
    cand = torch.tanh(tp['Wh'] @ torch.cat([rg * h, x]) + tp['bh'])
    h = (1 - zg) * h + zg * cand
    loss = loss - torch.log_softmax((tp['Why'] @ h + tp['by']).ravel(), dim=0)[targets[t]]
loss.backward()
ag = {n: tp[n].grad.numpy() for n in GN}
fd = central_diff(lambda: gru.forward(inputs, targets, h0)[0], gru, GN)
gmax.append(report('GRU.backward', g_gru, ag, fd, GN))

print('Day 02 solution backward passes (float64, V=%d H=%d T=%d; tolerance %.0e relative)' % (V, H, T, TOL))
print('  largest |grad| = %.3f  (< 5, so the +/-5 clip inside backward() is inactive)' % max(gmax))
for name, err, ok in rows:
    print('  %-62s rel err = %.1e  %s' % (name, err, 'PASS' if ok else 'FAIL'))
print('ALL PASS' if all(ok for _, _, ok in rows) else 'SOME FAILED')
