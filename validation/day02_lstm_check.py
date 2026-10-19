"""Day 02 reference checks: hand-written NumPy LSTM (implementation.LSTM) and the exercise-1 solution.

1. Forward pass vs torch.nn.LSTMCell with copied weights. Gate-order mapping:
     repo:    W_f, W_i, W_c (candidate), W_o, each (H, H+V) acting on [h_{t-1}; x_t]
     PyTorch: weight_ih (4H, V) and weight_hh (4H, H) stacked in order (i, f, g, o)
   so weight_ih = [W_i[:, H:]; W_f[:, H:]; W_c[:, H:]; W_o[:, H:]] and weight_hh uses [:, :H].
2. backward() vs central finite differences of the summed cross-entropy (all 10 parameters).
3. backward() vs torch.autograd through nn.LSTMCell + nn.Linear (gradients mapped back).
4. exercise solution_01 (LSTMFromScratch, 1-D vectors) backward vs central differences.
5. setup.py quick test; exercise_01 test_lstm() harness with the solution class swapped in.

Run:  OMP_NUM_THREADS=1 python validation/day02_lstm_check.py
"""
import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '02_Understanding_LSTM')
sys.path.insert(0, DAY)
from implementation import LSTM  # noqa: E402

torch.set_default_dtype(torch.float64)
np.random.seed(1)
torch.manual_seed(1)
V, H, T = 5, 4, 6
m = LSTM(V, H, seq_length=T)
for n in ['Wf', 'Wi', 'Wc', 'Wo']:
    setattr(m, n, np.random.randn(H, H + V) * 0.5)      # nonlinear regime, grads stay < 5
for n in ['bf', 'bi', 'bc', 'bo']:
    setattr(m, n, np.random.randn(H, 1) * 0.3)
m.Why = np.random.randn(V, H) * 0.5
m.by = np.random.randn(V, 1) * 0.1
inputs = [int(i) for i in np.random.randint(0, V, T)]
targets = [int(i) for i in np.random.randint(0, V, T)]
h0 = np.random.randn(H, 1) * 0.2
c0 = np.random.randn(H, 1) * 0.2
PN = ['Wf', 'Wi', 'Wc', 'Wo', 'Why', 'bf', 'bi', 'bc', 'bo', 'by']
rows = []

# ---- 1. forward vs nn.LSTMCell ---------------------------------------------------------------
cell = torch.nn.LSTMCell(V, H)
with torch.no_grad():
    order = [m.Wi, m.Wf, m.Wc, m.Wo]                      # PyTorch chunk order: i, f, g, o
    cell.weight_ih.copy_(torch.from_numpy(np.vstack([W[:, H:] for W in order])))
    cell.weight_hh.copy_(torch.from_numpy(np.vstack([W[:, :H] for W in order])))
    cell.bias_ih.copy_(torch.from_numpy(np.vstack([m.bi, m.bf, m.bc, m.bo]).ravel()))
    cell.bias_hh.zero_()
loss_np, _, _, cache = m.forward(inputs, targets, h0, c0)
h = torch.from_numpy(h0.T)
c = torch.from_numpy(c0.T)
dh = dc = 0.0
for t in range(T):
    x = torch.zeros(1, V)
    x[0, inputs[t]] = 1.0
    h, c = cell(x, (h, c))
    dh = max(dh, np.abs(h.detach().numpy().ravel() - cache['hs'][t].ravel()).max())
    dc = max(dc, np.abs(c.detach().numpy().ravel() - cache['Cs'][t].ravel()).max())
rows.append(('forward h_t vs nn.LSTMCell', dh, dh < 1e-12))
rows.append(('forward C_t vs nn.LSTMCell', dc, dc < 1e-12))

# ---- 2. backward vs central differences ------------------------------------------------------
grads = m.backward(inputs, targets, cache)
gmax = max(np.abs(g).max() for g in grads.values())


def loss_only():
    return m.forward(inputs, targets, h0, c0)[0]       # forward() already returns the SUM over t


eps = 1e-5
worst = 0.0
for n in PN:
    P = getattr(m, n)
    num = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + eps
        lp = loss_only()
        P[idx] = old - eps
        lm = loss_only()
        P[idx] = old
        num[idx] = (lp - lm) / (2 * eps)
    worst = max(worst, np.abs(num - grads[n]).max())
rows.append(('backward vs central differences (10 params)', worst, worst < 1e-7))

# ---- 3. backward vs autograd through nn.LSTMCell ---------------------------------------------
cell.zero_grad()
lin = torch.nn.Linear(H, V)
with torch.no_grad():
    lin.weight.copy_(torch.from_numpy(m.Why))
    lin.bias.copy_(torch.from_numpy(m.by.ravel()))
h = torch.from_numpy(h0.T)
c = torch.from_numpy(c0.T)
loss_t = torch.zeros(())
for t in range(T):
    x = torch.zeros(1, V)
    x[0, inputs[t]] = 1.0
    h, c = cell(x, (h, c))
    loss_t = loss_t - torch.log_softmax(lin(h)[0], dim=0)[targets[t]]
loss_t.backward()
gih = cell.weight_ih.grad.numpy()
ghh = cell.weight_hh.grad.numpy()
gb = cell.bias_ih.grad.numpy()
worst_ag = 0.0
for k, (wn, bn) in enumerate([('Wi', 'bi'), ('Wf', 'bf'), ('Wc', 'bc'), ('Wo', 'bo')]):
    gW = np.hstack([ghh[k * H:(k + 1) * H], gih[k * H:(k + 1) * H]])   # back to [h | x] layout
    worst_ag = max(worst_ag, np.abs(gW - grads[wn]).max(),
                   np.abs(gb[k * H:(k + 1) * H] - grads[bn].ravel()).max())
worst_ag = max(worst_ag, np.abs(lin.weight.grad.numpy() - grads['Why']).max(),
               np.abs(lin.bias.grad.numpy() - grads['by'].ravel()).max())
rows.append(('backward vs autograd (nn.LSTMCell+Linear)', worst_ag, worst_ag < 1e-12))
rows.append(('summed loss vs torch', abs(loss_t.item() - loss_np), abs(loss_t.item() - loss_np) < 1e-12))

print('Day 02 implementation.LSTM reference checks (float64, V=%d H=%d T=%d)' % (V, H, T))
print('  largest |grad| = %.3f  (< 5, so the +/-5 clip is inactive)' % gmax)
for name, err, ok in rows:
    print('  %-46s max|d| = %.2e  %s' % (name, err, 'PASS' if ok else 'FAIL'))


# ---- 4. exercise solution_01 -----------------------------------------------------------------
def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(DAY, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sol = load('sol01', 'exercises/solutions/solution_01_lstm_implementation.py')
s = sol.LSTMFromScratch(V, H, V)
for n in ['Wf', 'Wi', 'Wc', 'Wo']:
    setattr(s, n, np.random.randn(H, H + V) * 0.5)
for n in ['bf', 'bi', 'bc', 'bo']:
    setattr(s, n, np.random.randn(H) * 0.3)
s.Wy = np.random.randn(V, H) * 0.5
s.by = np.random.randn(V) * 0.1
hs0, cs0 = np.random.randn(H) * 0.2, np.random.randn(H) * 0.2
s.forward(inputs, targets, hs0, cs0)
s.backward()
SN = ['Wf', 'Wi', 'Wc', 'Wo', 'Wy', 'bf', 'bi', 'bc', 'bo', 'by']
analytic = {n: getattr(s, 'd' + n).copy() for n in SN}
worst_s = 0.0
for n in SN:
    P = getattr(s, n)
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + eps
        lp = s.forward(inputs, targets, hs0, cs0)
        P[idx] = old - eps
        lm = s.forward(inputs, targets, hs0, cs0)
        P[idx] = old
        worst_s = max(worst_s, abs((lp - lm) / (2 * eps) - analytic[n][idx]))
print('  %-46s max|d| = %.2e  %s' % ('solution_01 backward vs central differences', worst_s,
                                       'PASS' if worst_s < 1e-6 else 'FAIL'))
print('    (solution adds 1e-8 inside log(), so agreement is ~1e-7, not machine precision)')

# ---- 5. setup quick test and exercise harness ------------------------------------------------
st = load('setup02', 'setup.py')
buf = io.StringIO()
with redirect_stdout(buf):
    ok = st.run_quick_test()
print('  setup.py run_quick_test() ->', ok, '|', buf.getvalue().strip().splitlines()[-1])
ex = load('ex01', 'exercises/exercise_01_build_lstm.py')
ex.LSTMFromScratch = sol.LSTMFromScratch
np.random.seed(0)
buf = io.StringIO()
with redirect_stdout(buf):
    ex.test_lstm()
lines = [ln for ln in buf.getvalue().splitlines() if 'Iteration' in ln]
print('  exercise_01.test_lstm() with solution class:', lines[0].strip(), '->', lines[-1].strip())
