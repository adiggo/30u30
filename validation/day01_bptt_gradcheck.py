"""Day 01 reference checks: hand-written NumPy BPTT in implementation.CharRNN.

1. Forward pass vs torch.nn.RNN (tanh) with copied weights.
2. backward() vs central finite differences of the summed cross-entropy.
3. backward() vs torch.autograd on an independent float64 re-implementation.
4. Unit consistency of train_minimal.py's smoothed loss (per-sequence init vs per-character loss).
5. exercise_02_solution's temperature table vs the true softmax values.
6. setup.py quick test (calls forward with the wrong signature before the patch).

Run:  OMP_NUM_THREADS=1 python validation/day01_bptt_gradcheck.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '01_Unreasonable_Effectiveness')
sys.path.insert(0, DAY)
from implementation import CharRNN  # noqa: E402

torch.set_default_dtype(torch.float64)
np.random.seed(0)
torch.manual_seed(0)

V, H, T = 6, 5, 8
m = CharRNN(V, hidden_size=H, learning_rate=0.1, seq_length=T)
# Larger-than-default weights so tanh is in its nonlinear regime; gradients stay well below the
# +/-5 clip used inside backward(), so the clipped and true gradients coincide.
m.Wxh = np.random.randn(H, V) * 0.5
m.Whh = np.random.randn(H, H) * 0.5
m.Why = np.random.randn(V, H) * 0.5
m.bh = np.random.randn(H, 1) * 0.1
m.by = np.random.randn(V, 1) * 0.1
inputs = [int(i) for i in np.random.randint(0, V, T)]
targets = [int(i) for i in np.random.randint(0, V, T)]
h0 = np.random.randn(H, 1) * 0.1
NAMES = ['Wxh', 'Whh', 'Why', 'bh', 'by']
results = []


def summed_loss():
    _, _, _, ps = m.forward(inputs, h0)
    # model.loss() returns the per-character MEAN; backward() differentiates the SUM over time.
    return m.loss(ps, targets) * len(targets)


# ---- 1. forward vs torch.nn.RNN -------------------------------------------------------------
rnn = torch.nn.RNN(V, H, nonlinearity='tanh', bias=True)
with torch.no_grad():
    rnn.weight_ih_l0.copy_(torch.from_numpy(m.Wxh))
    rnn.weight_hh_l0.copy_(torch.from_numpy(m.Whh))
    rnn.bias_ih_l0.copy_(torch.from_numpy(m.bh.ravel()))
    rnn.bias_hh_l0.zero_()
x_onehot = torch.zeros(T, 1, V)
x_onehot[torch.arange(T), 0, torch.tensor(inputs)] = 1.0
out, _ = rnn(x_onehot, torch.from_numpy(h0.T).unsqueeze(0))
xs, hs, ys, ps = m.forward(inputs, h0)
hs_np = np.stack([hs[t].ravel() for t in range(T)])          # (T, H)
d_fwd = np.abs(hs_np - out[:, 0, :].detach().numpy()).max()
logits_t = out[:, 0, :] @ torch.from_numpy(m.Why).T + torch.from_numpy(m.by.ravel())
p_t = torch.softmax(logits_t, dim=-1).detach().numpy()
d_p = np.abs(np.stack([ps[t].ravel() for t in range(T)]) - p_t).max()
results.append(('forward h_t vs nn.RNN', d_fwd, d_fwd < 1e-12))
results.append(('softmax p_t vs torch', d_p, d_p < 1e-12))

# ---- 2. backward vs central finite differences ----------------------------------------------
grads = m.backward(xs, hs, ps, targets)
max_abs_grad = max(np.abs(g).max() for g in grads)
eps = 1e-5
worst_fd = 0.0
for name, g in zip(NAMES, grads):
    P = getattr(m, name)
    num = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + eps
        lp = summed_loss()
        P[idx] = old - eps
        lm = summed_loss()
        P[idx] = old
        num[idx] = (lp - lm) / (2 * eps)
    worst_fd = max(worst_fd, np.abs(num - g).max())
results.append(('backward vs central differences (all 5 params)', worst_fd, worst_fd < 1e-7))

# ---- 3. backward vs torch.autograd -----------------------------------------------------------
tp = {n: torch.tensor(getattr(m, n), requires_grad=True) for n in NAMES}
h = torch.from_numpy(h0)
loss = torch.zeros(())
for t in range(T):
    x = torch.zeros(V, 1)
    x[inputs[t]] = 1.0
    h = torch.tanh(tp['Wxh'] @ x + tp['Whh'] @ h + tp['bh'])
    y = tp['Why'] @ h + tp['by']
    loss = loss - torch.log_softmax(y.ravel(), dim=0)[targets[t]]
loss.backward()
worst_ag = max(np.abs(tp[n].grad.numpy() - g).max() for n, g in zip(NAMES, grads))
d_loss = abs(loss.item() - summed_loss())
results.append(('backward vs torch.autograd (all 5 params)', worst_ag, worst_ag < 1e-12))
results.append(('summed loss vs torch', d_loss, d_loss < 1e-12))

print('Day 01 CharRNN reference checks (float64, V=%d H=%d T=%d)' % (V, H, T))
print('  largest |grad| = %.3f  (< 5, so the in-place +/-5 clip is inactive)' % max_abs_grad)
for name, err, ok in results:
    print('  %-48s max|d| = %.2e  %s' % (name, err, 'PASS' if ok else 'FAIL'))

# ---- 4. smoothed-loss units in train_minimal.py ---------------------------------------------
seq_len = 25
init = -np.log(1.0 / V) * seq_len                 # what train_minimal.py started from before the fix
per_char = -np.log(1.0 / V)                       # what model.loss() actually returns at init
s = init
for _ in range(100):                              # default --epochs 100 == 100 updates
    s = 0.999 * s + 0.001 * per_char
print('\nSmoothed-loss units (train_minimal.py before the fix):')
print('  initial smooth_loss = %.2f (per-sequence sum), model.loss() at init ~ %.2f (per char)'
      % (init, per_char))
print('  after 100 updates with a constant per-char loss the printed value is still %.2f '
      '(%.0f%% of the init) -> the printed "Loss" mixes units' % (s, 100 * s / init))

# ---- 5. exercise_02_solution temperature table ----------------------------------------------
logits = np.array([2.0, 1.0, 0.5])
claimed = {0.5: [0.72, 0.22, 0.06], 1.0: [0.59, 0.24, 0.17], 2.0: [0.44, 0.31, 0.25]}
print('\nexercise_02_solution temperature table (logits [2, 1, 0.5]):')
for temp, cl in claimed.items():
    z = logits / temp
    p = np.exp(z - z.max())
    p /= p.sum()
    print('  T=%.1f  true=%s  claimed=%s  max|d|=%.2f' % (temp, np.round(p, 3), cl, np.abs(p - cl).max()))

# ---- 6. setup.py quick test -----------------------------------------------------------------
spec = importlib.util.spec_from_file_location('day01_setup', os.path.join(DAY, 'setup.py'))
mod = importlib.util.module_from_spec(spec)
cwd = os.getcwd()
os.chdir(DAY)
try:
    spec.loader.exec_module(mod)
    ok = mod.run_quick_test()
finally:
    os.chdir(cwd)
print('\nsetup.py run_quick_test() returned:', ok)
