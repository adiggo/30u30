"""Day 12 (Dropout, Srivastava et al. 2014) reference checks.

    OMP_NUM_THREADS=1 python validation/day12_dropout_checks.py [day_dir]

Checks
 1. Inverted dropout (implementation.Dropout, train_minimal.Dropout, solution_01.Dropout):
    E[y] = x in training (Monte Carlo z-score), exact identity at eval.
 2. Same output as torch.nn.functional.dropout when given torch's mask (p_keep = 1 - p_torch).
 3. Dropout.backward vs central finite differences (fixed mask).
 4. train_minimal.DropoutMLP hand-written NumPy backprop vs torch autograd with identical masks/weights.
 5. MC dropout on a linear read-out: sample mean/variance vs closed form
    E = w.x, Var = (1-p)/p * sum_i w_i^2 x_i^2.
 6. Paper Sec. 9.1 (marginalizing dropout for linear regression):
    E_R ||y - (R*X)w||^2 = ||y - pXw||^2 + p(1-p)||Gamma w||^2, Gamma = diag(X^T X)^(1/2).
 7. Exercise 1 test functions run against solution_01.Dropout.
 8. implementation.AlphaDropout: output mean/var for N(0,1) input vs torch alpha_dropout (should be 0/1).
 9. implementation.DropConnect.backward grad_W vs autograd.
"""
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'papers', '12_dropout')
sys.path.insert(0, DAY)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


impl = load(os.path.join(DAY, 'implementation.py'), 'impl12')
tm = load(os.path.join(DAY, 'train_minimal.py'), 'tm12')
sol1 = load(os.path.join(DAY, 'exercises/solutions/solution_01_build_dropout.py'), 'sol1_12')
ex1 = load(os.path.join(DAY, 'exercises/exercise_01_build_dropout.py'), 'ex1_12')

rng = np.random.default_rng(0)
np.random.seed(0)
summary = []

# ------------------------------------------------------------------ 1. expectation + eval identity
for name, cls in [('implementation.Dropout', impl.Dropout), ('train_minimal.Dropout', tm.Dropout),
                  ('solution_01.Dropout', sol1.Dropout)]:
    for p in (0.5, 0.8):
        d = cls(p=p)
        d.training = True
        x = rng.standard_normal(64)                     # (64,) one fixed input vector
        T = 20000
        X = np.tile(x, (T, 1))                          # np.tile -> (T, 64): T copies of x, one per MC draw
        Y = d.forward(X)                                # independent Bernoulli mask per element
        mean = Y.mean(axis=0)                           # axis=0 averages over the T draws -> (64,)
        se = np.abs(x) * np.sqrt((1 - p) / p) / np.sqrt(T)   # std error of the MC mean, per element
        z = np.max(np.abs(mean - x) / np.maximum(se, 1e-12))
        d.training = False
        exact = np.array_equal(d.forward(x), x)
        summary.append((f'{name} p={p}: E[y]=x (max z-score over 64 units, T={T})', f'{z:.2f}', z < 5))
        summary.append((f'{name} p={p}: eval is identity', str(exact), exact))

# ------------------------------------------------------------------ 2. vs torch (convention p_keep = 1 - p_torch)
torch.manual_seed(0)
x = torch.randn(1000, dtype=torch.float64)
p_keep = 0.7
y_t = F.dropout(x, p=1 - p_keep, training=True)
mask = (y_t != 0).numpy().astype(np.float64)
y_ours = x.numpy() * mask / p_keep                  # implementation.Dropout's formula with torch's mask
d_t = np.abs(y_ours - y_t.numpy()).max()
summary.append(('x*mask/p_keep vs F.dropout(p=1-p_keep) with the same mask', f'{d_t:.1e}', d_t < 1e-12))

# ------------------------------------------------------------------ 3. backward vs finite differences
d = impl.Dropout(p=0.6); d.training = True
x = rng.standard_normal((3, 5))
y = d.forward(x)
g = rng.standard_normal(y.shape)
analytic = d.backward(g)
num = np.zeros_like(x); eps = 1e-6
for idx in np.ndindex(x.shape):                       # perturb one entry at a time, mask held fixed
    xp = x.copy(); xp[idx] += eps
    xm = x.copy(); xm[idx] -= eps
    num[idx] = ((xp * d.mask / d.p * g).sum() - (xm * d.mask / d.p * g).sum()) / (2 * eps)
d_fd = np.abs(analytic - num).max()
summary.append(('implementation.Dropout.backward vs central differences', f'{d_fd:.1e}', d_fd < 1e-8))

# ------------------------------------------------------------------ 4. DropoutMLP backprop vs autograd
np.random.seed(1)
mlp = tm.DropoutMLP(input_size=20, hidden_sizes=[16, 12], output_size=5, dropout_p=0.5, input_dropout_p=0.8)
mlp.train()
Xb = np.random.randn(7, 20)
yb = np.random.randint(0, 5, 7)
probs = mlp.forward(Xb)
loss, grad = tm.cross_entropy_loss(probs, yb)
mlp.backward(grad)
lins = [l for l in mlp.layers if isinstance(l, tm.Linear)]
Ws = [torch.tensor(l.W, requires_grad=True) for l in lins]
bs = [torch.tensor(l.b, requires_grad=True) for l in lins]
h = torch.tensor(Xb) * torch.tensor(mlp.input_dropout.mask)       # same input mask (values 0 or 1/p)
for i in range(2):
    h = torch.relu(h @ Ws[i].T + bs[i]) * torch.tensor(mlp.dropouts[i].mask)
logits = h @ Ws[2].T + bs[2]
loss_t = F.cross_entropy(logits, torch.tensor(yb))
loss_t.backward()
d_mlp = max(max(np.abs(l.dW - W.grad.numpy()).max(), np.abs(l.db - b.grad.numpy()).max())
            for l, W, b in zip(lins, Ws, bs))
summary.append(('train_minimal.DropoutMLP: loss vs torch', f'{abs(loss - loss_t.item()):.1e}', abs(loss - loss_t.item()) < 1e-9))
summary.append(('train_minimal.DropoutMLP: dW, db vs autograd (same masks)', f'{d_mlp:.1e}', d_mlp < 1e-9))

# ------------------------------------------------------------------ 5. MC dropout on a linear read-out
d = impl.Dropout(p=0.5); d.training = True
x = rng.standard_normal(50); w = rng.standard_normal(50)
T = 100000
outs = d.forward(np.tile(x, (T, 1))) @ w             # (T, 50) @ (50,) -> (T,) one read-out per MC sample
mu_cf, var_cf = w @ x, (1 - d.p) / d.p * np.sum(w ** 2 * x ** 2)
rel_mu = abs(outs.mean() - mu_cf) / np.sqrt(var_cf / T)   # in standard errors
rel_var = abs(outs.var() - var_cf) / var_cf
summary.append(('MC dropout mean vs w.x (in std errors)', f'{rel_mu:.2f}', rel_mu < 5))
summary.append(('MC dropout variance vs (1-p)/p*sum w^2 x^2 (relative error)', f'{rel_var:.3%}', rel_var < 0.02))

# ------------------------------------------------------------------ 6. Sec. 9.1 marginalized dropout = data-dependent ridge
N, D, p = 40, 6, 0.8
X = rng.standard_normal((N, D)); yv = rng.standard_normal(N); w = rng.standard_normal(D)
T, chunk, acc = 100000, 10000, 0.0
for _ in range(T // chunk):
    R = (rng.random((chunk, N, D)) < p).astype(np.float64)   # (chunk, N, D) Bernoulli(p) keep-masks
    pred = np.einsum('tnd,nd,d->tn', R, X, w)                 # einsum: sum over d of R*X*w -> (chunk, N)
    acc += np.sum((yv[None, :] - pred) ** 2)                  # yv[None,:] broadcasts (1,N) against (chunk,N)
mc = acc / T
Gamma = np.sqrt(np.diag(X.T @ X))                     # diag(X^T X)^(1/2) as a vector (D,)
cf = np.sum((yv - p * X @ w) ** 2) + p * (1 - p) * np.sum((Gamma * w) ** 2)
rel = abs(mc - cf) / cf
summary.append(('Sec 9.1: MC E||y-(R*X)w||^2 vs ||y-pXw||^2 + p(1-p)||Gw||^2 (relative error)', f'{rel:.3%}', rel < 0.005))

# ------------------------------------------------------------------ 7. exercise 1 tests vs solution
ex1.Dropout = sol1.Dropout
ok7 = []
for t in ['test_forward_training', 'test_forward_eval', 'test_expected_value', 'test_backward',
          'test_backward_eval', 'test_numerical_gradient']:
    try:
        getattr(ex1, t)(); ok7.append(True)
    except Exception as e:  # noqa: BLE001
        print('  exercise test failed:', t, e); ok7.append(False)
summary.append(('exercise_01 tests with solution_01.Dropout', f'{sum(ok7)}/{len(ok7)} pass', all(ok7)))

# ------------------------------------------------------------------ 8. AlphaDropout self-normalisation
x = rng.standard_normal(2_000_000)
for p_keep in (0.5, 0.8):
    ad = impl.AlphaDropout(p=p_keep); ad.training = True
    yo = ad.forward(x)
    yt = F.alpha_dropout(torch.from_numpy(x), p=1 - p_keep, training=True).numpy()
    ok = abs(yo.mean()) < 0.01 and abs(yo.var() - 1) < 0.01
    summary.append((f'AlphaDropout p_keep={p_keep}: impl mean/var = {yo.mean():+.3f}/{yo.var():.3f}; '
                    f'torch = {yt.mean():+.3f}/{yt.var():.3f} (target 0/1)', 'ok' if ok else 'WRONG', ok))

# ------------------------------------------------------------------ 9. DropConnect backward vs autograd
np.random.seed(3)
dc = impl.DropConnect(6, 4, p=0.5); dc.training = True
xin = np.random.randn(5, 6)
out = dc.forward(xin)
gout = np.random.randn(*out.shape)
gin, gW, gb = dc.backward(gout)
Wt = torch.tensor(dc.W, requires_grad=True)
xt = torch.tensor(xin, requires_grad=True)
yt = xt @ (Wt * torch.tensor(dc.mask) / dc.p).T + torch.tensor(dc.b)
yt.backward(torch.tensor(gout))
dW = np.abs(gW - Wt.grad.numpy()).max(); dX = np.abs(gin - xt.grad.numpy()).max()
summary.append(('DropConnect.backward grad_input vs autograd', f'{dX:.1e}', dX < 1e-10))
summary.append(('DropConnect.backward grad_W vs autograd', f'{dW:.1e}', dW < 1e-10))

print('\nSUMMARY')
for name, val, ok in summary:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")
print('ALL PASS' if all(ok for *_, ok in summary) else 'SOME CHECKS FAILED')
