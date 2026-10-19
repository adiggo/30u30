"""Day 28 (CS231n): checks of the analytic backward pass of the NumPy CNN and of the backprop trainer.

Run from anywhere:
    OMP_NUM_THREADS=1 python validation/day28_backprop_check.py [path/to/papers/28_CS231n]
Uses CIFAR-10 images from <day_dir>/data/cifar-10-batches-py when present (random images otherwise);
check 8 needs CIFAR-10 and is skipped without it.

Relative error, as in the CS231n assignments: max over entries of |a - b| / max(1e-8, |a| + |b|).

 1. im2col (vectorized) == im2col_naive and col2im (np.add.at) == col2im_naive exactly; col2im is the adjoint of
    im2col (<im2col(x), C> = <x, col2im(C)>); the buffered fancy-index '+=' that np.add.at replaces loses gradient
    (shown on purpose; that is why col2im uses np.add.at).
 2. conv_backward_im2col and conv_backward_naive vs torch.autograd through F.conv2d (float64), 27 (stride, pad,
    kernel) configs incl. uncovered borders: dx, dw, db. Plus the notes' statement that the backward pass of a
    convolution is a convolution with spatially flipped filters (stride 1).
 3. pool_backward vs autograd through F.max_pool2d: random inputs, overlapping windows (3x3/s2, 3x3/s1), and
    integer-valued inputs full of ties; solution_02.pool_backward (exercise 2) agrees with it.
 4. relu_backward, fc_backward, softmax_cross_entropy_backward vs autograd.
 5. SimpleCNN.backward vs autograd of an equivalent torch.nn model (weights copied, float64): all 8 parameter
    gradients and the input gradient, at initialization and after a 50-step burn-in (CS231n: also check in a
    "characteristic" mode of operation, not only at the random init).
 6. Centred finite differences (CS231n neural-networks-3), using train_minimal.numerical_gradient: whole tensors
    b1, b2, b3, b4, W4 and 10 random coordinates of every other parameter and of the input, with kink detection
    (a coordinate whose +h/-h evaluations change a ReLU sign or a max-pool winner is reported, not scored).
 7. SimpleCNN.update vs torch.optim.SGD(momentum=0.9, weight_decay on the weights only), 20 steps, constant lr.
 8. End to end (CIFAR-10): train_minimal.train with momentum 0 on the old default setup (200 train / 50 test,
    batch 8, lr 0.01, 3 epochs, seed 0) reproduces plain SGD with autograd gradients (the trajectory in
    validation/day28_numgrad_and_trajectory.py), and the cost of one step vs a numerical-gradient step.
"""
import importlib.util
import itertools
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'papers', '28_CS231n')
sys.path.insert(0, DAY)
import implementation as impl  # noqa: E402
import train_minimal as tm  # noqa: E402

torch.set_default_dtype(torch.float64)
TOL = 1e-6          # analytic vs autograd / finite differences, float64 (the brief's bar)
rng = np.random.default_rng(0)
results = []        # (name, ok, detail)


def rel_error(a, b):
    """CS231n assignments' relative error: max |a - b| / max(1e-8, |a| + |b|) over all entries."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))) if a.size else 0.0


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sol2 = load(os.path.join(DAY, 'exercises/solutions/solution_02_pooling.py'), 'sol2')

cifar_dir = os.path.join(DAY, 'data', 'cifar-10-batches-py')
HAVE_CIFAR = os.path.exists(os.path.join(cifar_dir, 'data_batch_1'))
if HAVE_CIFAR:
    Xc, yc, _, _ = tm.load_cifar10(data_dir=cifar_dir, max_train=2000, max_test=1)
    Xc = Xc.astype(np.float64)
else:
    print('NOTE: CIFAR-10 not found -> checks 5-7 use random images, check 8 is skipped')
    Xc, yc = rng.standard_normal((2000, 3, 32, 32)), rng.integers(0, 10, 2000)

# ---------------------------------------------------------------- 1. im2col / col2im
worst_im2col = worst_col2im = worst_adj = 0.0
for (N, C, H, W), (fh, fw), stride, pad in [((2, 3, 9, 8), (3, 3), 1, 1), ((2, 3, 9, 8), (2, 3), 2, 0),
                                            ((1, 4, 7, 7), (5, 5), 3, 2), ((3, 2, 6, 5), (3, 2), 1, 0)]:
    x = rng.standard_normal((N, C, H, W))
    cols = impl.im2col(x, fh, fw, stride, pad)
    worst_im2col = max(worst_im2col, np.abs(cols - impl.im2col_naive(x, fh, fw, stride, pad)).max())
    dcols = rng.standard_normal(cols.shape)
    back = impl.col2im(dcols, x.shape, fh, fw, stride, pad)
    worst_col2im = max(worst_col2im, np.abs(back - impl.col2im_naive(dcols, x.shape, fh, fw, stride, pad)).max())
    lhs, rhs = np.sum(cols * dcols), np.sum(x * back)           # <im2col(x), C> and <x, col2im(C)>
    worst_adj = max(worst_adj, abs(lhs - rhs) / abs(lhs))
record('1a im2col == im2col_naive, col2im == col2im_naive', worst_im2col == 0 and worst_col2im < 1e-12,
       f'max |diff| = {worst_im2col:.1e} / {worst_col2im:.1e}')
record('1b col2im is the adjoint of im2col: <im2col(x), C> = <x, col2im(C)>', worst_adj < 1e-12,
       f'max relative difference {worst_adj:.1e}')
# Why np.add.at: the buffered x[idx] += v keeps one contribution per repeated index (expected to be wrong).
x_shape, fh, fw, stride, pad = (2, 3, 9, 8), 3, 3, 1, 1
dcols = rng.standard_normal((3 * 9, 2 * 9 * 8))
c_idx, h_idx, w_idx = impl.im2col_indices(x_shape, fh, fw, stride, pad)
xp = np.zeros((2, 3, 11, 10))
xp[:, c_idx, h_idx, w_idx] += dcols.reshape(27, 2, 72).transpose(1, 0, 2)
buffered_err = np.abs(xp[:, :, 1:10, 1:9] - impl.col2im(dcols, x_shape, fh, fw, stride, pad)).max()
print(f'     (info) col2im with a plain buffered "+=" instead of np.add.at: max |error| = {buffered_err:.2f} '
      f'-> overlapping windows lose gradient, as the col2im comment says')

# ---------------------------------------------------------------- 2. conv backward vs autograd
worst = {'im2col': 0.0, 'naive': 0.0}
configs = list(itertools.product([1, 2, 3], [0, 1, 2], [(3, 3), (5, 5), (2, 3)]))
for stride, pad, (fh, fw) in configs:
    x = rng.standard_normal((2, 3, 9, 8))
    w = rng.standard_normal((4, 3, fh, fw))
    b = rng.standard_normal(4)
    xt, wt, bt = (torch.tensor(a, requires_grad=True) for a in (x, w, b))
    out = F.conv2d(xt, wt, bt, stride=stride, padding=pad)
    dout = rng.standard_normal(tuple(out.shape))
    out.backward(torch.from_numpy(dout))
    ref = (xt.grad.numpy(), wt.grad.numpy(), bt.grad.numpy())
    for name, fn in (('im2col', impl.conv_backward_im2col), ('naive', impl.conv_backward_naive)):
        got = fn(dout, x, w, stride, pad)
        worst[name] = max(worst[name], *(rel_error(g, r) for g, r in zip(got, ref)))
for name in ('im2col', 'naive'):
    record(f'2 conv_backward_{name} (dx, dw, db) vs autograd(F.conv2d), {len(configs)} configs',
           worst[name] < TOL, f'max relative error {worst[name]:.1e}')
# CS231n: "The backward pass for a convolution operation (for both the data and the weights) is also a
# convolution (but with spatially-flipped filters)". For stride 1: dx = conv(dout, w flipped in space with
# the in/out channels swapped, padding F - 1 - pad).
worst_flip = 0.0
for f, pad in [(3, 0), (3, 1), (3, 2), (5, 2)]:
    x, w = rng.standard_normal((2, 3, 8, 8)), rng.standard_normal((4, 3, f, f))
    dout = rng.standard_normal((2, 4, 8 - f + 2 * pad + 1, 8 - f + 2 * pad + 1))
    dx, _, _ = impl.conv_backward_im2col(dout, x, w, 1, pad)
    # [:, :, ::-1, ::-1] reverses rows and columns of every filter; transpose(1, 0, 2, 3) swaps K and C
    w_flip = w[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)
    dx_conv = impl.conv_forward_im2col(dout, w_flip, np.zeros(3), stride=1, pad=f - 1 - pad)
    worst_flip = max(worst_flip, rel_error(dx, dx_conv))
record('2b stride-1 dx == convolution of dout with flipped filters (CS231n "Backpropagation")',
       worst_flip < TOL, f'max relative error {worst_flip:.1e}')

# ---------------------------------------------------------------- 3. max-pool backward vs autograd
worst_pool = worst_sol = 0.0
n_tied = 0
for k, s in [(2, 2), (3, 2), (3, 1), (2, 1)]:          # (3,2), (3,1), (2,1) have overlapping windows
    for kind in ('random', 'ties'):
        x = rng.standard_normal((2, 3, 9, 9)) if kind == 'random' else rng.integers(0, 3, (2, 3, 9, 9)).astype(float)
        xt = torch.tensor(x, requires_grad=True)
        yt = F.max_pool2d(xt, kernel_size=k, stride=s)
        dout = rng.standard_normal(tuple(yt.shape))
        yt.backward(torch.from_numpy(dout))
        dx = impl.pool_backward(dout, x, k, s)
        worst_pool = max(worst_pool, np.abs(dx - xt.grad.numpy()).max())
        worst_sol = max(worst_sol, np.abs(dx - sol2.pool_backward(dout, (x, k, s))).max())
        if kind == 'ties':   # count windows whose max is shared (a "==max" mask would split or copy gradient there)
            win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s]
            n_tied += int(np.sum((win == win.max(axis=(-2, -1), keepdims=True)).sum(axis=(-2, -1)) > 1))
record('3 pool_backward vs autograd(F.max_pool2d), incl. overlapping windows and ties', worst_pool == 0,
       f'max |diff| = {worst_pool:.1e} ({n_tied} windows with a tied maximum: first max wins in both)')
record('3b exercise 2 solution pool_backward == implementation.pool_backward', worst_sol == 0,
       f'max |diff| = {worst_sol:.1e}')

# ---------------------------------------------------------------- 4. relu, fc, softmax cross-entropy
x = rng.standard_normal((6, 7))
x[0, :3] = 0.0                                          # exact zeros: the derivative there is taken as 0, as in torch
xt = torch.tensor(x, requires_grad=True)
dout = rng.standard_normal(x.shape)
F.relu(xt).backward(torch.from_numpy(dout))
e_relu = rel_error(impl.relu_backward(dout, x), xt.grad.numpy())
x, w, b = rng.standard_normal((6, 5)), rng.standard_normal((5, 4)), rng.standard_normal(4)
xt, wt, bt = (torch.tensor(a, requires_grad=True) for a in (x, w, b))
dout = rng.standard_normal((6, 4))
(xt @ wt + bt).backward(torch.from_numpy(dout))
e_fc = max(rel_error(g, r.grad.numpy()) for g, r in zip(impl.fc_backward(dout, x, w), (xt, wt, bt)))
s, y = rng.standard_normal((6, 10)) * 3, rng.integers(0, 10, 6)
st = torch.tensor(s, requires_grad=True)
F.cross_entropy(st, torch.tensor(y)).backward()
probs = impl.softmax(s)
e_ce = rel_error(impl.softmax_cross_entropy_backward(probs, y), st.grad.numpy())
e_loss = abs(impl.cross_entropy_loss(probs, y) - F.cross_entropy(torch.tensor(s), torch.tensor(y)).item())
record('4 relu_backward / fc_backward / softmax_cross_entropy_backward vs autograd',
       max(e_relu, e_fc, e_ce) < TOL and e_loss < 1e-12,
       f'max relative error {e_relu:.1e} / {e_fc:.1e} / {e_ce:.1e}; |loss - F.cross_entropy| = {e_loss:.1e}')


# ---------------------------------------------------------------- 5. SimpleCNN.backward vs an equivalent torch model
class TorchSimpleCNN(nn.Module):
    """The same network as implementation.SimpleCNN, in torch.nn."""

    def __init__(self):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 16, 3, padding=1)
        self.fc1, self.fc2 = nn.Linear(16 * 8 * 8, 64), nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))           # flatten(1): (N, 16, 8, 8) -> (N, 1024) in C order, like reshape
        return self.fc2(x)

    def load_numpy(self, p):
        with torch.no_grad():
            self.conv1.weight.copy_(torch.from_numpy(p['W1'])); self.conv1.bias.copy_(torch.from_numpy(p['b1']))
            self.conv2.weight.copy_(torch.from_numpy(p['W2'])); self.conv2.bias.copy_(torch.from_numpy(p['b2']))
            # nn.Linear stores weight as (out, in) and computes x @ weight.T, so weight = W.T
            self.fc1.weight.copy_(torch.from_numpy(p['W3'].T)); self.fc1.bias.copy_(torch.from_numpy(p['b3']))
            self.fc2.weight.copy_(torch.from_numpy(p['W4'].T)); self.fc2.bias.copy_(torch.from_numpy(p['b4']))

    def grads_as_numpy(self):
        return {'W1': self.conv1.weight.grad, 'b1': self.conv1.bias.grad, 'W2': self.conv2.weight.grad,
                'b2': self.conv2.bias.grad, 'W3': self.fc1.weight.grad.T, 'b3': self.fc1.bias.grad,
                'W4': self.fc2.weight.grad.T, 'b4': self.fc2.bias.grad}


def compare_with_torch(model, xb, yb, label):
    probs = impl.softmax(model.forward(xb))
    loss = impl.cross_entropy_loss(probs, yb)
    grads, dx = model.backward(impl.softmax_cross_entropy_backward(probs, yb), return_dx=True)
    tmodel = TorchSimpleCNN()
    tmodel.load_numpy(model.params)
    xt = torch.tensor(xb, requires_grad=True)
    tloss = F.cross_entropy(tmodel(xt), torch.tensor(yb))
    tloss.backward()
    tgrads = {k: v.numpy() for k, v in tmodel.grads_as_numpy().items()}
    errs = {k: rel_error(grads[k], tgrads[k]) for k in grads}
    errs['x'] = rel_error(dx, xt.grad.numpy())
    worst_name = max(errs, key=errs.get)
    record(f'5 SimpleCNN.backward vs torch autograd, {label}', max(errs.values()) < TOL,
           f'loss {loss:.6f} vs {tloss.item():.6f}; max relative error per tensor: '
           + ', '.join(f'{k} {v:.1e}' for k, v in errs.items()) + f' (worst: {worst_name})')
    return errs


np.random.seed(0)
model = impl.SimpleCNN(num_classes=10)
errs_init = compare_with_torch(model, Xc[:16], yc[:16], 'at init (16 images)')
for step in range(50):                                   # burn-in: 50 SGD+momentum steps on batches of 32
    idx = rng.choice(len(Xc) - 16, 32, replace=False) + 16
    probs = impl.softmax(model.forward(Xc[idx]))
    model.update(model.backward(impl.softmax_cross_entropy_backward(probs, yc[idx])), lr=0.01, momentum=0.9,
                 weight_decay=5e-4)
errs_burn = compare_with_torch(model, Xc[:16], yc[:16], 'after a 50-step burn-in (16 images)')

# ---------------------------------------------------------------- 6. centred finite differences
xb, yb = Xc[:4], yc[:4]                                  # "use only few datapoints" (fewer kinks)


def kink_signature(m):
    """ReLU signs and max-pool winners of the last forward pass (CS231n: track the max 'winners')."""
    c = m.cache
    sig = [c['a1'] > 0, c['a2'] > 0, c['a3'] > 0]
    for r in (c['r1'], c['r2']):
        N, C, H, W = r.shape
        # (N, C, H, W) -> (N, C, H/2, 2, W/2, 2) -> windows last -> (N, C, H/2, W/2, 4); argmax = winner per window
        win = r.reshape(N, C, H // 2, 2, W // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H // 2, W // 2, 4)
        sig.append(np.argmax(win, axis=-1))
    return sig


def crosses_kink(m, x, before):
    m.forward(x)
    return any(not np.array_equal(a, b) for a, b in zip(kink_signature(m), before))


h = 1e-5
model_fd = impl.SimpleCNN(num_classes=10)
model_fd.params = {k: v.copy() for k, v in model.params.items()}   # the burned-in weights
probs = impl.softmax(model_fd.forward(xb))
base_sig = kink_signature(model_fd)
grads, dx = model_fd.backward(impl.softmax_cross_entropy_backward(probs, yb), return_dx=True)
fd_worst, n_checked, kinks = {}, 0, []
for name, p in model_fd.params.items():
    if name in ('b1', 'b2', 'b3', 'b4', 'W4'):
        idx = list(np.ndindex(p.shape))                  # the whole tensor
    else:
        idx = [tuple(int(rng.integers(n)) for n in p.shape) for _ in range(10)]
    num = tm.numerical_gradient(model_fd, xb, yb, name, h=h, indices=idx)
    errs = []
    for i in idx:
        old = p[i]
        p[i] = old + h
        k_plus = crosses_kink(model_fd, xb, base_sig)
        p[i] = old - h
        k_minus = crosses_kink(model_fd, xb, base_sig)
        p[i] = old
        e = rel_error(grads[name][i], num[i])
        if k_plus or k_minus:
            kinks.append((name, i, e))
        else:
            errs.append(e)
    n_checked += len(errs)
    fd_worst[name] = max(errs) if errs else 0.0
errs = []                                                # input coordinates (inline: numerical_gradient covers params)
for _ in range(10):
    i = tuple(int(rng.integers(n)) for n in xb.shape)
    xp, xm = xb.copy(), xb.copy()
    xp[i] += h
    xm[i] -= h
    lp = impl.cross_entropy_loss(impl.softmax(model_fd.forward(xp)), yb)
    k_plus = any(not np.array_equal(a, b) for a, b in zip(kink_signature(model_fd), base_sig))
    lm = impl.cross_entropy_loss(impl.softmax(model_fd.forward(xm)), yb)
    k_minus = any(not np.array_equal(a, b) for a, b in zip(kink_signature(model_fd), base_sig))
    e = rel_error(dx[i], (lp - lm) / (2 * h))
    (kinks.append(('x', i, e)) if (k_plus or k_minus) else errs.append(e))
n_checked += len(errs)
fd_worst['x'] = max(errs) if errs else 0.0
record(f'6 backprop vs centred differences (h={h:g}, 4 images, burned-in weights, {n_checked} coordinates)',
       max(fd_worst.values()) < TOL,
       'max relative error per tensor: ' + ', '.join(f'{k} {v:.1e}' for k, v in fd_worst.items())
       + f'; {len(kinks)} coordinate(s) crossed a kink and were not scored'
       + (f' (their errors: {", ".join(f"{e:.0e}" for _, _, e in kinks)})' if kinks else ''))

# ---------------------------------------------------------------- 7. update rule vs torch.optim.SGD
np.random.seed(1)
m_np = impl.SimpleCNN(num_classes=10)
tparams = {k: torch.tensor(v.copy(), requires_grad=True) for k, v in m_np.params.items()}
opt = torch.optim.SGD([{'params': [tparams[k] for k in tparams if k.startswith('W')], 'weight_decay': 5e-4},
                       {'params': [tparams[k] for k in tparams if k.startswith('b')], 'weight_decay': 0.0}],
                      lr=0.01, momentum=0.9)
for step in range(20):
    g = {k: rng.standard_normal(v.shape) for k, v in m_np.params.items()}
    m_np.update(g, lr=0.01, momentum=0.9, weight_decay=5e-4)
    for k, t in tparams.items():
        t.grad = torch.from_numpy(g[k].copy())
    opt.step()
e_upd = max(rel_error(m_np.params[k], tparams[k].detach().numpy()) for k in tparams)
record('7 SimpleCNN.update (CS231n momentum + L2 on weights) vs torch.optim.SGD, 20 steps', e_upd < 1e-10,
       f'max relative parameter difference {e_upd:.1e}')

# ---------------------------------------------------------------- 8. end to end: the trainer = SGD with exact gradients
if HAVE_CIFAR:
    Xtr, ytr, Xte, yte = tm.load_cifar10(data_dir=cifar_dir, max_train=200, max_test=50)
    np.random.seed(0)                                     # train_minimal.main(): seed, then SimpleCNN, then train()
    model = impl.SimpleCNN(num_classes=10)
    t0 = time.time()
    hist = tm.train(model, Xtr, ytr, Xte, yte, epochs=3, lr=0.01, batch_size=8, momentum=0.0, weight_decay=0.0,
                    lr_decay=1.0, verbose=False)
    t_train = time.time() - t0
    # Reference: plain SGD on autograd gradients, same init, data, batch order (as in day28_numgrad_and_trajectory.py)
    np.random.seed(0)
    ref = impl.SimpleCNN(num_classes=10)
    tmodel = TorchSimpleCNN()
    tmodel.load_numpy(ref.params)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float64), torch.tensor(ytr)
    ref_rows = []
    for epoch in range(3):
        perm = np.random.permutation(len(Xtr))
        tot, correct = 0.0, 0
        for bi in range(len(Xtr) // 8):
            idx = torch.tensor(perm[bi * 8:(bi + 1) * 8])
            out = tmodel(Xt[idx])
            loss = F.cross_entropy(out, yt[idx])
            tmodel.zero_grad()
            loss.backward()
            with torch.no_grad():
                for prm in tmodel.parameters():
                    prm -= 0.01 * prm.grad
            tot += loss.item()
            correct += (out.argmax(1) == yt[idx]).sum().item()
        with torch.no_grad():
            test_acc = (tmodel(torch.tensor(Xte, dtype=torch.float64)).argmax(1).numpy() == yte).mean()
        ref_rows.append((tot / (len(Xtr) // 8), correct / len(Xtr), test_acc))
    rows = [(r['batch_loss'], r['batch_acc'], r['test_acc']) for r in hist[1:]]
    d_loss = max(abs(a[0] - b[0]) for a, b in zip(rows, ref_rows))
    same_acc = all(a[1] == b[1] and a[2] == b[2] for a, b in zip(rows, ref_rows))
    tgrads = {'W1': tmodel.conv1.weight, 'W3': tmodel.fc1.weight}
    d_par = max(rel_error(model.params['W1'], tgrads['W1'].detach().numpy()),
                rel_error(model.params['W3'], tgrads['W3'].detach().numpy().T))
    record('8 train_minimal.train (momentum 0) == SGD on autograd gradients, 200/50 images, batch 8, 3 epochs',
           d_loss < 1e-8 and same_acc and d_par < 1e-8,
           'epoch loss / batch acc / test acc: '
           + ' -> '.join(f'{lo:.4f} / {ba:.1%} / {ta:.0%}' for lo, ba, ta in rows)
           + f'; max |loss diff| vs autograd {d_loss:.1e}, final weights rel. diff {d_par:.1e}')
    # Cost of one step: backprop vs the numerical gradient the old trainer used (process CPU time, so that other
    # jobs on a shared machine do not inflate it)
    np.random.seed(0)
    m = impl.SimpleCNN(num_classes=10)
    P = m.count_parameters()
    for bs in (8, 64):
        xb_, yb_ = Xc[:bs], yc[:bs]
        t0 = time.process_time()
        for _ in range(10):
            probs = impl.softmax(m.forward(xb_))
            m.update(m.backward(impl.softmax_cross_entropy_backward(probs, yb_)), lr=0.0)   # lr 0: weights unchanged
        t_step = (time.process_time() - t0) / 10
        t0 = time.process_time()
        for _ in range(10):
            m.forward(xb_)
        t_fwd = (time.process_time() - t0) / 10
        print(f'     (info) batch {bs}: backprop step {t_step * 1e3:.0f} ms CPU (forward alone {t_fwd * 1e3:.0f} ms); '
              f'a numerical-gradient step needs 2 x {P:,} + 1 forwards = {(2 * P + 1) * t_fwd / 60:.0f} min '
              f'with this forward, ~{(2 * P + 1) * t_fwd / t_step:.1g}x the backprop step')
    print(f'     (info) the 3-epoch run of check 8 took {t_train:.1f} s (wall)')
else:
    print('[SKIP] 8 end-to-end trajectory check (needs CIFAR-10)')

print('\nSUMMARY')
for name, ok, detail in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print('ALL PASS' if all(ok for _, ok, _ in results) else 'SOME CHECKS FAILED')
