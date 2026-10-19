"""Day 12: implementation.DropoutNetwork.backward() and the training in compare_dropout_rates().

1. backward() vs torch.autograd (float64) on an independent re-implementation of the same network,
   with the dropout masks that forward() drew; relative error per parameter, tolerance 1e-6.
2. backward() vs central finite differences (the same masks are redrawn by re-seeding before every
   forward pass); tolerance 1e-6 relative.
3. One sgd_step() on a batch lowers that batch's loss (same masks).
4. compare_dropout_rates() on real MNIST (data/mnist.npz of the day folder, or a day folder given as
   argv[1]): a training subset of 6,000 images, validation on 2,000 test images, 5 epochs per rate.
   Informational: train/val accuracy and gap per keep probability (no dropout vs keep 0.7/0.5/0.3).
   Measured (9 s): gap 0.072 / 0.059 / 0.051 / 0.059 and val 0.917 / 0.911 / 0.905 / 0.857 for keep
   1.0 / 0.7 / 0.5 / 0.3: dropout shrinks the gap, but 5 epochs are too few for it to raise val
   accuracy (with epochs=20, keep 0.8 reaches val 0.942 vs 0.934 without dropout).

Run:  OMP_NUM_THREADS=1 python validation/day12_dropout_network.py [day_dir]
"""
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'papers', '12_dropout')
sys.path.insert(0, DAY)
import implementation as I  # noqa: E402

torch.set_default_dtype(torch.float64)
rows = []

# ---- 1-2. gradients -------------------------------------------------------------------------
np.random.seed(0)
N, D, H1, H2, C, KEEP = 6, 7, 5, 4, 3, 0.6
X = np.random.randn(N, D)
y = np.random.randint(0, C, N)
net = I.DropoutNetwork(D, [H1, H2], C, dropout_p=KEEP)
for layer in net.layers:                         # float64 weights with non-zero biases
    layer.W = np.random.randn(*layer.W.shape) * 0.7
    layer.b = np.random.randn(*layer.b.shape) * 0.3
net.train()


def loss_with_fixed_masks():
    np.random.seed(123)                          # the same dropout masks on every call
    probs = net.forward(X)
    return -np.mean(np.log(probs[np.arange(N), y])), probs


loss0, probs = loss_with_fixed_masks()
grads = net.backward(probs, y)
masks = [d.mask.astype(np.float64) for d in net.dropout_layers]

tW = [torch.tensor(layer.W, requires_grad=True) for layer in net.layers]
tb = [torch.tensor(layer.b, requires_grad=True) for layer in net.layers]
h = torch.tensor(X)
for i in range(len(tW)):
    h = h @ tW[i].T + tb[i]
    if i < len(tW) - 1:
        h = torch.relu(h) * torch.tensor(masks[i]) / KEEP     # inverted dropout with forward()'s mask
loss_t = torch.nn.functional.cross_entropy(h, torch.tensor(y))
loss_t.backward()


def rel(a, ref):
    return np.abs(a - ref).max() / max(np.abs(ref).max(), 1e-12)


worst_ag = max(max(rel(gW, tW[i].grad.numpy()), rel(gb, tb[i].grad.numpy()))
               for i, (gW, gb) in enumerate(grads))
rows.append(('loss vs torch cross_entropy', abs(loss0 - loss_t.item()), abs(loss0 - loss_t.item()) < 1e-12))
rows.append(('backward() vs torch.autograd (3 layers, W and b)', worst_ag, worst_ag < 1e-6))

eps = 1e-6
worst_fd = 0.0
for i, layer in enumerate(net.layers):
    for P, g in ((layer.W, grads[i][0]), (layer.b, grads[i][1])):
        num = np.zeros_like(P)
        for idx in np.ndindex(P.shape):          # every entry
            old = P[idx]
            P[idx] = old + eps
            lp = loss_with_fixed_masks()[0]
            P[idx] = old - eps
            lm = loss_with_fixed_masks()[0]
            P[idx] = old
            num[idx] = (lp - lm) / (2 * eps)
        worst_fd = max(worst_fd, rel(g, num))
rows.append(('backward() vs central differences', worst_fd, worst_fd < 1e-6))

# ---- 3. one SGD step lowers the batch loss ----------------------------------------------------
before = loss_with_fixed_masks()[0]
_, probs = loss_with_fixed_masks()
net.sgd_step(net.backward(probs, y), lr=0.05)
after = loss_with_fixed_masks()[0]
rows.append(('sgd_step(lr=0.05) lowers the batch loss (after - before)', after - before, after < before))

print('Day 12 DropoutNetwork backward (float64, N=%d, %d-%d-%d-%d, keep p=%.1f)' % (N, D, H1, H2, C, KEEP))
for name, err, ok in rows:
    print('  %-58s %.1e  %s' % (name, err, 'PASS' if ok else 'FAIL'))

# ---- 4. compare_dropout_rates on real MNIST ----------------------------------------------------
path = os.path.join(DAY, 'data', 'mnist.npz')
if not os.path.exists(path):
    print('\n[SKIP] compare_dropout_rates: %s not found' % path)
    sys.exit(0)
d = np.load(path)
Xtr, ytr, Xte, yte = d['X_train'][:6000], d['y_train'][:6000], d['X_test'][:2000], d['y_test'][:2000]
print('\ncompare_dropout_rates on MNIST: 6,000 training images, 2,000 test images, 5 epochs per rate')
t0 = time.time()
res = I.compare_dropout_rates(Xtr, ytr, Xte, yte, dropout_rates=[0.0, 0.3, 0.5, 0.7], epochs=5)
print('\n  keep p   train    val      gap')
for keep, r in sorted(res.items(), reverse=True):
    print('  %.1f      %.3f    %.3f    %.3f' % (keep, r['train_acc'], r['val_acc'], r['gap']))
print('  (%.0f s)' % (time.time() - t0))
