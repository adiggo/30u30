"""Day 28: is train_minimal.py's numerical gradient correct, how slow is training with it, and what
trajectory does the old default setup give with exact gradients?

[PATCH 2026-10-18] train_minimal.py trained with this numerical gradient until 2026-10-18; it now trains with
backprop (implementation.SimpleCNN.backward) and keeps numerical_gradient as a gradient-checking tool. Items 2
and 3 describe the old trainer. The forward pass now uses a vectorized im2col, so item 2 prints less than the
old loop version needed (~15 min per step at batch 1); it times wall-clock, so a busy machine inflates it.
The W1 error in item 1 (~1e-5) is a kink crossing at h = 1e-5 (a ReLU or max-pool winner flips between
f(w+h) and f(w-h); CS231n neural-networks-3, "Kinks in the objective"): with h = 1e-6 it drops to ~2e-9.
The trajectory in 3 is what `train_minimal.py --max-train 200 --max-test 50 --batch-size 8 --momentum 0
--weight-decay 0 --lr-decay 1 --epochs 3` now prints (validation/day28_backprop_check.py, check 8, compares them).

    OMP_NUM_THREADS=1 python validation/day28_numgrad_and_trajectory.py [day_dir]
(needs CIFAR-10 in <day_dir>/data/cifar-10-batches-py)

 1. train_minimal.numerical_gradient (central differences, h=1e-5) vs torch autograd through the same
    SimpleCNN forward (F.conv2d pad 1 -> ReLU -> F.max_pool2d 2 -> ... -> FC), same weights and batch,
    for W1, b1, b2, b3, W4, b4 (954 of the 67,642 parameters).
 2. Cost: forward time vs batch size and the implied time per SGD step (2 * 67,642 + 1 forwards).
 3. Trajectory: the identical model (same NumPy init), data (load_cifar10, 200 train / 50 test), batch
    size 8, plain SGD lr 0.01, 3 epochs, same per-epoch permutations -- trained with exact gradients.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'papers', '28_CS231n')
sys.path.insert(0, DAY)
import implementation as impl  # noqa: E402
import train_minimal as tm  # noqa: E402

torch.set_default_dtype(torch.float64)


def torch_forward(params, x):
    """SimpleCNN.forward re-implemented with PyTorch ops (params: dict of tensors)."""
    h = F.max_pool2d(F.relu(F.conv2d(x, params['W1'], params['b1'], padding=1)), 2)
    h = F.max_pool2d(F.relu(F.conv2d(h, params['W2'], params['b2'], padding=1)), 2)
    h = F.relu(h.reshape(x.shape[0], -1) @ params['W3'] + params['b3'])
    return h @ params['W4'] + params['b4']


Xtr, ytr, Xte, yte = tm.load_cifar10(data_dir=os.path.join(DAY, 'data', 'cifar-10-batches-py'), max_train=200, max_test=50)

# ------------------------------------------------------------------ 1. numerical vs autograd gradient
np.random.seed(0)
model = impl.SimpleCNN(num_classes=10)
xb, yb = Xtr[:8], ytr[:8]
tp = {k: torch.tensor(v, requires_grad=True) for k, v in model.params.items()}
loss = F.cross_entropy(torch_forward(tp, torch.tensor(xb, dtype=torch.float64)), torch.tensor(yb))
loss.backward()
np_loss = impl.cross_entropy_loss(impl.softmax(model.forward(xb)), yb)
worst_rel = 0.0
for name in ('W1', 'b1', 'b2', 'b3', 'W4', 'b4'):
    g_num = tm.numerical_gradient(model, xb, yb, name)
    g_ad = tp[name].grad.numpy()
    rel = np.abs(g_num - g_ad).max() / (np.abs(g_ad).max() + 1e-12)
    worst_rel = max(worst_rel, rel)
    print(f'  {name:3s} max|num - autograd| / max|autograd| = {rel:.1e}')
print(f'[1] loss NumPy {np_loss:.10f} vs torch {loss.item():.10f}; worst relative gradient error {worst_rel:.1e} '
      f'-> {"PASS" if worst_rel < 1e-4 else "FAIL"}')

# ------------------------------------------------------------------ 2. cost per SGD step
P = model.count_parameters()
for n in (1, 2, 8):
    t0 = time.time()
    for _ in range(5):
        model.forward(Xtr[:n])
    t = (time.time() - t0) / 5
    print(f'[2] batch {n}: forward {t * 1e3:.1f} ms -> one numerical-gradient SGD step = {(2 * P + 1) * t / 60:.1f} min')

# ------------------------------------------------------------------ 3. exact-gradient trajectory
np.random.seed(0)
model = impl.SimpleCNN(num_classes=10)                    # same NumPy init order as train_minimal.main()
tp = {k: torch.tensor(v, requires_grad=True) for k, v in model.params.items()}
Xt, Xv = torch.tensor(Xtr, dtype=torch.float64), torch.tensor(Xte, dtype=torch.float64)
yt = torch.tensor(ytr)
lr, bs = 0.01, 8
n_batches = len(Xtr) // bs
for epoch in range(3):
    perm = np.random.permutation(len(Xtr))                # same call as train_minimal.train()
    tot, correct = 0.0, 0
    for b in range(n_batches):
        idx = torch.tensor(perm[b * bs:(b + 1) * bs])
        out = torch_forward(tp, Xt[idx])
        l = F.cross_entropy(out, yt[idx])
        for v in tp.values():
            v.grad = None
        l.backward()
        with torch.no_grad():
            for v in tp.values():
                v -= lr * v.grad
        tot += l.item(); correct += (out.argmax(1) == yt[idx]).sum().item()
    with torch.no_grad():
        test_acc = (torch_forward(tp, Xv[:100]).argmax(1).numpy() == yte[:100]).mean()
    print(f'[3] epoch {epoch + 1}: avg loss {tot / n_batches:.4f}, train acc {correct / len(Xtr):.2%}, test acc {test_acc:.2%}')
