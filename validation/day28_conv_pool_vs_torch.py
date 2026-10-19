"""Day 28 (CS231n) reference checks.

Run from anywhere:
    OMP_NUM_THREADS=1 python validation/day28_conv_pool_vs_torch.py [path/to/papers/28_CS231n]

Checks
 1. implementation.conv_forward_naive  vs torch.nn.functional.conv2d (float64)
 2. implementation.conv_forward_im2col vs F.conv2d, plus the CS231n AlexNet im2col shape example
 3. implementation.pool_forward        vs F.max_pool2d
 4. solution_02.pool_backward          vs autograd through F.max_pool2d (incl. overlapping 3x3/s2 windows)
 5. solution_01.conv_forward           vs F.conv2d
 6. solution_04 VGG-16 parameter count vs torchvision.models.vgg16()
 7. exercise check() functions run against the solution implementations (exercise/solution agreement)
 8. cost estimate for train_minimal.py (numerical-gradient training)
"""
import importlib.util
import itertools
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'papers', '28_CS231n')
sys.path.insert(0, DAY)
import implementation as impl  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sol1 = load(os.path.join(DAY, 'exercises/solutions/solution_01_conv_forward.py'), 'sol1')
sol2 = load(os.path.join(DAY, 'exercises/solutions/solution_02_pooling.py'), 'sol2')
sol3 = load(os.path.join(DAY, 'exercises/solutions/solution_03_output_sizes.py'), 'sol3')
sol4 = load(os.path.join(DAY, 'exercises/solutions/solution_04_parameter_count.py'), 'sol4')
sol5 = load(os.path.join(DAY, 'exercises/solutions/solution_05_feature_viz.py'), 'sol5')

rng = np.random.default_rng(0)
results = {}

# ---------------------------------------------------------------- 1, 2, 5: convolution
worst_naive = worst_im2col = worst_sol1 = 0.0
configs = list(itertools.product([1, 2, 3], [0, 1, 2], [(3, 3), (5, 5), (2, 3)]))
for stride, pad, (fh, fw) in configs:
    x = rng.standard_normal((2, 3, 9, 8))          # (N, C, H, W)
    w = rng.standard_normal((4, 3, fh, fw))         # (K, C, FH, FW)
    b = rng.standard_normal(4)                       # (K,)
    ref = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b),
                   stride=stride, padding=pad).numpy()
    worst_naive = max(worst_naive, np.abs(impl.conv_forward_naive(x, w, b, stride, pad) - ref).max())
    worst_im2col = max(worst_im2col, np.abs(impl.conv_forward_im2col(x, w, b, stride, pad) - ref).max())
    worst_sol1 = max(worst_sol1, np.abs(sol1.conv_forward(x, w, b, stride, pad) - ref).max())
results['conv_forward_naive vs F.conv2d'] = worst_naive
results['conv_forward_im2col vs F.conv2d'] = worst_im2col
results['solution_01.conv_forward vs F.conv2d'] = worst_sol1
print(f"[1] conv_forward_naive  vs F.conv2d over {len(configs)} (stride,pad,kernel) configs: max|d| = {worst_naive:.2e}")
print(f"[2] conv_forward_im2col vs F.conv2d: max|d| = {worst_im2col:.2e}")
print(f"[5] solution_01.conv_forward vs F.conv2d: max|d| = {worst_sol1:.2e}")

# CS231n's im2col example: 227x227x3 input, 96 filters of 11x11x3, stride 4 -> X_col [363 x 3025], W_row [96 x 363]
xa = rng.standard_normal((1, 3, 227, 227)).astype(np.float32)
cols = impl.im2col(xa, 11, 11, stride=4, pad=0)
print(f"    im2col(227x227x3, 11x11, stride 4) -> X_col shape {cols.shape} (CS231n: [363 x 3025])")
assert cols.shape == (363, 3025)

# ---------------------------------------------------------------- 3: pooling forward
worst_pool = 0.0
for k, s in [(2, 2), (3, 2), (3, 1), (2, 1)]:
    x = rng.standard_normal((2, 3, 9, 9))
    ref = F.max_pool2d(torch.from_numpy(x), kernel_size=k, stride=s).numpy()
    worst_pool = max(worst_pool, np.abs(impl.pool_forward(x, k, s) - ref).max())
    worst_pool = max(worst_pool, np.abs(sol2.pool_forward(x, k, s)[0] - ref).max())
results['pool_forward (impl + solution_02) vs F.max_pool2d'] = worst_pool
print(f"[3] pool_forward (implementation + solution_02) vs F.max_pool2d: max|d| = {worst_pool:.2e}")

# ---------------------------------------------------------------- 4: pooling backward vs autograd
worst_pb = 0.0
for k, s in [(2, 2), (3, 2), (3, 1)]:              # (3,2) and (3,1) have overlapping windows
    x = rng.standard_normal((2, 3, 9, 9))
    xt = torch.from_numpy(x).requires_grad_(True)
    yt = F.max_pool2d(xt, kernel_size=k, stride=s)
    dout = rng.standard_normal(tuple(yt.shape))
    yt.backward(torch.from_numpy(dout))
    out, cache = sol2.pool_forward(x, k, s)
    dx = sol2.pool_backward(dout, cache)
    worst_pb = max(worst_pb, np.abs(dx - xt.grad.numpy()).max())
results['solution_02.pool_backward vs autograd(F.max_pool2d)'] = worst_pb
print(f"[4] solution_02.pool_backward vs autograd through F.max_pool2d (incl. overlapping windows): max|d| = {worst_pb:.2e}")

# ---------------------------------------------------------------- 6: parameter counts
import torchvision  # noqa: E402
vgg_tv = sum(p.numel() for p in torchvision.models.vgg16(weights=None).parameters())
vgg16 = []
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
d = 3
for c in cfg:
    if c == 'M':
        vgg16.append({'type': 'pool'})
    else:
        vgg16.append({'type': 'conv', 'in_depth': d, 'filters': c, 'filter_size': 3}); d = c
vgg16 += [{'type': 'fc', 'in_size': 7 * 7 * 512, 'out_size': 4096},
          {'type': 'fc', 'in_size': 4096, 'out_size': 4096},
          {'type': 'fc', 'in_size': 4096, 'out_size': 1000}]
_, vgg_sol = sol4.count_parameters(vgg16)
fc1 = (7 * 7 * 512 + 1) * 4096
print(f"[6] VGG-16 params: solution_04 = {vgg_sol:,}; torchvision.vgg16 = {vgg_tv:,}; "
      f"first FC = {fc1:,} ({100 * fc1 / vgg_tv:.1f}% of total)")
results['VGG-16 param count |solution - torchvision|'] = abs(vgg_sol - vgg_tv)
# CS231n AlexNet CONV1 parameter-sharing example
print(f"    AlexNet CONV1 without sharing: 55*55*96*(363+1) = {55 * 55 * 96 * 364:,}; with sharing: 96*(363+1) = {96 * 364:,} "
      f"(96*363 = {96 * 363:,} weights + 96 biases)")

# ---------------------------------------------------------------- 7: exercise tests vs solutions
ex_status = {}
pairs = [('exercise_01_conv_forward.py', sol1, ['conv_forward']),
         ('exercise_02_pooling.py', sol2, ['pool_forward', 'pool_backward']),
         ('exercise_03_output_sizes.py', sol3, ['compute_output_size']),
         ('exercise_04_parameter_count.py', sol4, ['count_parameters']),
         ('exercise_05_feature_viz.py', sol5, ['visualize_filters', 'compute_activation_map', 'plot_activation_maps'])]
import matplotlib  # noqa: E402
matplotlib.use('Agg')
for fname, sol, names in pairs:
    ex = load(os.path.join(DAY, 'exercises', fname), fname[:-3])
    for n in names:
        if hasattr(sol, n):
            setattr(ex, n, getattr(sol, n))
    try:
        cwd = os.getcwd(); os.chdir(tempfile.gettempdir())  # check() may save plots: not into the repo
        ex.check()
        ex_status[fname] = 'PASS'
    except Exception as e:  # noqa: BLE001
        ex_status[fname] = f'FAIL: {type(e).__name__}: {e}'
    finally:
        os.chdir(cwd)
print('[7] exercise check() with solution functions swapped in:', ex_status)

# ---------------------------------------------------------------- 8: train_minimal cost estimate
model = impl.SimpleCNN(num_classes=10)
xb = rng.standard_normal((8, 3, 32, 32)).astype(np.float32)
t0 = time.time()
for _ in range(3):
    model.forward(xb)
t_fwd = (time.time() - t0) / 3
n_params = model.count_parameters()
per_batch = (2 * n_params + 1) * t_fwd
n_batches = 200 // 8
print(f"[8] SimpleCNN params = {n_params:,}; one forward on a batch of 8 = {t_fwd * 1e3:.1f} ms; "
      f"numerical gradient needs 2*params forwards per batch -> ~{per_batch / 3600:.2f} h per batch, "
      f"~{per_batch * n_batches * 3 / 86400:.1f} days for the default 3 epochs x {n_batches} batches")

print('\nSUMMARY')
for k, v in results.items():
    print(f'  {k}: {v:.3e}' if isinstance(v, float) else f'  {k}: {v}')
ok = all(v < 1e-10 for k, v in results.items() if 'count' not in k) and results['VGG-16 param count |solution - torchvision|'] == 0
print('ALL PASS' if ok and all(s == 'PASS' for s in ex_status.values()) else 'SOME CHECKS FAILED')
