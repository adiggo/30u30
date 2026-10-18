"""Day 24 (GPipe) reference checks. The repo simulates all K stages on ONE device (no real
inter-device transfer or concurrency); these checks verify the math the simulation is meant to show.

1. GPipe(model).forward + one backward on the concatenated output == plain sequential full-batch
   gradients (no checkpointing).                                              -> max |dgrad|
2. Re-materialization (torch.utils.checkpoint inside each PipelineStage) gives the same gradients,
   and each stage's forward is re-run exactly once per micro-batch during backward.
3. Gradient accumulation over M micro-batches with separate backward() calls:
   sum_m (n_m/N) * dL_m/dw == dL/dw (full batch, mean loss)                    -> max |dgrad|
   and the unscaled version (exercise solution_03) gives exactly M x the full-batch gradient.
4. Bubble fraction: a discrete-event simulation of the GPipe schedule (all M forwards through K
   stages, then all M backwards in reverse; each stage-step takes t_f forward, t_b backward)
   gives idle fraction == (K-1)/(M+K-1) for all K, M tested.
5. The implementation's forward loop visits (micro-batch m, stage k) at clock step m+k, uses
   M+K-1 steps, and never runs stage k on m before stage k-1 finished m.

Run:  OMP_NUM_THREADS=1 python validation/day24_gpipe_equivalence.py
"""
import copy
import os
import sys

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '24_gpipe')
sys.path.insert(0, DAY)
from implementation import GPipe  # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)  # isolate algorithmic differences from fp32 rounding


def make_model(d=16, L=8):
    layers = []
    for _ in range(L):
        layers += [nn.Linear(d, d), nn.Tanh()]
    return nn.Sequential(*layers, nn.Linear(d, 3))


def grads(m):
    return torch.cat([p.grad.flatten() for p in m.parameters()])


def check_equivalence():
    worst_plain, worst_ckpt = 0.0, 0.0
    for K, M, N in [(2, 4, 16), (4, 8, 64), (3, 5, 23), (4, 4, 8)]:  # (3,5,23): uneven torch.chunk split
        base = make_model()
        x, y = torch.randn(N, 16), torch.randn(N, 3)
        ref = copy.deepcopy(base)
        nn.functional.mse_loss(ref(x), y).backward()
        g_ref = grads(ref)
        for ckpt in (False, True):
            g = GPipe(copy.deepcopy(base), n_partitions=K, n_microbatches=M, use_checkpoint=ckpt)
            g.train()
            nn.functional.mse_loss(g(x), y).backward()
            d = (grads(g) - g_ref).abs().max().item()
            if ckpt:
                worst_ckpt = max(worst_ckpt, d)
            else:
                worst_plain = max(worst_plain, d)
    print(f'[1] GPipe (no checkpoint) vs sequential full batch: max |dgrad| = {worst_plain:.2e}')
    print(f'[2] GPipe with re-materialization vs sequential:    max |dgrad| = {worst_ckpt:.2e}')
    return worst_plain, worst_ckpt


def check_recompute_count():
    K, M = 4, 6
    g = GPipe(make_model(), n_partitions=K, n_microbatches=M, use_checkpoint=True)
    g.train()
    counts = {}
    for k, stage in enumerate(g.stages):
        stage.layers[0].register_forward_hook(lambda mod, i, o, k=k: counts.__setitem__(k, counts.get(k, 0) + 1))
    out = g(torch.randn(24, 16))
    fwd = dict(counts)
    out.sum().backward()
    bwd = {k: counts[k] - fwd[k] for k in counts}
    print(f'[2] forward calls per stage during forward: {fwd}; extra calls during backward (recompute): {bwd}')
    assert all(v == M for v in fwd.values()) and all(v == M for v in bwd.values())


def check_accumulation():
    model = make_model()
    N, M = 24, 4
    x, y = torch.randn(N, 16), torch.randn(N, 3)
    ref = copy.deepcopy(model)
    nn.functional.mse_loss(ref(x), y).backward()
    g_ref = grads(ref)
    scaled, unscaled = copy.deepcopy(model), copy.deepcopy(model)
    for xs, ys in zip(torch.chunk(x, M), torch.chunk(y, M)):
        nn.functional.mse_loss(scaled(xs), ys).mul(xs.shape[0] / N).backward()  # weight n_m / N
        nn.functional.mse_loss(unscaled(xs), ys).backward()                     # solution_03 as shipped
    d = (grads(scaled) - g_ref).abs().max().item()
    ratio = (grads(unscaled) / g_ref)
    print(f'[3] accumulated (loss_m * n_m/N) vs full batch: max |dgrad| = {d:.2e}')
    print(f'[3] unscaled accumulation (solution_03 before fix) / full-batch grad: '
          f'min {ratio.min().item():.4f}, max {ratio.max().item():.4f} (M = {M})')
    return d


def simulate_bubble(K, M, t_f=1.0, t_b=2.0):
    """Event simulation of the synchronous GPipe schedule: F(k, m) needs F(k-1, m) and F(k, m-1);
    B(k, m) needs B(k+1, m) and B(k, m+1) (reverse order), and all backwards start after the last forward."""
    fin = {}
    dev_free = [0.0] * K
    for m in range(M):
        for k in range(K):
            start = max(dev_free[k], fin.get(('F', k - 1, m), 0.0))
            fin[('F', k, m)] = start + t_f
            dev_free[k] = start + t_f
    for m in reversed(range(M)):
        for k in reversed(range(K)):
            start = max(dev_free[k], fin.get(('B', k + 1, m), fin[('F', K - 1, M - 1)]))
            fin[('B', k, m)] = start + t_b
            dev_free[k] = start + t_b
    makespan = max(fin.values())
    busy = M * (t_f + t_b)
    return 1 - busy / makespan


def check_bubble():
    worst = 0.0
    rows = []
    for K in (2, 4, 8):
        for M in (1, 4, 8, 32):
            sim = simulate_bubble(K, M)
            formula = (K - 1) / (M + K - 1)
            worst = max(worst, abs(sim - formula))
            rows.append(f'K={K},M={M}: sim {sim:.4f} formula {formula:.4f}')
    print('[4] ' + '; '.join(rows[:6]) + ' ...')
    print(f'[4] bubble fraction simulation vs (K-1)/(M+K-1): max |d| = {worst:.2e}; '
          f'K=4,M=32 -> {3/35:.4f} (CHEATSHEET says 3/34)')
    return worst


def check_schedule_order():
    K, M = 4, 6
    g = GPipe(make_model(), n_partitions=K, n_microbatches=M, use_checkpoint=False)
    order = []
    for k, stage in enumerate(g.stages):
        stage.register_forward_pre_hook(lambda mod, inp, k=k: order.append(k))
    # recover micro-batch index from the call order within each clock step
    g(torch.randn(24, 16))
    steps = M + K - 1
    expected = []
    for s in range(steps):
        for m in range(M):
            k = s - m
            if 0 <= k < K:
                expected.append(k)
    print(f'[5] stage call sequence matches clock-step schedule over M+K-1={steps} steps: {order == expected}')
    assert order == expected


def check_solution_03():
    """The exercise solution's accumulation must reproduce the full-batch gradient."""
    sys.path.insert(0, os.path.join(DAY, 'exercises', 'solutions'))
    from solution_03_gradients import compute_accumulated_gradients
    model = make_model()
    N, M = 20, 4
    x, y = torch.randn(N, 16), torch.randn(N, 3)
    ref = copy.deepcopy(model)
    nn.functional.mse_loss(ref(x), y).backward()
    acc = copy.deepcopy(model)
    compute_accumulated_gradients(acc, list(torch.chunk(x, M)), list(torch.chunk(y, M)), nn.MSELoss())
    d = (grads(acc) - grads(ref)).abs().max().item()
    ratio = (grads(acc) / grads(ref)).mean().item()
    print(f'[3] exercises/solutions/solution_03 vs full batch: max |dgrad| = {d:.2e} (mean ratio {ratio:.4f})')
    return d


if __name__ == '__main__':
    check_solution_03()
    a, b = check_equivalence()
    check_recompute_count()
    c = check_accumulation()
    d = check_bubble()
    check_schedule_order()
    print(f'SUMMARY day24: plain {a:.1e}, ckpt {b:.1e}, accumulation {c:.1e}, bubble {d:.1e}')
