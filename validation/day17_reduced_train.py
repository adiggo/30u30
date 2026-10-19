"""Day 17 reduced-configuration training check (the full train_minimal.py did ~1.3 s/iteration on one CPU thread
and had done only ~1,100 of its 10,001 iterations after 25 minutes).
[2026-10-18] Since the vectorized shift, an iteration takes ~0.1 s and train_minimal.py's default run (3,000
iterations, paper settings) learns the task itself; "save" now runs it with --iters 21.

mode "save"  : execute train_minimal.py as a script with --iters 21 to see whether it reaches its end
               (torch.save) without crashing; the checkpoint it saves goes to
               validation/outputs/day17_reduced_train/ instead of the day directory.
mode "learn" : same model / optimizer / clipping as train_minimal.py (RMSprop lr 1e-4, momentum 0.9, alpha 0.95,
               clip_grad_norm 10, controller 100, M = 20, batch 32) but N = 32 slots and lengths 1..5, for a fixed
               wall-clock budget; every 100 iterations it reports BCE and bit error rate on a FIXED held-out batch
               (64 sequences, lengths 1..5), so the curve is not dominated by the random training length.

Usage: OMP_NUM_THREADS=1 python validation/day17_reduced_train.py papers/17_neural_turing_machines save|learn [budget_seconds]
"""
import os
import random
import sys
import time

DAY, MODE = os.path.abspath(sys.argv[1]), sys.argv[2]
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 240.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "day17_reduced_train")
os.makedirs(OUT, exist_ok=True)
os.chdir(OUT)  # keep anything written to the current directory out of papers/
sys.path.insert(0, DAY)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

if MODE == "save":
    tm_path = os.path.join(DAY, "train_minimal.py")
    src = open(tm_path).read()
    sys.argv = [tm_path, "--iters", "21", "--eval-every", "10"]  # [2026-10-18] run length is a CLI flag now
    t0 = time.time()
    try:
        # train_minimal.py saves its checkpoint next to __file__: point __file__ into OUT
        exec(compile(src, tm_path, "exec"),
             {"__name__": "__main__", "__file__": os.path.join(OUT, "train_minimal.py")})
        print(f"SAVE-CHECK: completed in {time.time() - t0:.0f}s")
    except Exception as e:  # noqa: BLE001
        print(f"SAVE-CHECK: CRASH after {time.time() - t0:.0f}s -> {type(e).__name__}: {e}")
    sys.exit(0)

from implementation import NTM  # noqa: E402
from train_minimal import generate_copy_instance  # noqa: E402

torch.manual_seed(0); random.seed(0)
W, N, M, H, B, MAXLEN = 8, 32, 20, 100, 32, 5
model = NTM(W + 1, W, H, N, M)
opt = torch.optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9, alpha=0.95)
crit = nn.BCEWithLogitsLoss()

# fixed evaluation set: 64 sequences, lengths 1..5 (padded per length group)
torch.manual_seed(123)
evalset = [generate_copy_instance(16, L, W) for L in (1, 2, 3, 4, 5)]


def run(inputs, batch):
    st = model.reset(batch); outs = []
    for t in range(inputs.size(0)):
        o, st = model(inputs[t], st); outs.append(o)
    return torch.stack(outs)


def evaluate():
    with torch.no_grad():
        losses, errs, bits = 0.0, 0.0, 0
        for (x, y) in evalset:
            L = (x.size(0) - 1) // 2
            o = run(x, x.size(1))[L + 1:]
            losses += crit(o, y[L + 1:]).item() * y[L + 1:].numel()
            errs += ((o > 0).float() != y[L + 1:]).sum().item()
            bits += y[L + 1:].numel()
    return losses / bits, errs / bits


torch.manual_seed(0)
t0, it = time.time(), 0
print(f"reduced config: N={N}, M={M}, controller={H}, batch={B}, lengths 1..{MAXLEN}; budget {BUDGET:.0f}s")
while time.time() - t0 < BUDGET:
    if it % 100 == 0:
        l, ber = evaluate()
        print(f"iter {it:5d}  t={time.time() - t0:5.0f}s  eval BCE {l:.4f}  eval BER {100 * ber:5.2f}%", flush=True)
    L = random.randint(1, MAXLEN)
    x, y = generate_copy_instance(B, L, W)
    opt.zero_grad()
    loss = crit(run(x, B)[L + 1:], y[L + 1:])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    it += 1
l, ber = evaluate()
print(f"final iter {it:5d}  t={time.time() - t0:5.0f}s  eval BCE {l:.4f}  eval BER {100 * ber:5.2f}%  "
      f"({(time.time() - t0) / max(it, 1):.3f} s/iter); chance BER = 50%")
