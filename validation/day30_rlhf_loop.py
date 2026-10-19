"""Day 30: the exercise-5 RLHF loop (exercises/solutions/solution_05_rlhf_loop.py) on CartPole.

1. The exercise's own harness (exercise_05_rlhf_loop.py __main__) with the solution's run_rlhf_step
   swapped in: three cycles run and return the documented keys.
2. For each seed: 30 RLHF cycles (policy trained on the learned reward only) and the solution's control
   (same PPO on a constant +1 reward, no labels). Reports the true reward of the first 3 and last 5
   cycles and the reward model's accuracy on new pairs before it trains on them.
   PASS if the RLHF policy's last-5 true reward exceeds twice a uniformly random policy (~22) for
   every seed. The control is informational: in CartPole the labels only encode segment length.

Run:  OMP_NUM_THREADS=1 python validation/day30_rlhf_loop.py [seeds=0,1,2] [cycles=30]
"""
import importlib.util
import io
import os
import sys
import textwrap
import time
from contextlib import redirect_stdout

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
EX = os.path.join(REPO, 'papers', '30_RLHF', 'exercises')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sol = load('sol05', os.path.join(EX, 'solutions', 'solution_05_rlhf_loop.py'))
ex = load('ex05', os.path.join(EX, 'exercise_05_rlhf_loop.py'))

# ---- 1. exercise harness with the solution swapped in -----------------------------------------
ex.run_rlhf_step = sol.run_rlhf_step
harness = open(os.path.join(EX, 'exercise_05_rlhf_loop.py')).read().split('if __name__ == "__main__":')[1]
buf = io.StringIO()
with redirect_stdout(buf):
    exec(textwrap.dedent(harness), ex.__dict__)   # the exercise's __main__ block, in its own namespace
lines = buf.getvalue().strip().splitlines()
ok1 = len(lines) == 4 and lines[0].startswith('cycle 1:')
print(f"[{'PASS' if ok1 else 'FAIL'}] exercise_05 harness with solution's run_rlhf_step: {lines[-2] if lines else ''}")

# ---- 2. learning runs -------------------------------------------------------------------------
seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else '0,1,2').split(',')]
cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 30
RANDOM = 22.0
ok2 = True
print(f"\n{cycles} cycles per run; uniformly random policy ~{RANDOM:.0f}")
print(f"{'seed':>4s} {'RLHF first-3':>13s} {'last-5':>8s} {'RM acc new pairs':>17s} {'untrained':>10s} "
      f"{'control first-3':>16s} {'last-5':>8s} {'time':>6s}")
for seed in seeds:
    t0 = time.time()
    hist = sol.run_rlhf(cycles=cycles, seed=seed, verbose=False)
    ctrl = sol.run_constant_reward_control(cycles=cycles, seed=seed)
    tr = [h['true_reward'] for h in hist]
    acc = [h['heldout_acc'] for h in hist if not np.isnan(h['heldout_acc'])]
    ok2 &= np.mean(tr[-5:]) > 2 * RANDOM
    print(f"{seed:4d} {np.mean(tr[:3]):13.1f} {np.mean(tr[-5:]):8.1f} {100 * np.mean(acc):16.0f}% "
          f"{100 * hist[0]['heldout_acc']:9.0f}% {np.mean(ctrl[:3]):16.1f} {np.mean(ctrl[-5:]):8.1f} "
          f"{time.time() - t0:5.0f}s", flush=True)
print(f"[{'PASS' if ok2 else 'FAIL'}] RLHF last-5 true reward > 2 x random policy for every seed")
