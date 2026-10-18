"""Day 15 exercise <-> solution agreement check.

For each exercise file: import it, swap in every class/function that the matching
solution file defines under the same name (the student's TODOs), then call every
test_* function the *exercise* file ships. If the exercise's own tests pass with the
solution's code, the exercise and its solution agree.

Run from anywhere:
  OMP_NUM_THREADS=1 python validation/day15_exercises_crosscheck.py
(plots the tests save go to validation/outputs/15_bahdanau_attention/)
"""
import importlib.util
import inspect
import os
import sys
import traceback

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "15_bahdanau_attention")
PAIRS = [
    ("exercise_1.py", "solution_1.py"),
    ("exercise_2.py", "solution_2.py"),
    ("exercise_3.py", "solution_3.py"),
    ("exercise_5.py", "solution_5.py"),
]
# exercise_4.py has no test_* functions (it is a training script); solution_4.py is run
# directly instead (trains the reversal task to 100% val accuracy in ~200 s).
sys.path.insert(0, os.path.join(DAY, "exercises", "solutions"))  # solution_4/5 import solution_1..3


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cross_check(ex_path, sol_path, tag):
    ex = load(ex_path, f"ex_{tag}")
    sol = load(sol_path, f"sol_{tag}")
    swapped = []
    for name, obj in vars(sol).items():
        if name.startswith("test_") or name.startswith("_"):
            continue
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and obj.__module__ == sol.__name__ \
                and hasattr(ex, name):
            setattr(ex, name, obj)
            swapped.append(name)
    results = {}
    for name, fn in list(vars(ex).items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            np.random.seed(0)
            torch.manual_seed(0)
            try:
                fn()
                results[name] = "pass"
            except Exception as e:  # noqa: BLE001
                results[name] = f"FAIL {type(e).__name__}: {e}"
                traceback.print_exc()
    return swapped, results


if __name__ == "__main__":
    # the exercise tests save plots to the current directory (exercise_5 writes test_attention.png):
    # run them from validation/outputs/<day>/ so nothing is written under papers/
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", os.path.basename(DAY))
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    total = ok = 0
    for ex_name, sol_name in PAIRS:
        swapped, res = cross_check(os.path.join(DAY, "exercises", ex_name),
                                   os.path.join(DAY, "exercises", "solutions", sol_name),
                                   ex_name[:10])
        n_pass = sum(v == "pass" for v in res.values())
        total += 1
        ok += n_pass == len(res) and len(res) > 0
        print(f"{ex_name}: swapped {swapped}; exercise tests {n_pass}/{len(res)} pass")
        for k, v in res.items():
            if v != "pass":
                print(f"   {k}: {v}")
    print(f"SUMMARY: {ok}/{total} exercise files fully agree with their solutions")
