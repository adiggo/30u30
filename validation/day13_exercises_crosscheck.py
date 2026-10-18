"""Day 13 exercise <-> solution agreement check.

For each exercise file: import it, swap in every class/function that the matching
solution file defines under the same name (the student's TODOs), then call every
test_* function the *exercise* file ships. If the exercise's own tests pass with the
solution's code, the exercise and its solution agree.

Run from anywhere:
  OMP_NUM_THREADS=1 python validation/day13_exercises_crosscheck.py
"""
import importlib.util
import inspect
import os
import sys
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "13_attention")
PAIRS = [
    ("exercise_01_scaled_dot_product.py", "solution_01_scaled_dot_product.py"),
    ("exercise_02_multi_head_attention.py", "solution_02_multi_head_attention.py"),
    ("exercise_03_positional_encoding.py", "solution_03_positional_encoding.py"),
    ("exercise_04_encoder_block.py", "solution_04_encoder_block.py"),
    ("exercise_05_full_transformer.py", "solution_05_full_transformer.py"),
]


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
            try:
                fn()
                results[name] = "pass"
            except Exception as e:  # noqa: BLE001
                results[name] = f"FAIL {type(e).__name__}: {e}"
                traceback.print_exc()
    return swapped, results


if __name__ == "__main__":
    # run from validation/outputs/<day>/ so anything the tests save to the current directory
    # stays out of papers/
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", os.path.basename(DAY))
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    total = ok = 0
    for ex_name, sol_name in PAIRS:
        swapped, res = cross_check(os.path.join(DAY, "exercises", ex_name),
                                   os.path.join(DAY, "exercises", "solutions", sol_name),
                                   ex_name[:11])
        n_pass = sum(v == "pass" for v in res.values())
        total += 1
        ok += n_pass == len(res) and len(res) > 0
        print(f"{ex_name}: swapped {swapped}; exercise tests {n_pass}/{len(res)} pass")
        for k, v in res.items():
            if v != "pass":
                print(f"   {k}: {v}")
    print(f"SUMMARY: {ok}/{total} exercise files fully agree with their solutions")
