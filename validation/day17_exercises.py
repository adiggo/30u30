"""Day 17 exercise check: (1) run every reference solution as a script; (2) load every exercise file,
swap in the solution's same-named functions/classes, then run the exercise's own `if __name__ == "__main__":`
test block, so the exercise's tests are checked against the reference implementation.

Usage: OMP_NUM_THREADS=1 python validation/day17_exercises.py [path/to/day_dir] [n_seeds]
"""
import ast
import glob
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "17_neural_turing_machines"))
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def load_namespace(path, run_name="not_main"):
    src = open(path).read()
    ns = {"__name__": run_name, "__file__": path}
    exec(compile(src, path, "exec"), ns)
    return ns, src


def main_block(src, path):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            return compile(ast.Module(body=node.body, type_ignores=[]), path + ":__main__", "exec")
    return None


def defined_names(src):
    return [n.name for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]


os.chdir(DAY)
sys.path.insert(0, DAY)
sys.path.insert(0, os.path.join(DAY, "exercises"))
env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")

print("== reference solutions, run as scripts (300 s cap each)")
sol_ok = 0
sols = sorted(glob.glob("exercises/solutions/solution_*.py"))
for s in sols:
    try:
        p = subprocess.run([sys.executable, s], capture_output=True, text=True, timeout=300, env=env)
        status = "pass" if p.returncode == 0 else f"FAIL rc={p.returncode}"
        tail = (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]
    except subprocess.TimeoutExpired:
        status, tail = "TIMEOUT", [""]
    sol_ok += status == "pass"
    print(f"  {os.path.basename(s):40s} {status:12s} | {tail[0][:110]}")

print(f"\n== exercise tests run against the solution implementations ({N_SEEDS} seeds each)")
agree = 0
exs = sorted(glob.glob("exercises/exercise_*.py"))
for ex in exs:
    num = os.path.basename(ex).split("_")[1]
    cand = glob.glob(f"exercises/solutions/solution_{num}_*.py")
    try:
        ex_ns, ex_src = load_namespace(ex)
    except SyntaxError as e:
        print(f"  {os.path.basename(ex):40s} DOES NOT PARSE: {e}")
        continue
    if not cand:
        print(f"  {os.path.basename(ex):40s} no matching solution")
        continue
    sol_ns, sol_src = load_namespace(cand[0])
    swapped = [n for n in defined_names(ex_src) if n in sol_ns]
    block = main_block(ex_src, ex)
    if block is None:
        print(f"  {os.path.basename(ex):40s} has no __main__ test block; swapped {swapped}")
        continue
    fails = []
    import torch
    import numpy as np
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        ns, _ = load_namespace(ex)                 # fresh namespace, patched IN PLACE so any test function
        ns["__name__"] = "__main__"                # defined in the exercise (globals = ns) sees the solution
        for n in swapped:
            ns[n] = sol_ns[n]
        try:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                exec(block, ns)
        except Exception as e:  # noqa: BLE001
            fails.append(f"seed {seed}: {type(e).__name__}: {str(e)[:80]}")
    ok = not fails
    agree += ok
    print(f"  {os.path.basename(ex):40s} {'agree' if ok else 'DISAGREE'} (swapped {swapped}; "
          f"{N_SEEDS - len(fails)}/{N_SEEDS} seeds pass){'' if ok else ' first: ' + fails[0]}")

print(f"\nRESULT: {sol_ok}/{len(sols)} solutions pass; {agree}/{len(exs)} exercise test blocks agree with solutions")
