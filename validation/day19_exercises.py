"""Day 19 exercise check.
(1) Run every reference solution as a script from the day directory (300 s cap).
(2) Exercises with test functions (01, 02, 04, 06): load the exercise, swap in the solution's same-named
    function/class, run the exercise's own __main__ test block; a test that prints [FAIL] or raises counts
    as disagreement.
(3) Exercises 03, 05, 07 are open-ended stubs (no test); the solutions use different names
    (verify_permutation_invariance, test_spatial_logic, prove_counting_bias), so they are called directly.

Usage: OMP_NUM_THREADS=1 python validation/day19_exercises.py [path/to/19_relational_reasoning]
"""
import ast
import contextlib
import glob
import io
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "19_relational_reasoning"))
os.chdir(DAY)
sys.path.insert(0, DAY)
env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")


def load(path, name="not_main"):
    src = open(path).read()
    ns = {"__name__": name, "__file__": os.path.abspath(path)}
    exec(compile(src, path, "exec"), ns)
    return ns, src


def main_block(src, path):
    for node in ast.parse(src).body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            return compile(ast.Module(body=node.body, type_ignores=[]), path + ":__main__", "exec")


print("== reference solutions run as scripts")
sols = sorted(glob.glob("exercises/solutions/solution_*.py"))
n_ok = 0
for s in sols:
    p = subprocess.run([sys.executable, s], capture_output=True, text=True, timeout=300, env=env)
    out = (p.stdout + p.stderr).strip().splitlines()
    n_ok += p.returncode == 0
    print(f"  {os.path.basename(s):36s} rc={p.returncode} | {(out[-1] if out else '(no output: defines functions only)')[:90]}")

print("\n== exercise tests against the solutions")
agree = 0
import torch  # noqa: E402
for num, ex, sol in [("01", "exercise_01_pair_generator", "solution_01_pair_generator"),
                     ("02", "exercise_02_g_theta", "solution_02_g_theta"),
                     ("04", "exercise_04_sort_of_clevr", "solution_04_sort_of_clevr"),
                     ("06", "exercise_06_masking", "solution_06_masking")]:
    ex_ns, ex_src = load(f"exercises/{ex}.py")
    sol_ns, sol_src = load(f"exercises/solutions/{sol}.py")
    names = [n.name for n in ast.parse(ex_src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    swapped = [n for n in names if n in sol_ns]
    fails = 0
    for seed in range(10):
        torch.manual_seed(seed)
        ns, _ = load(f"exercises/{ex}.py")          # fresh namespace; patch it IN PLACE so the test
        ns.update({n: sol_ns[n] for n in swapped})  # functions (whose __globals__ is ns) see the solution
        ns["__name__"] = "__main__"
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(main_block(ex_src, ex), ns)
            fails += "FAIL" in buf.getvalue()
        except Exception as e:  # noqa: BLE001
            fails += 1
            last = f"{type(e).__name__}: {e}"
    agree += fails == 0
    print(f"  {ex:36s} {'agree' if fails == 0 else 'DISAGREE'} (swapped {swapped}; {10 - fails}/10 seeds)")

from implementation import RelationNetwork  # noqa: E402
s3, _ = load("exercises/solutions/solution_03_permutation.py")
r3 = all(s3["verify_permutation_invariance"](RelationNetwork(object_dim=8, output_dim=4), torch.randn(1, 5, 8))
         for _ in range(10))
s5, _ = load("exercises/solutions/solution_05_coordinates.py")
try:
    with contextlib.redirect_stdout(io.StringIO()):
        s5["test_spatial_logic"]()
    r5 = True
except Exception:  # noqa: BLE001
    r5 = False
s7, _ = load("exercises/solutions/solution_07_counting.py")
r7 = all(s7["prove_counting_bias"](n) for n in [2, 8, 17])
for name, r in [("exercise_03 via verify_permutation_invariance", r3), ("exercise_05 via test_spatial_logic", r5),
                ("exercise_07 via prove_counting_bias", r7)]:
    agree += r
    print(f"  {name:52s} {'pass' if r else 'FAIL'}")
print("  note: solution_07 sums a vector of ones; it never builds a RelationNetwork as exercise 7 asks")

print(f"\nRESULT: {n_ok}/{len(sols)} solutions exit 0; {agree}/7 exercises agree with / are satisfied by the solutions")
