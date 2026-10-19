"""Day 21 exercise check.
(1) every reference solution runs as a script from the day directory (300 s cap);
(2) every exercise stub runs and reports NOT IMPLEMENTED (no crash);
(3) every exercise test function passes with the solution's same-named functions/classes swapped into
    the exercise namespace (patched in place, so the test's own globals see them). The tests print
    PASSED / FAILED / NOT IMPLEMENTED rather than raising, so stdout is parsed.

Usage: OMP_NUM_THREADS=1 python validation/day21_exercises.py [path/to/21_Neural_Message_Passing]
"""
import ast
import contextlib
import glob
import io
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "21_Neural_Message_Passing"))
os.chdir(DAY)
sys.path.insert(0, DAY)
env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")

print("== reference solutions as scripts")
sols = sorted(glob.glob("exercises/solutions/solution_*.py"))
n_sol = 0
for s in sols:
    t0 = time.time()
    p = subprocess.run([sys.executable, s], capture_output=True, text=True, timeout=300, env=env)
    out = (p.stdout + p.stderr).strip().splitlines()
    n_sol += p.returncode == 0 and not any("FAIL" in line for line in out)
    print(f"  {os.path.basename(s):40s} rc={p.returncode} {time.time() - t0:4.1f}s | {out[-1][:70] if out else ''}")

print("\n== exercise stubs as scripts")
exs = sorted(glob.glob("exercises/exercise_*.py"))
for ex in exs:
    p = subprocess.run([sys.executable, ex], capture_output=True, text=True, timeout=300, env=env)
    txt = p.stdout + p.stderr
    print(f"  {os.path.basename(ex):40s} rc={p.returncode} NOT-IMPLEMENTED markers: {txt.count('NOT IMPLEMENTED')}"
          f"{' (traceback!)' if 'Traceback' in txt else ''}")


def load(path):
    src = open(path).read()
    ns = {"__name__": "not_main", "__file__": os.path.abspath(path)}
    exec(compile(src, path, "exec"), ns)
    return ns, src


print("\n== exercise tests with solutions swapped in")
n_agree = 0
for ex in exs:
    num = os.path.basename(ex).split("_")[1]
    sol = glob.glob(f"exercises/solutions/solution_{num}_*.py")[0]
    sol_ns, _ = load(sol)
    ns, src = load(ex)
    names = [n.name for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and not n.name.startswith("test_") and n.name in sol_ns]
    ns.update({n: sol_ns[n] for n in names})
    test = next(n.name for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns[test]()
    txt = buf.getvalue()
    ok = "PASSED" in txt and "FAILED" not in txt and "NOT IMPLEMENTED" not in txt
    n_agree += ok
    print(f"  {os.path.basename(ex):40s} {'agree' if ok else 'DISAGREE'} (swapped {names})"
          + ("" if ok else "\n" + txt[-400:]))

print(f"\nRESULT: {n_sol}/{len(sols)} solutions pass; {n_agree}/{len(exs)} exercise tests agree with solutions")
