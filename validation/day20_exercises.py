"""Day 20 exercise check.
(1) every exercise file parses (they did not: `x = # YOUR CODE HERE` is a SyntaxError);
(2) every untouched stub, run as a script, stops with NotImplementedError (not a crash elsewhere);
(3) reference solutions run as scripts from the day directory (300 s cap);
(4) the exercise test blocks pass with the solution's function/class swapped in (exercise namespace patched in place).

Usage: OMP_NUM_THREADS=1 python validation/day20_exercises.py [path/to/20_Relational_RNNs]
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
DAY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "20_Relational_RNNs"))
os.chdir(DAY)
sys.path.insert(0, DAY)
sys.path.insert(0, os.path.join(DAY, "exercises"))
env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
exs = sorted(glob.glob("exercises/exercise_*.py"))

print("== parse + untouched-stub behaviour")
n_parse = 0
for ex in exs:
    try:
        ast.parse(open(ex).read())
        n_parse += 1
    except SyntaxError as e:
        print(f"  {os.path.basename(ex):30s} DOES NOT PARSE: {e.msg} (line {e.lineno})")
        continue
    p = subprocess.run([sys.executable, ex], capture_output=True, text=True, timeout=300, env=env)
    last = (p.stdout + p.stderr).strip().splitlines()[-1]
    print(f"  {os.path.basename(ex):30s} parses; stub run -> {last[:100]}")

print("\n== reference solutions as scripts")
n_sol = 0
sols = sorted(glob.glob("exercises/solutions/solution_*.py"))
for s in sols:
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, s], capture_output=True, text=True, timeout=300, env=env)
        rc, out = p.returncode, (p.stdout + p.stderr).strip().splitlines()
    except subprocess.TimeoutExpired:
        rc, out = "TIMEOUT", ["(300 s cap)"]
    n_sol += rc == 0
    tail = " / ".join(out[-2:]) if out else ""
    print(f"  {os.path.basename(s):30s} rc={rc} {time.time() - t0:5.1f}s | {tail[:110]}")


def load(path):
    src = open(path).read()
    ns = {"__name__": "not_main", "__file__": os.path.abspath(path)}
    exec(compile(src, path, "exec"), ns)
    return ns, src


print("\n== exercise test blocks with the solutions swapped in")
import torch  # noqa: E402
n_agree = 0
for ex in exs[:4]:                                    # exercise 5 is a training loop; covered by solution_05 above
    num = os.path.basename(ex).split("_")[1]
    sol = glob.glob(f"exercises/solutions/solution_{num}_*.py")[0]
    sol_ns, _ = load(sol)
    ok = 0
    for seed in range(10):
        torch.manual_seed(seed)
        ns, src = load(ex)
        names = [n.name for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and not n.name.startswith("test_")]
        ns.update({n: sol_ns[n] for n in names})
        block = next(n for n in ast.parse(src).body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test))
        ns["__name__"] = "__main__"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(ast.Module(body=block.body, type_ignores=[]), ex, "exec"), ns)
            ok += 1
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    n_agree += ok == 10
    print(f"  {os.path.basename(ex):30s} {'agree' if ok == 10 else 'DISAGREE'} (swapped {names}; {ok}/10 seeds)")

print(f"\nRESULT: {n_parse}/{len(exs)} exercises parse; {n_sol}/{len(sols)} solutions exit 0; "
      f"{n_agree}/4 exercise tests agree with solutions (exercise 5 = solution_05 run)")
