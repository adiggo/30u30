"""Exercise <-> solution agreement (Days 06, 26, 27).

For each exercise file: import it as a module (its __main__ block does not run), overwrite every
top-level function/class that the solution also defines with the solution's version, then
execute the exercise's own `if __name__ == "__main__":` block (its tests / asserts / PASS-FAIL
prints) in the exercise's namespace.  This verifies that the exercise harness and the
reference solution agree.  Usage:  python validation/day06_exercise_agreement.py [day_dir]
The plots that the exercise blocks save go to validation/outputs/<day dir name>/, not into papers/.
"""
import ast
import contextlib
import importlib.util
import io
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def _defs(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}, tree

def run_day(day_dir):
    day_dir = os.path.abspath(day_dir)
    # the exercise __main__ blocks save plots to the current directory: run them from
    # validation/outputs/<day>/ so nothing is written under papers/
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", os.path.basename(day_dir))
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir); sys.path.insert(0, day_dir)
    ex_dir = os.path.join(day_dir, "exercises"); sol_dir = os.path.join(ex_dir, "solutions")
    results = []
    for f in sorted(os.listdir(ex_dir)):
        if not (f.startswith("exercise_") and f.endswith(".py")):
            continue
        ex_path = os.path.join(ex_dir, f)
        sol_path = os.path.join(sol_dir, f.replace("exercise_", "solution_"))
        ex_names, tree = _defs(ex_path); sol_names, _ = _defs(sol_path)
        swap = sorted(ex_names & sol_names)
        buf = io.StringIO(); status = "ok"
        try:
            with contextlib.redirect_stdout(buf):
                ex = _load(ex_path, "ex_" + f[:-3]); sol = _load(sol_path, "sol_" + f[:-3])
                for n in swap:
                    setattr(ex, n, getattr(sol, n))
                main = [n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)]
                code = compile(ast.Module(body=main[0].body, type_ignores=[]), ex_path, "exec")
                exec(code, ex.__dict__)
        except Exception as e:  # noqa: BLE001 - report any failure of the exercise's own checks
            status = f"EXCEPTION {type(e).__name__}: {e}"
            traceback.print_exc(file=buf)
        out = buf.getvalue()
        fails = [l.strip() for l in out.splitlines() if "FAIL" in l or "[NOTE]" in l]
        passes = sum(("PASS" in l or "[OK]" in l) for l in out.splitlines())
        results.append((f, swap, status, passes, fails))
        print(f"{f}: swapped {swap} -> {status}; PASS/OK lines={passes}; FAIL lines={fails[:3]}")
    return results

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    run_day(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "06_Complexodynamics"))
