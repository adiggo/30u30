"""Day 18 exercise <-> solution agreement.

The Day 18 exercises have no test_* functions: each file's checks live in its
`if __name__ == "__main__":` block, and the solution files contain only the function/class.
For each pair: exec the exercise's top half, overwrite the TODO object with the solution's,
then exec the exercise's __main__ block and look for its [OK]/[FAIL] verdict.

Run: OMP_NUM_THREADS=1 python validation/day18_exercises_crosscheck.py
"""
import contextlib
import importlib.util
import io
import os
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
EX = os.path.join(REPO, "papers", "18_pointer_networks", "exercises")
PAIRS = [("exercise_01_pointer_attention.py", "solution_01_pointer_attention.py", "PointerAttention"),
         ("exercise_02_masking.py", "solution_02_masking.py", "apply_mask"),
         ("exercise_03_convex_hull.py", "solution_03_convex_hull.py", "get_convex_hull_pointers"),
         ("exercise_04_tsp_cost.py", "solution_04_tsp_cost.py", "calculate_tour_cost"),
         ("exercise_05_greedy_decoding.py", "solution_05_greedy_decoding.py", "greedy_decode")]

ok_count = 0
for ex_name, sol_name, obj in PAIRS:
    spec = importlib.util.spec_from_file_location(sol_name[:-3], f"{EX}/solutions/{sol_name}")
    sol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sol)
    src = open(f"{EX}/{ex_name}").read()
    top, main = src.split('if __name__ == "__main__":', 1)
    ns = {"__name__": "exercise"}
    exec(compile(top, ex_name, "exec"), ns)
    ns[obj] = getattr(sol, obj)                     # swap in the solution
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(textwrap.dedent(main), ex_name, "exec"), ns)
    out = buf.getvalue()
    verdict = "pass" if "[OK]" in out and "[FAIL]" not in out else "FAIL"
    ok_count += verdict == "pass"
    lines = [l.strip() for l in out.splitlines() if "[OK]" in l or "[FAIL]" in l or "Decoded" in l or "cost" in l.lower()]
    print(f"{ex_name} + {sol_name}: {verdict}  | " + " | ".join(lines))
print(f"SUMMARY: {ok_count}/{len(PAIRS)} exercise checks pass with the solution code")
