"""Day 07 check: the interacting coffee automaton must conserve cream particles.

Paper (Aaronson, Carroll, Ouellette 2014, Sec. 3.1): "at each time step, one pair of
horizontally or vertically adjacent, differing particles is selected, and the particles'
positions are swapped."  Every step is a transposition, so sum(grid) is invariant.

The repo batches ~n^2/4 candidate swaps per step and applies them simultaneously with
NumPy fancy indexing.  When two selected pairs share a cell the simultaneous assignment is
not a permutation, so cream is created/destroyed.  This script runs the ORIGINAL code
(from git at ORIG_REF, before the fix) and the PATCHED code side by side, reports the
particle-count drift, and checks that every patched batch is a set of disjoint adjacent
transpositions.

Run from anywhere:  OMP_NUM_THREADS=1 python validation/day07_particle_conservation.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "07_coffee_automaton")
# Pre-fix code for the original-vs-patched comparison: the commit the fixes on this branch were
# made on (read with `git show`, no checkout; needs the repo's git history).
ORIG_REF = "195dcf2"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Original file straight from git at ORIG_REF (read-only: `git show`, no checkout).
orig_src = subprocess.run(["git", "-C", REPO, "show", ORIG_REF + ":papers/07_coffee_automaton/implementation.py"],
                          capture_output=True, text=True, check=True).stdout
tmp = tempfile.NamedTemporaryFile("w", suffix="_impl07_orig.py", delete=False)
tmp.write(orig_src)
tmp.close()
orig = load_module("impl07_orig", tmp.name)
patched = load_module("impl07_patched", DAY + "/implementation.py")


def drift(mod, n=50, batches=100, per=1000, seed=42):
    """Run n x n interacting model; return initial count and min/max/final counts."""
    ca = mod.CoffeeAutomaton(n, "interacting", seed=seed)
    n0 = int(ca.grid.sum())
    counts = []
    for _ in range(batches):
        ca.step(per)
        counts.append(int(ca.grid.sum()))
    return n0, min(counts), max(counts), counts[-1]


def check_disjoint_transpositions(mod, n=20, steps=300, seed=0):
    """After each single batch, the set of changed cells must pair up into adjacent
    cells whose values were exchanged (i.e. the batch is a product of disjoint swaps)."""
    ca = mod.CoffeeAutomaton(n, "interacting", seed=seed)
    ca.step(40)  # get some mixing first so many pairs differ
    bad = 0
    for _ in range(steps):
        before = ca.grid.copy()
        ca.step(1)
        after = ca.grid
        changed = np.argwhere(before != after)          # (k, 2) array of (row, col)
        # Each changed cell flipped 0<->1.  A disjoint-swap batch flips equal numbers of
        # 0->1 and 1->0 cells, and every changed cell has a changed 4-neighbour.
        ups = int(((before == 0) & (after == 1)).sum())
        downs = int(((before == 1) & (after == 0)).sum())
        if ups != downs:
            bad += 1
            continue
        chg = np.zeros_like(before, dtype=bool)
        chg[changed[:, 0], changed[:, 1]] = True
        # neighbour-has-changed test via shifted views (pad with False at borders)
        p = np.pad(chg, 1)
        has_nb = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        if np.any(chg & ~has_nb):
            bad += 1
    return bad


if __name__ == "__main__":
    for label, mod in (("original", orig), ("patched", patched)):
        n0, lo, hi, fin = drift(mod)
        print(f"{label:8s}: initial cream={n0}  min={lo}  max={hi}  final={fin}  "
              f"max |drift|={max(abs(lo - n0), abs(hi - n0))}")
    bad = check_disjoint_transpositions(patched)
    print(f"patched: batches that are not disjoint adjacent transpositions: {bad}/300")
    n0, lo, hi, fin = drift(patched)
    ok = (lo == hi == n0) and bad == 0
    print("RESULT:", "PASS" if ok else "FAIL", "(patched conserves particles exactly)")
    sys.exit(0 if ok else 1)
