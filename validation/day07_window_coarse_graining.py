"""Day 07 check: the paper's centred-window coarse-graining and the other 2026-10-18 changes.

Paper (arXiv:1405.6903, Sec. 5.1): "we construct a new array in which the value of each cell is
the average of the values of the nearby cells in the fine-grained array. We define 'nearby' cells
as those within a g x g square centered at the cell in question."  Fig. 1 illustrates it with a
5 x 5 array and g = 3: the output is 5 x 5, and windows that stick out of the grid are averaged
over their in-bounds cells.

Checks:
  1. implementation.coarse_grain_window (2-D prefix sums) vs a direct loop over every window,
     odd and even g, 0/1 grids and particle-count grids, square and non-square;
  2. the paper's Fig. 1 example: how many of its 25 printed values (one decimal, truncated) the
     in-bounds window average reproduces, vs zero padding;
  3. coarse_grain(method='block') is still the non-overlapping block mean (reshape-mean);
  4. the non-interacting model's vectorized count rebuild (np.bincount) vs a per-particle loop,
     and particle conservation;
  5. the interacting model's swap counter: every batch changes exactly 2 x (new swaps) cells, the
     particle count is conserved, and paper_time == swaps;
  6. grid_to_bytes clips counts above 255 instead of wrapping.
Prints PASS/FAIL with the numbers; exits 0.
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
sys.path.insert(0, os.path.join(REPO, "papers", "07_coffee_automaton"))
import implementation as impl  # noqa: E402

results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def loop_window_mean(grid, g):
    """Direct definition: mean over the in-bounds cells of the window rows i-g//2 .. i+(g-1)//2
    (centred for odd g), same for columns."""
    n0, n1 = grid.shape
    lo, hi = g // 2, (g - 1) // 2
    out = np.zeros((n0, n1))
    for i in range(n0):
        for j in range(n1):
            total, count = 0.0, 0
            for k in range(i - lo, i + hi + 1):
                for m in range(j - lo, j + hi + 1):
                    if 0 <= k < n0 and 0 <= m < n1:
                        total += float(grid[k, m])
                        count += 1
            out[i, j] = total / count
    return out


# ---- 1. window vs loop -----------------------------------------------------------------------
rng = np.random.default_rng(0)
worst = 0.0
cases = 0
for shape in ((5, 5), (13, 13), (30, 30), (12, 17)):
    for g in (1, 2, 3, 4, 5, 7, 10):
        for kind in ("binary", "counts"):
            x = rng.integers(0, 2, size=shape) if kind == "binary" else rng.poisson(0.7, size=shape)
            got = impl.coarse_grain(x, g)                      # default method = 'window'
            ref = loop_window_mean(x, g)
            assert got.shape == x.shape
            worst = max(worst, float(np.abs(got - ref).max()))
            cases += 1
report("window (prefix sums) vs direct loop", worst <= 1e-12,
       f"{cases} cases (g = 1..10, odd and even, 0/1 and count grids), max |diff| = {worst:.1e} (tol 1e-12); "
       f"output shape == input shape")

# ---- 2. paper Fig. 1 -------------------------------------------------------------------------
fine = np.array([[0, 1, 0, 0, 1],
                 [0, 1, 1, 0, 1],
                 [0, 1, 0, 1, 0],
                 [1, 0, 1, 1, 1],
                 [1, 0, 1, 0, 0]])
printed = np.array([[.5, .5, .5, .5, .5],
                    [.5, .4, .4, .4, .5],
                    [.5, .5, .6, .6, .3],
                    [.5, .5, .5, .5, .5],
                    [.5, .6, .5, .6, .5]])
win = impl.coarse_grain(fine, 3)
trunc = np.floor(win * 10 + 1e-9) / 10                    # the figure prints one decimal, truncated
match = int((np.isclose(trunc, printed)).sum())
bad = [(int(i), int(j), round(float(win[i, j]), 3), float(printed[i, j]))
       for i, j in zip(*np.nonzero(~np.isclose(trunc, printed)))]
zero_pad = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        zero_pad[i, j] = fine[max(0, i - 1):i + 2, max(0, j - 1):j + 2].sum() / 9.0
match_zero = int(np.isclose(np.floor(zero_pad * 10 + 1e-9) / 10, printed).sum())
report("paper Fig. 1 (5 x 5, g = 3)", match == 23,
       f"in-bounds window average reproduces {match}/25 printed values (zero padding: {match_zero}/25); "
       f"mismatches (row, col, ours, printed): {bad}")

# ---- 3. block method unchanged ---------------------------------------------------------------
x = rng.integers(0, 2, size=(40, 40))
blk = impl.coarse_grain(x, 5, method="block")
ref = x.reshape(8, 5, 8, 5).mean(axis=(1, 3))             # (8,5,8,5): axes 1 and 3 run inside a block
report("block method = non-overlapping block mean", blk.shape == (8, 8) and np.allclose(blk, ref),
       f"shape {blk.shape}, max |diff| = {np.abs(blk - ref).max():.1e}")

# ---- 4. non-interacting count rebuild ----------------------------------------------------------
ca = impl.CoffeeAutomaton(30, "non_interacting", seed=3)
worst_cnt, conserved = 0, True
for k in (1, 5, 17):
    ca.step(k)
    loop = np.zeros((30, 30), dtype=np.int64)
    for p in ca.particles:                                  # the original per-particle loop
        loop[p[0], p[1]] += 1
    worst_cnt = max(worst_cnt, int(np.abs(loop - ca.grid.astype(np.int64)).max()))
    conserved &= int(ca.grid.sum()) == 30 * 30 // 2
report("non-interacting bincount rebuild vs per-particle loop", worst_cnt == 0 and conserved,
       f"max |diff| = {worst_cnt}, particles conserved = {conserved}, get_state() is the count grid: "
       f"{ca.get_state() is ca.grid}")

# ---- 5. swap counter ---------------------------------------------------------------------------
ca = impl.CoffeeAutomaton(40, "interacting", seed=7)
ok_swaps, ok_mass = True, True
for _ in range(300):
    before, s0 = ca.grid.copy(), ca.swaps
    ca.step(1)
    ok_swaps &= int((before != ca.grid).sum()) == 2 * (ca.swaps - s0)
    ok_mass &= int(ca.grid.sum()) == 40 * 40 // 2
report("interacting swap counter", ok_swaps and ok_mass and ca.paper_time == ca.swaps,
       f"300 batches, {ca.swaps} swaps ({ca.swaps / 300:.0f} per batch at n = 40, early mixing); "
       f"changed cells == 2 x swaps every batch: {ok_swaps}; particles conserved: {ok_mass}")

# ---- 6. byte conversion clips -----------------------------------------------------------------
b = impl.grid_to_bytes(np.array([[0, 1, 255, 256, 300]], dtype=np.uint16))
report("grid_to_bytes clips counts at 255", list(b) == [0, 1, 255, 255, 255], f"bytes {list(b)}")

print("RESULT:", "PASS" if all(results) else "FAIL")
sys.exit(0)
