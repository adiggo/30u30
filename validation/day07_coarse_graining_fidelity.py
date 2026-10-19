"""Day 07 check: does the repo's coarse-graining reproduce the paper's results?

Paper (arXiv:1405.6903, Sec. 5.1): "we construct a new array in which the value of each cell
is the average of the values of the nearby cells in the fine-grained array ... within a g x g
square centered at the cell in question"  -> a sliding (box-filter) average, SAME size as the
grid.  Then threshold into 3 buckets (Sec. 5) or 7 buckets + row-majority adjustment (Sec. 6),
and gzip the thresholded array.

The repo's implementation.coarse_grain instead averages NON-overlapping g x g blocks, giving
an (n//g) x (n//g) array (10 x 10 for the default g = n//10, whatever n is).
[PATCH 2026-10-18] implementation.coarse_grain now defaults to the paper's window; "repo" below
means its earlier block method, coarse_grain(..., method="block"). See day07_paper_experiments.py
for the re-run with the window method in the library.

Compared here (interacting dynamics = patched, particle-conserving code):
  1. hump shape for the interacting model,
  2. interacting vs non-interacting contrast under the adjusted (7-bucket) method (Sec. 6.2),
  3. scaling of max complexity with n (paper Fig. 7: ~linear in n).
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
sys.path.insert(0, os.path.join(REPO, "papers", "07_coffee_automaton"))
import implementation as impl  # patched repo code

def box_filter_centered(grid, g):
    """Paper Sec. 5.1: mean over the g x g window centred on each cell (g odd), clipped at
    the borders (average over in-bounds cells only).  Output has the same shape as grid."""
    a = grid.astype(np.float64)                     # (n, n)
    h = g // 2
    # 2-D prefix sums with a zero row/col in front: S[i, j] = sum(a[:i, :j]); shape (n+1, n+1)
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    S[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)     # cumsum down rows (axis 0), then across cols (axis 1)
    n0, n1 = a.shape
    r0 = np.clip(np.arange(n0) - h, 0, n0)          # window top row for each cell row, shape (n0,)
    r1 = np.clip(np.arange(n0) + h + 1, 0, n0)      # window bottom row (exclusive)
    c0 = np.clip(np.arange(n1) - h, 0, n1)
    c1 = np.clip(np.arange(n1) + h + 1, 0, n1)
    # Broadcasting: r*[:, None] has shape (n0, 1), c*[None, :] has shape (1, n1) -> (n0, n1)
    tot = S[r1[:, None], c1[None, :]] - S[r0[:, None], c1[None, :]] \
        - S[r1[:, None], c0[None, :]] + S[r0[:, None], c0[None, :]]
    cnt = (r1 - r0)[:, None] * (c1 - c0)[None, :]  # number of in-bounds cells per window
    return tot / cnt

def complexity(grid, method, g, buckets, adjust):
    coarse = impl.coarse_grain(grid, g, method="block") if method == "repo" else box_filter_centered(grid, g)
    d = impl.threshold_array(np.clip(coarse, 0, 1), buckets)
    if adjust:
        d = impl.row_majority_adjust(d)
    return impl.gzip_size(impl.grid_to_bytes(d))

def verdict(c):
    """Genuine rise-and-fall: interior peak and final value back at least halfway to the start."""
    c = np.asarray(c, dtype=float)
    k = int(np.argmax(c))
    hump = 0 < k < len(c) - 1 and c[-1] <= c[0] + 0.5 * (c[k] - c[0])
    return f"init={c[0]:.0f} peak={c[k]:.0f}@{k}/{len(c)-1} final={c[-1]:.0f} -> {'HUMP' if hump else 'no hump'}"

def run_interacting(n, batches, snaps, seed=42):
    ca = impl.CoffeeAutomaton(n, "interacting", seed=seed)
    out = [ca.grid.copy()]
    for _ in range(snaps):
        ca.step(batches // snaps)
        out.append(ca.grid.copy())
    return out

def run_noninteracting(n, steps, snaps, seed=42):
    ca = impl.CoffeeAutomaton(n, "non_interacting", seed=seed)
    counts, occ = [ca.grid.copy()], [ca.get_binary_grid().copy()]
    for _ in range(snaps):
        ca.step(steps // snaps)
        counts.append(ca.grid.copy()); occ.append(ca.get_binary_grid().copy())
    return counts, occ

if __name__ == "__main__":
    n = 50
    print("== 1/2. n=50, interacting (100k batches) and non-interacting (5000 steps), 40 snapshots")
    gi = run_interacting(n, 100_000, 40)
    cnt, occ = run_noninteracting(n, 5_000, 40)
    for method, g in (("repo", 5), ("paper", 5), ("paper", 9)):
        for buckets, adjust in ((3, False), (7, True)):
            ci = [complexity(x, method, g, buckets, adjust) for x in gi]
            src = occ if method == "repo" else cnt     # repo measures occupancy; paper averages counts
            cn = [complexity(x, method, g, buckets, adjust) for x in src]
            print(f"{method:5s} g={g} {buckets}b{'+adj' if adjust else '    '} | interacting: {verdict(ci)}"
                  f" | non-interacting: {verdict(cn)} | peak ratio int/non = {max(ci)/max(cn):.2f}")
    print("== 3. scaling of max complexity (interacting, 40*n^2 batches, 7 buckets + adjust, g = n//10 for repo, g=5 paper)")
    rows = []
    for m in (20, 30, 40, 50):
        gs = run_interacting(m, 40 * m * m, 40)
        mr = max(complexity(x, "repo", max(2, m // 10), 7, True) for x in gs)
        mp = max(complexity(x, "paper", 5, 7, True) for x in gs)
        rows.append((m, mr, mp)); print(f"  n={m}: max complexity repo={mr}  paper-method={mp}")
    arr = np.array(rows, dtype=float)
    for j, name in ((1, "repo"), (2, "paper")):
        coef = np.polyfit(arr[:, 0], arr[:, j], 1)                      # least-squares line
        pred = np.polyval(coef, arr[:, 0])
        r2 = 1 - ((arr[:, j] - pred) ** 2).sum() / ((arr[:, j] - arr[:, j].mean()) ** 2).sum()
        print(f"  {name}: slope={coef[0]:.2f} bytes per unit n, linear r^2={r2:.3f}")
