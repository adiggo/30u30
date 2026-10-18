"""Day 06 checks (Aaronson 2011 blog post; measurement conventions from Day 07, arXiv:1405.6903).

1. batch_swap picks a uniformly random EDGE: enumerate all (cell, direction) draws and count
   how many map to each undirected edge (must be exactly 2 for every edge -> P = 1/(2N)).
   So, conditioned on a swap happening, it is uniform over differing adjacent pairs = the
   blog's rule "pick an adjacent coffee pixel and milk pixel uniformly at random, and swap".
   Also check particle conservation over a run.
2. coarse_grain == NumPy reshape-mean reference (max |delta|).
3. grid_entropy == binary entropy closed form; boundary_fraction == brute-force count.
4. The "hump": run a 32x32 grid to equilibrium and compare raw gzip (fine grid; the Day-07
   paper calls this the ENTROPY estimate) with Day-06's coarse_grained_kc and with a
   paper-style thresholded coarse-grained measure. Verdict rule: interior peak AND the final
   value has fallen at least halfway back to the initial value.
5. Solution 02's mixing assertion (0.3 < top fraction < 0.7) at the default and larger n_steps.
"""
import gzip
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "06_Complexodynamics")
sys.path.insert(0, DAY)
import implementation as impl

def verdict(c):
    c = np.asarray(c, dtype=float); k = int(np.argmax(c))
    hump = 0 < k < len(c) - 1 and c[-1] <= c[0] + 0.5 * (c[k] - c[0])
    return f"init={c[0]:.0f} peak={c[k]:.0f} (snapshot {k}/{len(c)-1}) final={c[-1]:.0f} -> {'HUMP' if hump else 'no hump'}"

ok = True
# ---- 1. edge-selection uniformity by exact enumeration ----------------------------------
R = C = 7
hits = {}
for r in range(R):
    for c in range(C):
        for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):   # same order as batch_swap
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                e = tuple(sorted(((r, c), (nr, nc))))
                hits[e] = hits.get(e, 0) + 1
n_edges = R * (C - 1) + (R - 1) * C
uniform = set(hits.values()) == {2} and len(hits) == n_edges
print(f"1. {len(hits)}/{n_edges} edges, draws per edge = {sorted(set(hits.values()))} -> uniform: {uniform}")
np.random.seed(0)
g = impl.create_initial_grid(32); s0 = int(g.sum())
impl.batch_swap(g, 200_000)
print(f"   conservation after 200k attempted swaps: {s0} -> {int(g.sum())}")
ok &= uniform and int(g.sum()) == s0

# ---- 2. coarse_grain vs reshape-mean ------------------------------------------------------
rng = np.random.default_rng(1)
x = rng.integers(0, 2, size=(32, 32)).astype(np.uint8)
for b in (2, 4, 8):
    # reshape (32,32) -> (32/b, b, 32/b, b): axis 1 and 3 index the position inside a block;
    # mean over axes (1, 3) gives the block average, shape (32/b, 32/b)
    ref = x.reshape(32 // b, b, 32 // b, b).mean(axis=(1, 3))
    d = np.abs(impl.coarse_grain(x, b) - ref).max()
    print(f"2. coarse_grain block={b}: max |delta| vs reshape-mean = {d:.1e}"); ok &= d == 0

# ---- 3. closed forms -----------------------------------------------------------------------
p = x.mean(); H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
bf = sum(x[i, j] != x[i, j + 1] for i in range(32) for j in range(31)) + \
     sum(x[i, j] != x[i + 1, j] for i in range(31) for j in range(32))
bf /= (32 * 31 * 2)
print(f"3. grid_entropy |delta| = {abs(impl.grid_entropy(x) - H):.1e}; boundary_fraction |delta| = {abs(impl.boundary_fraction(x) - bf):.1e}")
ok &= abs(impl.grid_entropy(x) - H) < 1e-12 and abs(impl.boundary_fraction(x) - bf) < 1e-12

# ---- 4. hump or no hump ----------------------------------------------------------------
def thresholded_kc(grid, b=4, buckets=3):
    """Day-07 style: block-average, round to `buckets` levels (0 / mixed / 1), gzip."""
    q = np.round(impl.coarse_grain(grid, b) * (buckets - 1)).astype(np.uint8)
    return len(gzip.compress(q.tobytes(), compresslevel=9))

np.random.seed(42)
g = impl.create_initial_grid(32)
raw, ckc, thr = [], [], []
for k in range(61):                      # 60 x 50k = 3M attempted swaps (~equilibrium for 32x32)
    if k: impl.batch_swap(g, 50_000)
    raw.append(impl.gzip_complexity(g)); ckc.append(impl.coarse_grained_kc(g, 4)); thr.append(thresholded_kc(g))
print("4. raw gzip (fine grid)       :", verdict(raw))
print("   Day-06 coarse_grained_kc(4) :", verdict(ckc))
print("   thresholded 3-level, block 4:", verdict(thr))
print("   summarize_results on a full train_minimal.py run's numbers (39 -> 681 -> 672, peak at 97% of run):")
res = {"times": list(range(101)), "gzip_complexity": [39] + [600] * 96 + [681, 675, 672, 672],
       "entropy": [1.0] * 101, "grid_size": 64, "n_steps": 50000, "swaps_per_step": 50}
print("  ", [l.strip() for l in impl.summarize_results(res).splitlines() if "[" in l])

# ---- 5. solution 02 assertion -------------------------------------------------------------
sys.path.insert(0, os.path.join(DAY, "exercises", "solutions"))
import solution_02_coffee_mixing as s2
for steps in (5000, 20000):
    snaps = s2.run_and_snapshot(n_steps=steps)
    top = snaps[-1][1][:16, :].mean()
    print(f"5. solution_02 n_steps={steps}: top-half coffee fraction = {top:.3f} -> assertion 0.3<f<0.7 {'passes' if 0.3 < top < 0.7 else 'FAILS'}")
print("RESULT:", "PASS" if ok else "FAIL", "(items 1-3: implementation math)")
