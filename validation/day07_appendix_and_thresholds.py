"""Day 07 checks against the paper's text (arXiv:1405.6903).

(a) Appendix, Claim 1 recursion for the NON-interacting model with periodic boundaries:
      E[a_{t+1}(x,y)] = (E[a_t(x-1,y)] + E[a_t(x+1,y)] + E[a_t(x,y-1)] + E[a_t(x,y+1)]) / 4.
    Starting from the top-half-full state this is a 1-D (row-only) diffusion.  We iterate the
    exact recursion and compare with the repo simulation's mean row density (Monte Carlo).
(b) Sec. 5.1 thresholding: 3 buckets = mostly coffee (~0), mixed (~0.5), mostly cream (~1).
(c) Sec. 6.1 adjustment: "If a cell is within one threshold value of the majority value in its
    row, it is adjusted to the majority value."
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
sys.path.insert(0, os.path.join(REPO, "papers", "07_coffee_automaton"))
import implementation as impl

# ---- (a) exact expectation vs simulation ---------------------------------------------------
n, T, seeds = 30, 40, 12
E = np.zeros((n, n)); E[: n // 2, :] = 1.0          # E[a_0]: one cream particle per top-half cell
for _ in range(T):
    # np.roll(E, s, axis) shifts with wrap-around = periodic boundary; average of the 4 neighbours
    E = (np.roll(E, 1, 0) + np.roll(E, -1, 0) + np.roll(E, 1, 1) + np.roll(E, -1, 1)) / 4.0
exact_rows = E.mean(axis=1)                          # mean over columns (axis 1) -> shape (n,)

sim = np.zeros(n)
for s in range(seeds):
    ca = impl.CoffeeAutomaton(n, "non_interacting", seed=s)
    ca.step(T)
    sim += ca.grid.mean(axis=1)                      # particle COUNTS averaged over each row
sim /= seeds
err = np.abs(sim - exact_rows).max()
# Monte Carlo scale: each row mean averages n*seeds cells with per-cell variance <= E (Poisson-like)
mc = np.sqrt(exact_rows.max() / (n * seeds))
print(f"(a) appendix recursion vs simulation: max |delta| over rows = {err:.4f} (MC std ~ {mc:.4f})")
print(f"    total particles conserved: {int(ca.grid.sum())} == {n * n // 2}")

# ---- (b) thresholds ----------------------------------------------------------------------
x = np.array([[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]])
print("(b) 3-bucket threshold of", x.ravel().tolist(), "->", impl.threshold_array(x, 3).ravel().tolist())
print("    7-bucket threshold ->", impl.threshold_array(x, 7).ravel().tolist())

# ---- (c) row-majority adjustment ---------------------------------------------------------
row = np.array([[3, 3, 3, 2, 4, 1, 5, 3]], dtype=np.uint8)
adj = impl.row_majority_adjust(row)
print("(c) row", row.ravel().tolist(), "-> adjusted", adj.ravel().tolist(),
      "(cells within 1 of majority 3 snap to 3; 1 and 5 kept)")
ok = err < 5 * mc and adj.ravel().tolist() == [3, 3, 3, 3, 3, 1, 5, 3]
print("RESULT:", "PASS" if ok else "FAIL")
