"""Day 07: re-run the paper's three experiments with the paper's coarse-graining (2026-10-18).

Coarse-graining is implementation.coarse_grain(method='window'): the g x g average centred on each
cell (Sec. 5.1), compared with the repo's earlier non-overlapping block average (method='block').
"S5" = 3 buckets, no adjustment (Sec. 5); "S6" = 7 buckets + row-majority adjustment (Sec. 6.1).

  1. Rise and fall for the interacting model, n = 50, 3 seeds (Sec. 5.2 / 6.2, Figs. 2 and 10).
  2. Interacting vs non-interacting, n = 50, 3 seeds, g = 7 and 9 (Sec. 6.2: after adjustment the
     non-interacting curve is "flattened to a lower value"). The non-interacting model is measured on
     particle counts (Sec. 3.2, Appendix) and, for reference, on occupancy (the repo's earlier choice).
  3. Scaling of max complexity with n (Fig. 7) and of the time to the peak (Fig. 8), n = 20..60,
     2 seeds, window vs block at the same grain.
A curve "rises and falls" if its peak is interior and its final value is at least halfway back down.
Informational (prints the numbers, no pass/fail); exits 0. ~3 min on one core.
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
sys.path.insert(0, os.path.join(REPO, "papers", "07_coffee_automaton"))
import implementation as impl  # noqa: E402


def hump(v):
    v = np.asarray(v, dtype=float)
    k = int(np.argmax(v))
    return 0 < k < len(v) - 1 and v[-1] <= v[0] + 0.5 * (v[k] - v[0])


def run(n, model, seed, measures, snaps=100):
    """Simulate once (default_total_steps) and apply every measure (name -> f(state)) at each snapshot."""
    ca = impl.CoffeeAutomaton(n, model, seed)
    per = max(1, impl.default_total_steps(n, model) // snaps)
    series = {name: [] for name in measures}
    for i in range(snaps + 1):
        if i:
            ca.step(per)
        for name, f in measures.items():
            series[name].append(f(ca))
    return {k: np.array(v, dtype=float) for k, v in series.items()}


def summarize(label, runs):
    """runs: list of 1-D series (one per seed)."""
    a = np.array([[r[0], r.max(), r[-1]] for r in runs])
    print(f"    {label:34s} start {a[:, 0].mean():5.0f}  peak {a[:, 1].mean():5.0f} +- {a[:, 1].std():3.0f}"
          f"  final {a[:, 2].mean():5.0f}   rises and falls: {sum(hump(r) for r in runs)}/{len(runs)}")
    return a[:, 1].mean()


def cx(g, buckets, adjust, method="window", occupancy=False):
    def f(ca):
        state = ca.get_binary_grid() if occupancy else ca.get_state()
        return impl.measure_complexity(state, g, buckets, adjust, method)
    return f


SEEDS = (0, 1, 2)
n = 50

print("== 1. interacting model, n = 50, 4 n^2 = 10,000 batches, 3 seeds")
meas = {
    "window g=7 S5": cx(7, 3, False), "window g=7 S6": cx(7, 7, True),
    "window g=5 S5": cx(5, 3, False), "window g=5 S6": cx(5, 7, True),
    "block g=5 (n//10) S5": cx(5, 3, False, "block"), "block g=5 (n//10) S6": cx(5, 7, True, "block"),
}
runs = [run(n, "interacting", s, meas) for s in SEEDS]
for name in meas:
    summarize(name, [r[name] for r in runs])

print("\n== 2. interacting vs non-interacting, n = 50 (non-interacting: n^2/2 = 1,250 steps), 3 seeds")
for g in (7, 9):
    mi = {"S5": cx(g, 3, False), "S6": cx(g, 7, True)}
    mn = {"S5": cx(g, 3, False), "S6": cx(g, 7, True),
          "S5 occ": cx(g, 3, False, occupancy=True), "S6 occ": cx(g, 7, True, occupancy=True)}
    ri = [run(n, "interacting", s, mi) for s in SEEDS]
    rn = [run(n, "non_interacting", s, mn) for s in SEEDS]
    print(f"  g = {g}")
    pi5 = summarize("interacting S5", [r["S5"] for r in ri])
    pn5 = summarize("non-interacting S5 (counts)", [r["S5"] for r in rn])
    pi6 = summarize("interacting S6", [r["S6"] for r in ri])
    pn6 = summarize("non-interacting S6 (counts)", [r["S6"] for r in rn])
    po6 = summarize("non-interacting S6 (occupancy)", [r["S6 occ"] for r in rn])
    print(f"    peak ratio non-interacting / interacting: S5 {pn5 / pi5:.2f}, S6 {pn6 / pi6:.2f} "
          f"(occupancy {po6 / pi6:.2f}); paper Sec. 6.2: S6 non-interacting 'flattened to a lower value'")

print("\n== 3. scaling, interacting, n = 20..60, grain 7 for every n, 2 seeds")
for method in ("window", "block"):
    res = impl.scaling_experiment(sizes=[20, 30, 40, 50, 60], seed=42, grain_size=7,
                                  method=method, seeds=2, verbose=False)
    c5, r5 = impl.fit_r2(res["sizes"], res["max_complexity"], 1)
    c6, r6 = impl.fit_r2(res["sizes"], res["max_complexity_adjusted"], 1)
    _, rt = impl.fit_r2(res["sizes"], res["peak_swaps"], 2)
    if np.all(res["peak_swaps"] > 0):
        # slope of log(time) vs log(n) = exponent of a power law
        p_s = f"{np.polyfit(np.log(res['sizes']), np.log(res['peak_swaps']), 1)[0]:.2f}"
    else:
        p_s = "n/a (a run peaked at t = 0)"
    print(f"  {method:6s} max cx S5: {np.round(res['max_complexity'], 1).tolist()} -> slope {c5[0]:+.2f}/n, "
          f"linear r^2 {r5:.3f} (paper Fig. 7: 0.980)")
    print(f"  {method:6s} max cx S6: {np.round(res['max_complexity_adjusted'], 1).tolist()} -> slope {c6[0]:+.2f}/n, "
          f"linear r^2 {r6:.3f}")
    print(f"  {method:6s} time to S5 peak (swaps): {[f'{t:.3g}' for t in res['peak_swaps']]}: quadratic fit "
          f"r^2 {rt:.3f} (paper Fig. 8: 0.988), power-law exponent {p_s}")
print("  (block with g = n // 10, the repo's earlier default, is 10 x 10 at every n: see "
      "day07_coarse_graining_fidelity.py)")
sys.exit(0)
