"""
Solution 2: Interacting vs Non-Interacting Model Comparison
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import (CoffeeAutomaton, measure_complexity, measure_entropy,
                            default_total_steps)


def run_and_measure(model: str, grid_size: int = 50,
                    total_steps=None, num_snapshots: int = 50):
    """Run a simulation and collect measurements.

    [PATCH 2026-10-18] Each model gets a run length matched to its own time unit
    (was 50,000 steps for both: ~5x the interacting model's mixing time, and for the
    non-interacting model, where every particle moves every step, ~50x).
    ca.complexity() / ca.entropy() measure the paper's state (particle counts for the
    non-interacting model) with the centred-window coarse-graining.
    """
    if total_steps is None:
        total_steps = default_total_steps(grid_size, model)
    ca = CoffeeAutomaton(grid_size, model, seed=42)
    steps_per = max(1, total_steps // num_snapshots)

    times = [0]
    complexities = [ca.complexity()]
    entropies = [ca.entropy()]

    for i in range(num_snapshots):
        ca.step(steps_per)
        times.append(ca.time)
        complexities.append(ca.complexity())
        entropies.append(ca.entropy())

    return times, complexities, entropies


def plot_comparison(times_inter, cx_inter, times_non, cx_non):
    """Plot complexity for both models (separate panels: different time units)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    ax1.plot(times_inter, cx_inter, 'r-', linewidth=2)
    ax1.set_xlabel('Time (batches of swaps)')
    ax1.set_ylabel('Apparent complexity (gzip bytes)')
    ax1.set_title('Interacting')
    ax2.plot(times_non, cx_non, 'b-', linewidth=2)
    ax2.set_xlabel('Time (steps: every particle moves)')
    ax2.set_title('Non-interacting')
    plt.suptitle('7 x 7 centred window, 7 buckets + row-majority adjustment (Sec. 6)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Running interacting model...")
    t_i, cx_i, e_i = run_and_measure('interacting')

    print("Running non-interacting model...")
    t_n, cx_n, e_n = run_and_measure('non_interacting')

    print(f"\nInteracting:     start {cx_i[0]}, peak {max(cx_i)}, end {cx_i[-1]} bytes")
    print(f"Non-interacting: start {cx_n[0]}, peak {max(cx_n)}, end {cx_n[-1]} bytes")
    print(f"Peak ratio non-interacting / interacting: {max(cx_n) / max(cx_i):.1f}x")
    print("\nPaper (Sec. 6.2): with this adjusted metric the non-interacting curve is"
          " 'flattened to a lower value'.")
    if max(cx_n) >= max(cx_i):
        print("Not reproduced at 50 x 50: the non-interacting peak is the higher one. Both curves"
              " rise and fall.\nThe Appendix guarantee needs grains L >> G*sqrt(3 ln(2n^2)) ~ 35"
              " cells here (G = 7), far above g = 7,\nand particle counts fluctuate more than"
              " 0/1 cells (variance ~ rho vs rho(1 - rho)), so the count image flickers more.")
    else:
        print("Reproduced in direction: the non-interacting peak is lower.")

    plot_comparison(t_i, cx_i, t_n, cx_n)
