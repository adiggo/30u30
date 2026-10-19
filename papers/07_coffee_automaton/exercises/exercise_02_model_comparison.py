"""
Exercise 2: Interacting vs Non-Interacting Model Comparison

What the paper reports: with 3 buckets (Sec. 5.2) BOTH models' complexity
rises and falls; with the adjusted 7-bucket method (Sec. 6.2) the interacting
curve keeps its shape while the non-interacting one is "flattened to a lower
value". The Appendix proves that the non-interacting model's coarse-grained
state stays simple, but only for grains L >> G*sqrt(3 ln(2n^2)) (~35 cells for
n = 50 and G = 7 grey levels).

Your task:
  1. Run both models on a 50x50 grid
  2. Measure complexity (CoffeeAutomaton.complexity(): centred 7 x 7 window,
     7 buckets + row-majority adjustment) at regular intervals
  3. Plot complexity over time for both models
  4. Compare the peaks with the paper's claim, and explain what you see
[PATCH 2026-10-18] Step 4 used to promise "non-interacting stays flat"; at this
grid size it does not (see the solution's printout and the README).

Note the time units differ: one interacting step is a batch of swaps, one
non-interacting step moves every particle, so give each model its own run
length (implementation.default_total_steps).

Paper reference: Sections 3, 5.2, 6.2, Appendix
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import (CoffeeAutomaton, measure_complexity, measure_entropy,
                            default_total_steps)


def run_and_measure(model: str, grid_size: int = 50,
                    total_steps=None, num_snapshots: int = 50):
    """
    TODO: Run a coffee automaton simulation and collect measurements.

    Args:
        model: 'interacting' or 'non_interacting'
        grid_size: N
        total_steps: how many steps total (None: default_total_steps(grid_size, model))
        num_snapshots: how many measurement points

    Returns:
        times: list of time steps
        complexities: list of complexity measurements
        entropies: list of entropy measurements
    """
    # YOUR CODE HERE
    # Hint:
    #   1. Create a CoffeeAutomaton with the given model
    #   2. Calculate steps_per_snapshot = total_steps // num_snapshots
    #   3. Loop: step(), then measure complexity and entropy
    #   4. Return the collected data
    pass


def plot_comparison(times_inter, cx_inter, times_non, cx_non):
    """
    TODO: Plot complexity over time for both models, one panel each (their
    time axes have different units).
    """
    import matplotlib.pyplot as plt

    # YOUR CODE HERE
    # Hint: fig, (ax1, ax2) = plt.subplots(1, 2)
    #       ax1.plot(times_inter, cx_inter); ax2.plot(times_non, cx_non)
    pass


if __name__ == '__main__':
    print("Running interacting model...")
    res_i = run_and_measure('interacting')

    print("Running non-interacting model...")
    res_n = run_and_measure('non_interacting')

    # [PATCH 2026-10-18] the unfilled stub returns None; say so instead of crashing on unpacking
    if res_i is None or res_n is None:
        print("\nrun_and_measure() is not implemented yet (it returns None).")
        cx_i = cx_n = None
    else:
        t_i, cx_i, e_i = res_i
        t_n, cx_n, e_n = res_n

    # Print summary
    if cx_i is not None and cx_n is not None:
        print(f"\nInteracting:     start {cx_i[0]}, peak {max(cx_i)}, end {cx_i[-1]} bytes")
        print(f"Non-interacting: start {cx_n[0]}, peak {max(cx_n)}, end {cx_n[-1]} bytes")
        print(f"Peak ratio non-interacting / interacting: {max(cx_n) / max(cx_i):.1f}x")
        print("\nPaper (Sec. 6.2): with this adjusted metric the non-interacting curve is"
              " 'flattened to a lower value'.")
        print("Is it here? If not, why might a 7 x 7 grain on a 50 x 50 cup be too small for the"
              " Appendix argument, and why do particle counts fluctuate more than 0/1 cells?")

        plot_comparison(t_i, cx_i, t_n, cx_n)
