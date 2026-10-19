"""
Exercise 3: Scaling Analysis

Replicate the scaling results from Figures 6-8 of the paper.

The paper found:
  - Max entropy ~ n^2 (quadratic in grid size)
  - Max complexity ~ n (linear in grid size)
  - Time to max complexity ~ n^2 (quadratic)

Your task:
  1. Run the interacting model at multiple grid sizes, measuring complexity
     with the Sec. 5 metric (centred-window coarse-graining, ONE grain size
     for every n, 3 buckets, no adjustment)
  2. Record max entropy, max complexity, and time to peak
  3. Fit curves and check the scaling exponents
  4. Plot your results

[PATCH 2026-10-18] With the old block averaging (and g = n // 10) the coarse
array was 10 x 10 at every n, so max complexity could not grow. Also compare
the time to peak in batches with ca.swaps (the paper's unit: one swap = one
step): a quadratic fit can look good even when the growth is closer to n^4.

Paper reference: Section 5.2, Figures 6-8
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import run_simulation


def scaling_experiment(sizes=None, steps_multiplier=4, grain_size=7):
    """
    TODO: Run simulations at multiple grid sizes and collect scaling data.

    For each grid size n:
      - Run total_steps = steps_multiplier * n^2 batches (so larger grids get more
        time; 4 n^2 covers the rise, the peak and the fall)
      - Measure with run_simulation(..., grain_size=grain_size, num_buckets=3,
        adjust=False) (Sec. 5 metric, same grain for every n)
      - Record max entropy, max complexity, time of max complexity

    Args:
        sizes: list of grid sizes to test (e.g., [10, 20, 30, 50])
        steps_multiplier: steps = multiplier * n^2

    Returns:
        sizes: array of grid sizes
        max_entropy: array of max entropy values
        max_complexity: array of max complexity values
        peak_time: array of times to peak complexity
    """
    if sizes is None:
        sizes = [20, 30, 40, 50]

    # YOUR CODE HERE
    pass


def fit_and_plot(sizes, max_entropy, max_complexity, peak_time):
    """
    TODO: Fit curves and plot results.

    1. Fit quadratic to max_entropy vs n  (expect ~ n^2)
    2. Fit linear to max_complexity vs n  (expect ~ n)
    3. Fit quadratic to peak_time vs n    (expect ~ n^2)
    4. Plot all three with fits and r^2 values
    """
    import matplotlib.pyplot as plt

    # YOUR CODE HERE
    # Hint: np.polyfit(sizes, values, degree) gives coefficients
    #       R^2 = 1 - SS_res / SS_tot
    pass


if __name__ == '__main__':
    print("Scaling experiment (Figures 6-8)")
    print("This may take a few minutes for larger grid sizes.")
    print("=" * 50)

    result = scaling_experiment(sizes=[20, 30, 40, 50])

    if result is not None:
        sizes, max_e, max_c, peak_t = result
        print(f"\n{'Size':>6} {'Max Entropy':>12} {'Max Complexity':>14} {'Peak Time':>10}")
        for i in range(len(sizes)):
            print(f"{sizes[i]:>6} {max_e[i]:>12} {max_c[i]:>14} {peak_t[i]:>10}")

        fit_and_plot(sizes, max_e, max_c, peak_t)
