"""
Exercise 4: Adjusted Coarse-Graining (Section 6)

The paper found that the 3-bucket thresholding (Section 5) produces artifacts:
border pixels fluctuate between bucket values, creating fake complexity
especially in the non-interacting model.

Section 6 fixes this with:
  1. More buckets (7 instead of 3)
  2. Row-majority adjustment: snap cells near the row majority value

Your task:
  1. Implement the row-majority adjustment algorithm
  2. Run both models and measure 3-bucket (no adjustment) and 7-bucket
     (with adjustment) complexity on the same states
  3. Check the paper's claim (Sec. 6.2): the adjustment flattens the
     non-interacting curve "to a lower value" but keeps the interacting
     curve's shape. Does it, at 50 x 50?
[PATCH 2026-10-18] Step 3 used to say "show that"; at this grid size the
claim does not reproduce (see the solution's printout), and explaining why is
the exercise. Measurements use ca.get_state() (particle counts for the
non-interacting model) and the paper's centred-window coarse-graining.

Paper reference: Section 6, Appendix
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import (
    CoffeeAutomaton, coarse_grain, threshold_array, gzip_size, grid_to_bytes,
    default_total_steps
)


def row_majority_adjust(thresholded: np.ndarray) -> np.ndarray:
    """
    TODO: Implement the row-majority adjustment from Section 6.1.

    For each row:
      1. Find the majority (most common) value
      2. For each cell: if it's within 1 bucket of the majority, set it to majority

    "If a cell is within one threshold value of the majority value in its row,
    it is adjusted to the majority value." (Section 6.1)

    Returns adjusted array (don't modify input).
    """
    # YOUR CODE HERE
    pass


def measure_with_method(grid, grain_size, num_buckets, use_adjustment):
    """
    TODO: Measure complexity using specified method.

    1. Coarse-grain the grid
    2. Threshold into num_buckets
    3. If use_adjustment, apply row_majority_adjust
    4. Compress and return size
    """
    # YOUR CODE HERE
    pass


if __name__ == '__main__':
    print("Adjusted Coarse-Graining Experiment (Section 6)")
    print("=" * 50)

    grid_size = 50
    num_snapshots = 50
    grain = 7

    peaks = {}
    for model in ['interacting', 'non_interacting']:
        print(f"\n{model.replace('_', '-').title()} model:")

        ca = CoffeeAutomaton(grid_size, model, seed=42)
        # [PATCH 2026-10-18] each model's own time unit (was 50,000 steps for both)
        steps_per = default_total_steps(grid_size, model) // num_snapshots

        cx_3bucket = []
        cx_7adjusted = []

        for i in range(num_snapshots):
            ca.step(steps_per)
            grid = ca.get_state()

            c3 = measure_with_method(grid, grain_size=grain, num_buckets=3, use_adjustment=False)
            c7 = measure_with_method(grid, grain_size=grain, num_buckets=7, use_adjustment=True)

            if c3 is not None:
                cx_3bucket.append(c3)
                cx_7adjusted.append(c7)

        if cx_3bucket:
            print(f"  3-bucket peak: {max(cx_3bucket)}")
            print(f"  7-bucket adjusted peak: {max(cx_7adjusted)} "
                  f"({100 * (1 - max(cx_7adjusted) / max(cx_3bucket)):.0f}% lower)")
            peaks[model] = max(cx_7adjusted)

    if len(peaks) == 2:
        print("\nPaper (Sec. 6.2): after adjustment the non-interacting curve is 'flattened to a"
              " lower value' while the interacting one keeps its hump.")
        print(f"Here: adjusted peaks interacting {peaks['interacting']}, non-interacting "
              f"{peaks['non_interacting']} bytes. Which one is lower, and why? (Appendix: the"
              " guarantee needs grains L >> G*sqrt(3 ln(2n^2)).)")
