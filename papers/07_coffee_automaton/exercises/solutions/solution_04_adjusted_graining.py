"""
Solution 4: Adjusted Coarse-Graining (Section 6)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import (
    CoffeeAutomaton, coarse_grain, threshold_array, gzip_size, grid_to_bytes,
    default_total_steps
)


def row_majority_adjust(thresholded: np.ndarray) -> np.ndarray:
    """
    Row-majority adjustment from Section 6.1.
    Snap cells within 1 bucket of the row majority value.
    """
    adjusted = thresholded.copy()
    for i in range(adjusted.shape[0]):
        row = adjusted[i]
        values, counts = np.unique(row, return_counts=True)
        majority = values[np.argmax(counts)]
        mask = np.abs(row.astype(np.int16) - int(majority)) <= 1
        adjusted[i, mask] = majority
    return adjusted


def measure_with_method(grid, grain_size, num_buckets, use_adjustment):
    """Measure complexity with specified coarse-graining method."""
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold_array(coarse, num_buckets)
    if use_adjustment:
        discrete = row_majority_adjust(discrete)
    return gzip_size(grid_to_bytes(discrete))


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
        # [PATCH 2026-10-18] each model's own time unit (was 50,000 steps for both, which
        # put the non-interacting peak, at ~200-400 steps, before the first snapshot)
        steps_per = default_total_steps(grid_size, model) // num_snapshots

        cx_3bucket = []
        cx_7adjusted = []

        for i in range(num_snapshots):
            ca.step(steps_per)
            grid = ca.get_state()   # [PATCH 2026-10-18] counts for the non-interacting model

            c3 = measure_with_method(grid, grain_size=grain, num_buckets=3, use_adjustment=False)
            c7 = measure_with_method(grid, grain_size=grain, num_buckets=7, use_adjustment=True)

            cx_3bucket.append(c3)
            cx_7adjusted.append(c7)

        print(f"  3-bucket peak: {max(cx_3bucket)}")
        print(f"  7-bucket adjusted peak: {max(cx_7adjusted)} "
              f"({100 * (1 - max(cx_7adjusted) / max(cx_3bucket)):.0f}% lower)")
        peaks[model] = max(cx_7adjusted)

    print("\nPaper (Sec. 6.2): after adjustment the non-interacting curve is 'flattened to a"
          " lower value' while the interacting one keeps its hump.")
    print(f"Here: adjusted peaks interacting {peaks['interacting']}, non-interacting "
          f"{peaks['non_interacting']} bytes.")
    if peaks['non_interacting'] >= peaks['interacting']:
        print("Not reproduced at 50 x 50: the adjustment lowers both models' peaks, and the"
              " non-interacting one stays higher.\nThe Appendix guarantee needs grains"
              " L >> G*sqrt(3 ln(2n^2)) ~ 35 cells (G = 7, n = 50), and particle counts"
              " fluctuate more than 0/1 cells,\nso the count image keeps flickering across"
              " more than one bucket, which the +-1 adjustment cannot remove.")
    else:
        print("Reproduced in direction: the non-interacting adjusted peak is the lower one.")
