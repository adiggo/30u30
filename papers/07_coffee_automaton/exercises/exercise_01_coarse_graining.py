"""
Exercise 1: Coarse-Graining and Complexity Measurement

Build the paper's core measurement tool from scratch.

The paper (Section 5.1) measures "apparent complexity" by:
  1. Replacing every cell by the average of the g x g square CENTRED on it
     (the output has the same N x N shape as the grid; Fig. 1)
  2. Thresholding the averages into discrete buckets
  3. Compressing with gzip
  4. The compressed size = complexity estimate
[PATCH 2026-10-18] Step 1 used to say "divide the grid into g x g blocks and
average each block"; that is a different (downsampling) method, not Sec. 5.1.

Your task: implement coarse_grain() and measure both entropy and complexity
on a coffee automaton grid at different time steps.

Paper reference: Sections 4 and 5
"""

import sys
import os
import gzip
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import CoffeeAutomaton


def coarse_grain(grid: np.ndarray, grain_size: int) -> np.ndarray:
    """
    TODO: Implement coarse-graining.

    Given an N x N binary grid and a grain size g (odd, e.g. 7), produce an
    N x N array where cell (i, j) is the average of the g x g window centred
    on (i, j). Near the edges, average only the window cells that lie inside
    the grid (the paper's Fig. 1: a corner cell averages a 2 x 2 window when
    g = 3).

    Returns float array with values in [0, 1].
    """
    # YOUR CODE HERE
    pass


def threshold(coarse: np.ndarray, num_buckets: int = 3) -> np.ndarray:
    """
    TODO: Threshold floating-point values into discrete buckets.

    Map [0, 1] values to integers in [0, num_buckets-1].
    For 3 buckets: 0 = mostly coffee, 1 = mixed, 2 = mostly cream.

    Returns uint8 array.
    """
    # YOUR CODE HERE
    pass


def gzip_complexity(grid: np.ndarray, grain_size: int = 7,
                    num_buckets: int = 3) -> int:
    """
    TODO: Measure apparent complexity.

    1. Coarse-grain the grid
    2. Threshold into buckets
    3. Compress with gzip
    4. Return compressed size in bytes
    """
    # YOUR CODE HERE
    pass


def gzip_entropy(grid: np.ndarray) -> int:
    """
    TODO: Measure entropy estimate.

    Compress the fine-grained grid directly with gzip.
    Return compressed size in bytes.
    """
    # YOUR CODE HERE
    pass


if __name__ == '__main__':
    # Test your implementation:
    # Create a 50x50 automaton and measure at t = 0, 2500 and 10000 steps.
    # [PATCH 2026-10-18] One step here is a batch of swaps; the 3-bucket complexity
    # of a 50x50 cup peaks near t = 2,500 and the cup is mixed by t = 10,000 (the
    # old times, 10,000 and 50,000, were both past the peak).
    ca = CoffeeAutomaton(grid_size=50, model='interacting', seed=42)

    print("Time  | Entropy | Complexity")
    print("------|---------|----------")

    measured = []
    for t in [0, 2500, 10000]:
        if t > 0:
            ca.step(t - ca.time)
        grid = ca.get_binary_grid()
        e = gzip_entropy(grid)
        c = gzip_complexity(grid, grain_size=7, num_buckets=3)
        measured.append(c)
        print(f"{t:>5} | {e} | {c}")

    # Expected behavior:
    # - Entropy should increase over time
    # - Complexity should be low at t=0, much higher at t=2500, low again at t=10000
    if None not in measured and measured[1] > 2 * max(measured[0], measured[2]):
        print("\nComplexity peaks at intermediate t: your implementation is working.")
    else:
        print("\nNo clear peak at t=2500 yet: check coarse_grain (N x N output, centred window).")
