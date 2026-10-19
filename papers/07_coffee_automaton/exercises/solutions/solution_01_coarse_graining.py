"""
Solution 1: Coarse-Graining and Complexity Measurement
"""

import sys
import os
import gzip
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import CoffeeAutomaton


def coarse_grain(grid: np.ndarray, grain_size: int) -> np.ndarray:
    """
    Section 5.1: replace each cell by the average of the g x g square centred
    on it; the output is N x N. Windows that stick out of the grid are averaged
    over their in-bounds cells (paper Fig. 1).

    [PATCH 2026-10-18] Was a non-overlapping block average ((N//g) x (N//g)
    output), which is not the paper's method. implementation.coarse_grain does
    the same as this loop with 2-D prefix sums (four lookups per window).
    """
    n0, n1 = grid.shape
    h = grain_size // 2                     # half-width: rows i-h .. i+h for odd g
    coarse = np.zeros((n0, n1), dtype=np.float64)
    for i in range(n0):
        for j in range(n1):
            # Slicing clips at the grid edge for us: max(0, .) on the left; Python
            # slices simply stop at the end on the right.
            window = grid[max(0, i - h):i + h + 1, max(0, j - h):j + h + 1]
            coarse[i, j] = window.mean()
    return coarse


def threshold(coarse: np.ndarray, num_buckets: int = 3) -> np.ndarray:
    """
    Threshold into discrete buckets.
    Section 5.1: 3 buckets (mostly coffee, mixed, mostly cream).
    """
    scaled = coarse * (num_buckets - 1)
    return np.clip(np.round(scaled), 0, num_buckets - 1).astype(np.uint8)


def gzip_complexity(grid: np.ndarray, grain_size: int = 7,
                    num_buckets: int = 3) -> int:
    """Apparent complexity: gzip(coarse-grained state)."""
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold(coarse, num_buckets)
    return len(gzip.compress(discrete.tobytes(), compresslevel=9))


def gzip_entropy(grid: np.ndarray) -> int:
    """Entropy estimate: gzip(fine-grained state)."""
    return len(gzip.compress(grid.astype(np.uint8).tobytes(), compresslevel=9))


if __name__ == '__main__':
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
        print(f"{t:>5} | {e:>7} | {c:>10}")

    if measured[1] > 2 * max(measured[0], measured[2]):
        print("\nComplexity peaks at intermediate t: your implementation is working.")
    else:
        print("\nNo clear peak at t=2500 yet: check coarse_grain (N x N output, centred window).")

    # For comparison: the old block average gives a 7 x 7 array at N = 50, g = 7.
    from implementation import coarse_grain as library_coarse_grain
    print(f"\nOutput shapes: centred window {coarse_grain(grid, 7).shape}, "
          f"blocks {library_coarse_grain(grid, 7, method='block').shape}; "
          f"window matches implementation.coarse_grain: "
          f"{np.allclose(coarse_grain(grid, 7), library_coarse_grain(grid, 7))}")
