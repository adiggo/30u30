"""
Solution 5: Compressor Comparison (Figure 5)
"""

import sys
import os
import gzip
import bz2
import lzma
import zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import (
    CoffeeAutomaton, coarse_grain, threshold_array,
    row_majority_adjust, grid_to_bytes, DEFAULT_GRAIN
)


def compress_with_all(data: bytes) -> dict:
    """Compress data with multiple algorithms."""
    return {
        'gzip': len(gzip.compress(data, compresslevel=9)),
        'bz2': len(bz2.compress(data, compresslevel=9)),
        'lzma': len(lzma.compress(data)),
        'zlib': len(zlib.compress(data, level=9)),
    }


# [PATCH 2026-10-18] 10,000 batches (4 n^2) instead of 50,000: the 50x50 cup peaks
# at ~2,000-5,000 batches and is mixed by ~10,000, so 50,000 squeezed the hump into
# the first few snapshots.
def run_compressor_comparison(grid_size=50, total_steps=10000, num_snapshots=60):
    """Run the automaton and collect complexity estimates from each compressor."""
    ca = CoffeeAutomaton(grid_size, 'interacting', seed=42)
    grain_size = DEFAULT_GRAIN  # [PATCH 2026-10-18] was max(2, n // 10); centred window
    steps_per = max(1, total_steps // num_snapshots)

    times = []
    results = {'gzip': [], 'bz2': [], 'lzma': [], 'zlib': []}

    # Initial measurement
    grid = ca.get_binary_grid()
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold_array(coarse, 7)
    discrete = row_majority_adjust(discrete)
    data = grid_to_bytes(discrete)

    times.append(0)
    for name, size in compress_with_all(data).items():
        results[name].append(size)

    for i in range(num_snapshots):
        ca.step(steps_per)
        times.append(ca.time)

        grid = ca.get_binary_grid()
        coarse = coarse_grain(grid, grain_size)
        discrete = threshold_array(coarse, 7)
        discrete = row_majority_adjust(discrete)
        data = grid_to_bytes(discrete)

        for name, size in compress_with_all(data).items():
            results[name].append(size)

    return times, results


def plot_comparison(times, results):
    """Plot complexity curves for all compressors."""
    import matplotlib.pyplot as plt

    colors = {'gzip': 'blue', 'bz2': 'red', 'lzma': 'green', 'zlib': 'orange'}

    plt.figure(figsize=(10, 6))
    for name, sizes in results.items():
        plt.plot(times, sizes, color=colors.get(name, 'black'),
                 label=name, linewidth=1.5)

    plt.xlabel('Time step')
    plt.ylabel('Compressed size (bytes)')
    plt.title('Complexity with Different Compressors (cf. Figure 5)')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Compressor Comparison (Figure 5)")
    print("=" * 50)

    times, compressor_results = run_compressor_comparison()

    print("\nComplexity by compressor (start -> peak -> end):")
    for name, sizes in compressor_results.items():
        peak = max(sizes)
        peak_t = times[sizes.index(peak)]
        fell = sizes[-1] <= sizes[0] + 0.5 * (peak - sizes[0])
        print(f"  {name:>6}: {sizes[0]:>4} -> {peak:>4} bytes at t={peak_t:>5} -> {sizes[-1]:>4}"
              f"  ({'rises and falls' if fell else 'no clear fall'})")

    print("\nPaper (Fig. 5): different compressors give qualitatively similar curves;"
          " absolute sizes differ by each format's fixed overhead.")

    plot_comparison(times, compressor_results)
