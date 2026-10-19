"""
Coffee Automaton — Implementation of Aaronson, Carroll, Ouellette (2014)
Paper: arXiv:1405.6903

Implements the two models from the paper:
  - Interacting model (Section 3.1): binary grid, random adjacent swaps
  - Non-interacting model (Section 3.2): independent cream particle random walks

And the two measurement approaches:
  - Coarse-graining with 3-bucket thresholding (Section 5)
  - Adjusted coarse-graining with 7-bucket + row-majority (Section 6)

Complexity and entropy are estimated via gzip compressed size,
as a proxy for Kolmogorov complexity.
"""

import gzip
import numpy as np
from typing import Optional, Tuple, List, Dict


# ---------------------------------------------------------------------------
# Utility: gzip-based Kolmogorov complexity proxy
# ---------------------------------------------------------------------------

def gzip_size(data: bytes, level: int = 9) -> int:
    """
    Compressed size in bytes — our proxy for K(x).
    The paper uses gzip throughout (Section 4), and shows different compressors
    give qualitatively similar curves (Figure 5).
    """
    return len(gzip.compress(data, compresslevel=level))


def grid_to_bytes(grid: np.ndarray) -> bytes:
    """Convert a numpy grid to bytes (one byte per cell) for compression."""
    # [PATCH 2026-10-18] clip before the uint8 cast: the non-interacting model is now
    # measured on particle COUNTS (uint16), and a plain astype would wrap 256 -> 0.
    return np.clip(grid, 0, 255).astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# Coarse-graining (Sections 5 and 6)
# ---------------------------------------------------------------------------

# [PATCH 2026-10-18] One grain size for every grid size (was g = n // 10). The paper
# selects g "experimentally" (Sec. 5.1) and does not report it; a fixed observation
# scale (Sec. 4: the cup "as it would be seen by a human observer from a few feet
# away") is the reading under which the paper's Fig. 7 trend appears (see README).
# Why 7: a g x g average of 0/1 cells at density 1/2 has standard deviation 0.5/g,
# and the 3-bucket edges sit 0.25 from 0.5, i.e. g/2 standard deviations away. At
# g = 5 (2.5 sd) ~1% of cells of a fully mixed cup still flicker, so the coarse
# image never becomes uniform and the curves do not fall back; g = 7 (3.5 sd) is
# the smallest odd (exactly centred) grain at which both models and both metrics
# rise and fall at n = 50 (validation/day07_paper_experiments.py).
DEFAULT_GRAIN = 7


def coarse_grain_window(grid: np.ndarray, grain_size: int) -> np.ndarray:
    """
    The paper's coarse-graining (Section 5.1): replace every cell by the
    average over the g x g square centred on it.

    "we construct a new array in which the value of each cell is the average
    of the values of the nearby cells in the fine-grained array. We define
    'nearby' cells as those within a g x g square centered at the cell in
    question." The output has the SAME shape as the input (n x n), one
    average per cell (paper Fig. 1 maps a 5 x 5 array to a 5 x 5 array).

    Borders: a window that sticks out of the grid is averaged over its
    in-bounds cells only. This is what Fig. 1 shows (corner cell = mean of
    a 2 x 2 window, edge cell = mean of a 2 x 3 window; 23 of its 25 printed
    values agree, the other two look like typos; zero-padding would agree
    with only 8).

    Even g cannot be centred exactly; the window then covers rows
    i - g//2 .. i + g//2 - 1 (the convention of scipy.ndimage.uniform_filter).

    Computed with a summed-area table (2-D prefix sums), so each window sum
    costs four lookups: O(n^2) for any g. validation/day07_window_coarse_graining.py
    checks it against a direct loop over every window.

    Args:
        grid: (n0, n1) array of cell values: 0/1 for the interacting model,
              particle counts for the non-interacting model.
        grain_size: g.
    Returns:
        (n0, n1) float64 array of local averages (in [0, 1] for 0/1 input).
    """
    # [PATCH 2026-10-18] new: the paper's sliding (centred) window
    a = np.asarray(grid, dtype=np.float64)              # (n0, n1)
    n0, n1 = a.shape
    lo, hi = grain_size // 2, (grain_size - 1) // 2      # window = i-lo .. i+hi (g cells)
    # Summed-area table with a leading zero row/col: S[i, j] = a[:i, :j].sum(), shape (n0+1, n1+1).
    # cumsum along axis 0 (down the rows), then along axis 1 (across the columns).
    S = np.zeros((n0 + 1, n1 + 1))
    S[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    # For every row index i: first row of its window and one past its last row, clipped to the grid.
    r0 = np.clip(np.arange(n0) - lo, 0, n0)              # (n0,)
    r1 = np.clip(np.arange(n0) + hi + 1, 0, n0)          # (n0,)
    c0 = np.clip(np.arange(n1) - lo, 0, n1)              # (n1,)
    c1 = np.clip(np.arange(n1) + hi + 1, 0, n1)          # (n1,)
    # r[:, None] is (n0, 1) and c[None, :] is (1, n1); indexing S with both broadcasts to
    # (n0, n1). Inclusion-exclusion: sum over rows r0..r1-1 and cols c0..c1-1 for every cell.
    total = (S[r1[:, None], c1[None, :]] - S[r0[:, None], c1[None, :]]
             - S[r1[:, None], c0[None, :]] + S[r0[:, None], c0[None, :]])
    count = (r1 - r0)[:, None] * (c1 - c0)[None, :]      # (n0, n1) in-bounds cells per window
    return total / count


def coarse_grain_block(grid: np.ndarray, grain_size: int) -> np.ndarray:
    """
    NOT the paper's method; kept for comparison. Averages NON-overlapping
    g x g blocks, so the output is (n//g) x (n//g) (trailing rows/columns that
    do not fill a block are dropped). With g = n//10 the coarse array is 10 x 10
    at every n, which is why max complexity measured this way cannot grow
    with n (paper Fig. 7 reports ~linear growth).
    """
    n = grid.shape[0]
    # Number of coarse cells
    cn = n // grain_size
    if cn == 0:
        cn = 1

    coarse = np.zeros((cn, cn), dtype=np.float64)
    for i in range(cn):
        for j in range(cn):
            r0 = i * grain_size
            c0 = j * grain_size
            r1 = min(r0 + grain_size, n)
            c1 = min(c0 + grain_size, n)
            coarse[i, j] = grid[r0:r1, c0:c1].mean()
    return coarse


def coarse_grain(grid: np.ndarray, grain_size: int, method: str = 'window') -> np.ndarray:
    """
    Coarse-grain a grid (Section 5.1).

    [PATCH 2026-10-18] method='window' (default) is the paper's centred sliding
    window (n x n output, see coarse_grain_window). method='block' is the
    repo's earlier non-overlapping block average ((n//g) x (n//g) output), kept
    only for comparison.
    """
    if method == 'window':
        return coarse_grain_window(grid, grain_size)
    if method == 'block':
        return coarse_grain_block(grid, grain_size)
    raise ValueError(f"method must be 'window' or 'block', got '{method}'")


def threshold_array(coarse: np.ndarray, num_buckets: int = 3) -> np.ndarray:
    """
    Threshold floating-point coarse-grained array into discrete buckets.

    Section 5.1: 3 buckets — "areas which are mostly coffee (values close to 0),
    mostly cream (values close to 1), or mixed (values close to 0.5)."

    Section 6.1: 7 buckets — finer resolution to reduce border artifacts.

    Returns uint8 array with values in [0, num_buckets-1].
    """
    # Map [0, 1] to [0, num_buckets-1] and round
    scaled = coarse * (num_buckets - 1)
    return np.clip(np.round(scaled), 0, num_buckets - 1).astype(np.uint8)


def row_majority_adjust(thresholded: np.ndarray) -> np.ndarray:
    """
    Adjusted coarse-graining from Section 6.1.

    "If a cell is within one threshold value of the majority value in its row,
    it is adjusted to the majority value."

    This removes border pixel artifacts that caused fake complexity
    in the non-interacting model.
    """
    adjusted = thresholded.copy()
    for i in range(adjusted.shape[0]):
        row = adjusted[i]
        # Find majority value in this row
        values, counts = np.unique(row, return_counts=True)
        majority = values[np.argmax(counts)]
        # Snap cells within 1 bucket of majority
        mask = np.abs(row.astype(np.int16) - int(majority)) <= 1
        adjusted[i, mask] = majority
    return adjusted


# ---------------------------------------------------------------------------
# Complexity and entropy measurement
# ---------------------------------------------------------------------------

def measure_entropy(grid: np.ndarray) -> int:
    """
    Entropy estimate: gzip compressed size of the fine-grained state.

    Section 5.1: "the estimated entropy of the automaton state is the compressed
    file size of the fine-grained array."
    """
    return gzip_size(grid_to_bytes(grid))


def measure_complexity(grid: np.ndarray, grain_size: int = DEFAULT_GRAIN,
                       num_buckets: int = 7, adjust: bool = True,
                       method: str = 'window') -> int:
    """
    Apparent complexity estimate: gzip compressed size of coarse-grained state.

    Section 5.1: "The estimated complexity of the state, K(S), is the file size
    of the thresholded, coarse-grained array after compression."

    Args:
        grid: The fine-grained state (N x N): 0/1 cells (interacting model) or
              particle counts (non-interacting model)
        grain_size: g in the paper (side of the averaging window)
        num_buckets: Number of threshold buckets (3 for Section 5, 7 for Section 6)
        adjust: Whether to apply row-majority adjustment (Section 6)
        method: 'window' (paper, Sec. 5.1) or 'block' (comparison only)
    """
    # [PATCH 2026-10-18] default grain DEFAULT_GRAIN (was 10) and the paper's window method
    coarse = coarse_grain(grid, grain_size, method)
    discrete = threshold_array(coarse, num_buckets)
    if adjust:
        discrete = row_majority_adjust(discrete)
    return gzip_size(grid_to_bytes(discrete))


# ---------------------------------------------------------------------------
# The Coffee Automaton (Section 3)
# ---------------------------------------------------------------------------

class CoffeeAutomaton:
    """
    The coffee automaton from Section 3 of the paper.

    A 2D grid of binary values: 1 = cream, 0 = coffee.
    Initial state: top half = cream (1), bottom half = coffee (0).

    Two models:
    - 'interacting' (Section 3.1): One adjacent-different-pair swap per step.
      "only one particle may occupy each cell."
    - 'non_interacting' (Section 3.2): Each cream particle random-walks
      independently. Multiple can occupy one cell.
    """

    def __init__(self, grid_size: int = 100, model: str = 'interacting',
                 seed: Optional[int] = None):
        """
        Args:
            grid_size: Side length of the square grid (N in paper)
            model: 'interacting' or 'non_interacting'
            seed: Random seed for reproducibility
        """
        self.n = grid_size
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.time = 0
        # [PATCH 2026-10-18] Swaps actually performed (interacting model). The paper's
        # time step is ONE swap (Sec. 3.1), so this is time in the paper's units;
        # self.time counts the repo's batched steps.
        self.swaps = 0

        if model == 'interacting':
            # Binary grid: top half = 1 (cream), bottom half = 0 (coffee)
            # Section 3: "The automaton begins in a state in which the top half
            # of the cells are filled with ones, and the bottom half is filled
            # with zeros."
            self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
            self.grid[:grid_size // 2, :] = 1
        elif model == 'non_interacting':
            # For non-interacting, track cream particle positions.
            # Count grid: how many cream particles at each cell.
            # Initially, one per cell in top half.
            self.grid = np.zeros((grid_size, grid_size), dtype=np.uint16)
            self.grid[:grid_size // 2, :] = 1
            # Also store individual particle positions for random walks
            self._init_particles()
        else:
            raise ValueError(f"model must be 'interacting' or 'non_interacting', got '{model}'")

    def _init_particles(self):
        """Initialize particle position list for non-interacting model."""
        self.particles = []
        for r in range(self.n // 2):
            for c in range(self.n):
                self.particles.append([r, c])
        self.particles = np.array(self.particles, dtype=np.int32)

    def step(self, num_steps: int = 1):
        """
        Advance the automaton by num_steps.

        For the interacting model, we do batch swaps for efficiency:
        select many random adjacent pairs, filter to those that differ,
        and swap them. This is faster than one-swap-per-step while
        preserving the qualitative dynamics.

        The paper does one swap per step (Section 3.1), but that's
        extremely slow for large grids. Batch swapping speeds things
        up without changing the equilibrium behavior.

        [PATCH 2026-10-18] Units: one interacting "step" here is one batch,
        which performs ~0.07 n^2 swaps once the cup is mixing (~175 at
        n = 50); self.swaps counts them, i.e. time in the paper's units.
        A batch applies disjoint swaps of differing neighbours, so it equals
        that many single paper steps applied one after another; it differs
        from the paper's chain only in how the pairs are drawn (pairs that
        collide inside a batch are dropped). For the non-interacting model a
        step is exactly the paper's step (every particle moves once).
        """
        if self.model == 'interacting':
            self._step_interacting(num_steps)
        else:
            self._step_non_interacting(num_steps)
        self.time += num_steps

    def _step_interacting(self, num_steps: int):
        """
        Interacting model: random adjacent swaps.

        Section 3.1: "at each time step, one pair of horizontally or
        vertically adjacent, differing particles is selected, and the
        particles' positions are swapped."

        We batch multiple non-conflicting swaps per call for speed.
        """
        n = self.n
        grid = self.grid

        # How many candidate swaps per batch —
        # roughly n*n/4 so each pixel gets ~1 chance per batch
        batch_size = max(1, (n * n) // 4)

        for _ in range(num_steps):
            # Pick random cells
            rows = self.rng.integers(0, n, size=batch_size)
            cols = self.rng.integers(0, n, size=batch_size)

            # Pick random direction: 0=up, 1=down, 2=left, 3=right
            dirs = self.rng.integers(0, 4, size=batch_size)

            # Compute neighbor coordinates
            dr = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            nr = rows + dr[dirs, 0]
            nc = cols + dr[dirs, 1]

            # Filter: within bounds
            valid = (nr >= 0) & (nr < n) & (nc >= 0) & (nc < n)

            rows = rows[valid]
            cols = cols[valid]
            nr = nr[valid]
            nc = nc[valid]

            # Filter: different values (only swap if they differ)
            diff = grid[rows, cols] != grid[nr, nc]
            rows = rows[diff]
            cols = cols[diff]
            nr = nr[diff]
            nc = nc[diff]

            # [PATCH 2026-10-18] Pairs drawn in one batch can share a cell; swapping
            # them simultaneously with fancy indexing is then not a permutation and
            # creates/destroys cream (a 50x50 run drifted from 1250 to 1208-1342
            # particles). Keep only pairs whose cells occur in no other pair, so each
            # batch is a set of disjoint swaps and the particle count is conserved.
            a = rows * n + cols
            b = nr * n + nc
            cells, counts = np.unique(np.concatenate([a, b]), return_counts=True)
            shared = cells[counts > 1]
            keep = ~(np.isin(a, shared) | np.isin(b, shared))
            rows, cols, nr, nc = rows[keep], cols[keep], nr[keep], nc[keep]

            # Swap
            if len(rows) > 0:
                temp = grid[rows, cols].copy()
                grid[rows, cols] = grid[nr, nc]
                grid[nr, nc] = temp
            self.swaps += len(rows)  # [PATCH 2026-10-18] paper time: one swap = one step

    def _step_non_interacting(self, num_steps: int):
        """
        Non-interacting model: each cream particle random-walks.

        Section 3.2: "at each time step, each cream particle in the system
        moves one step in a randomly chosen direction."
        """
        n = self.n
        directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        for _ in range(num_steps):
            # Move each particle in a random direction
            choices = self.rng.integers(0, 4, size=len(self.particles))
            moves = directions[choices]          # (P,) codes -> (P, 2) offsets via fancy indexing
            self.particles += moves

            # Periodic boundary conditions
            # Section 9 (Appendix) assumes periodic boundaries
            self.particles[:, 0] %= n
            self.particles[:, 1] %= n

        # [PATCH 2026-10-18] Rebuild the count grid once, vectorized (was a Python loop over
        # every particle at every step; same result, the RNG draws are unchanged).
        # Flatten (row, col) -> row * n + col; np.bincount counts how often each cell id
        # occurs (minlength=n*n gives one bin per cell, empty cells -> 0); reshape back to (n, n).
        flat = self.particles[:, 0] * n + self.particles[:, 1]
        self.grid[:] = np.bincount(flat, minlength=n * n).reshape(n, n)

    def get_binary_grid(self) -> np.ndarray:
        """
        Occupancy as a 0/1 grid (for pictures and the mixing fraction).

        For the interacting model, this is just self.grid.
        For the non-interacting model, convert counts to binary:
        1 if any cream particle present, 0 otherwise.
        Measurements use get_state() instead.
        """
        if self.model == 'interacting':
            return self.grid
        else:
            return (self.grid > 0).astype(np.uint8)

    def get_state(self) -> np.ndarray:
        """
        [PATCH 2026-10-18] The fine-grained array the paper measures.

        Interacting model: the 0/1 grid. Non-interacting model: the particle
        COUNT per cell ("any number of cream particles may occupy a single
        cell in the state array", Sec. 3.2; the Appendix coarse-grains the
        counts a_t(x, y)). The repo previously measured occupancy (count > 0),
        which caps every cell at 1 and changes the coarse-grained densities.
        """
        return self.grid

    def entropy(self) -> int:
        """Entropy estimate: gzip(fine-grained state). See Section 5.1."""
        return measure_entropy(self.get_state())

    def complexity(self, grain_size: Optional[int] = None,
                   num_buckets: int = 7, adjust: bool = True,
                   method: str = 'window') -> int:
        """
        Apparent complexity estimate: gzip(coarse-grained state).
        See Sections 5 and 6.

        Args:
            grain_size: g in the paper. Default: DEFAULT_GRAIN (7)
            num_buckets: 3 (Section 5) or 7 (Section 6)
            adjust: Apply row-majority adjustment (Section 6)
            method: 'window' (paper) or 'block' (comparison only)
        """
        if grain_size is None:
            grain_size = DEFAULT_GRAIN
        return measure_complexity(self.get_state(), grain_size,
                                  num_buckets, adjust, method)

    @property
    def paper_time(self) -> int:
        """[PATCH 2026-10-18] Time in the paper's units: swaps (interacting), steps (non-interacting)."""
        return self.swaps if self.model == 'interacting' else self.time

    def fraction_mixed(self) -> float:
        """
        Fraction of cells that are "wrong" (cream in bottom half or
        coffee in top half). A simple monotonic proxy for entropy.
        """
        grid = self.get_binary_grid()
        n = self.n
        # Cream in bottom half + coffee in top half
        wrong = (np.sum(grid[n // 2:, :] == 1) +
                 np.sum(grid[:n // 2, :] == 0))
        total = n * n
        return wrong / total


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def default_total_steps(grid_size: int, model: str = 'interacting') -> int:
    """
    [PATCH 2026-10-18] A run length that covers the rise, the peak and most of the
    fall (the old default, 100,000 batches at n = 50, is ~10x the mixing time, so the
    whole hump sat in the first few percent of the plot).

    Interacting: 4 n^2 batches (the coarse profile relaxes over ~n^2 batches; the
    3-bucket peak comes at ~n^2 batches). Non-interacting: n^2 // 2 steps (every
    particle moves every step, so mixing is much faster per step).
    """
    if model == 'interacting':
        return 4 * grid_size * grid_size
    return max(1, grid_size * grid_size // 2)


def run_simulation(grid_size: int = 50, total_steps: Optional[int] = None,
                   num_snapshots: int = 100, model: str = 'interacting',
                   grain_size: Optional[int] = None, num_buckets: int = 7,
                   adjust: bool = True, seed: Optional[int] = 42,
                   method: str = 'window'
                   ) -> Dict[str, np.ndarray]:
    """
    Run the coffee automaton and collect measurements over time.

    Returns a dict with keys:
        'times': array of time steps (repo steps: batches for the interacting model)
        'paper_time': the same instants in the paper's units (swaps / steps)
        'entropy': array of entropy estimates
        'complexity': array of complexity estimates
        'fraction_mixed': array of mixing fractions
        'grids': list of state snapshots (if grid_size <= 100); counts for
                 the non-interacting model

    Args:
        grid_size: N (side length)
        total_steps: Total number of steps (default: default_total_steps)
        num_snapshots: Number of measurement points
        model: 'interacting' or 'non_interacting'
        grain_size: g, side of the averaging window (default: DEFAULT_GRAIN)
        num_buckets: 3 or 7 (Section 5 vs Section 6)
        adjust: Row-majority adjustment (Section 6)
        seed: Random seed
        method: coarse-graining, 'window' (paper) or 'block' (comparison)
    """
    # [PATCH 2026-10-18] paper coarse-graining by default, measured on get_state()
    # (counts for the non-interacting model), paper-unit time recorded, run length
    # matched to the mixing time unless given.
    ca = CoffeeAutomaton(grid_size, model, seed)

    if grain_size is None:
        grain_size = DEFAULT_GRAIN
    if total_steps is None:
        total_steps = default_total_steps(grid_size, model)

    steps_per_snapshot = max(1, total_steps // num_snapshots)
    store_grids = grid_size <= 100

    times = []
    paper_times = []
    entropies = []
    complexities = []
    fractions = []
    grids = []

    def record():
        times.append(ca.time)
        paper_times.append(ca.paper_time)
        entropies.append(ca.entropy())
        complexities.append(ca.complexity(grain_size, num_buckets, adjust, method))
        fractions.append(ca.fraction_mixed())
        if store_grids:
            grids.append(ca.get_state().copy())

    record()  # initial state
    for i in range(num_snapshots):
        ca.step(steps_per_snapshot)
        record()

    results = {
        'times': np.array(times),
        'paper_time': np.array(paper_times),
        'entropy': np.array(entropies),
        'complexity': np.array(complexities),
        'fraction_mixed': np.array(fractions),
    }
    if store_grids:
        results['grids'] = grids

    return results


def compare_models(grid_size: int = 50, total_steps: Optional[int] = None,
                   num_snapshots: int = 100, seed: int = 42,
                   grain_size: Optional[int] = None, num_buckets: int = 7,
                   adjust: bool = True, method: str = 'window',
                   noninteracting_steps: Optional[int] = None
                   ) -> Tuple[Dict, Dict]:
    """
    Run both interacting and non-interacting models and return results.
    Useful for replicating the paper's comparison (Figures 2 and 10).

    [PATCH 2026-10-18] The two models have different time units (a batch of swaps
    vs. every particle moving once), so each gets its own run length
    (default_total_steps) unless total_steps / noninteracting_steps are given.
    Both are measured with the same grain and thresholds.
    """
    print(f"Running interacting model ({grid_size}x{grid_size})...")
    interacting = run_simulation(grid_size, total_steps, num_snapshots,
                                 'interacting', grain_size, num_buckets,
                                 adjust, seed, method)

    print(f"Running non-interacting model ({grid_size}x{grid_size})...")
    non_interacting = run_simulation(grid_size, noninteracting_steps, num_snapshots,
                                     'non_interacting', grain_size, num_buckets,
                                     adjust, seed, method)

    return interacting, non_interacting


def fit_r2(x, y, degree: int = 1) -> Tuple[np.ndarray, float]:
    """[PATCH 2026-10-18] Least-squares polynomial fit (np.polyfit) and its r^2 = 1 - SS_res / SS_tot."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coef = np.polyfit(x, y, degree)            # highest power first, e.g. [slope, intercept]
    resid = y - np.polyval(coef, x)            # evaluate the fitted polynomial at x
    ss_tot = np.sum((y - y.mean()) ** 2)
    return coef, (1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float('nan')


def scaling_experiment(sizes: List[int] = None, steps_per_pixel: float = 4,
                       num_snapshots: int = 150, seed: int = 42,
                       grain_size: int = DEFAULT_GRAIN, method: str = 'window',
                       seeds: int = 1, verbose: bool = True
                       ) -> Dict[str, np.ndarray]:
    """
    Replicate the scaling experiment from Figures 6-8 (interacting model).
    Measure max entropy, max complexity, and time to max complexity
    as a function of grid size.

    The paper found:
    - Max entropy ~ n^2
    - Max complexity ~ n
    - Time to max complexity ~ n^2

    [PATCH 2026-10-18] Paper coarse-graining (centred window) with ONE grain size for
    all n; both complexity metrics are measured on the same runs: 'max_complexity'
    is the Sec. 5 metric (3 buckets, no adjustment), the one Figs. 7-8 are about,
    and 'max_complexity_adjusted' the Sec. 6 metric (7 buckets + row majority).
    Each n runs steps_per_pixel * n^2 batches (the old 40 n^2 was ~10x the mixing
    time, so the peak fell between the first snapshots). Values are averaged over
    `seeds` runs (seed, seed+1, ...); peak times are reported in batches and in
    swaps (the paper's unit).
    """
    if sizes is None:
        sizes = [20, 30, 40, 50, 60]

    keys = ('max_entropy', 'max_complexity', 'max_complexity_adjusted',
            'peak_time', 'peak_swaps')
    out = {k: [] for k in keys}
    out['max_complexity_std'] = []

    for n in sizes:
        total_steps = int(steps_per_pixel * n * n)
        steps_per = max(1, total_steps // num_snapshots)
        per_seed = {k: [] for k in keys}
        for s in range(seeds):
            if verbose:
                print(f"  Grid size {n}x{n}, {total_steps} batches, seed {seed + s}...")
            ca = CoffeeAutomaton(n, 'interacting', seed + s)
            ent, c5, c6, times, swaps = [], [], [], [], []
            for i in range(num_snapshots + 1):
                if i:
                    ca.step(steps_per)
                state = ca.get_state()
                ent.append(measure_entropy(state))
                c5.append(measure_complexity(state, grain_size, 3, False, method))
                c6.append(measure_complexity(state, grain_size, 7, True, method))
                times.append(ca.time)
                swaps.append(ca.swaps)
            k = int(np.argmax(c5))             # first snapshot where the Sec. 5 metric peaks
            per_seed['max_entropy'].append(max(ent))
            per_seed['max_complexity'].append(c5[k])
            per_seed['max_complexity_adjusted'].append(max(c6))
            per_seed['peak_time'].append(times[k])
            per_seed['peak_swaps'].append(swaps[k])
        for key in keys:
            out[key].append(float(np.mean(per_seed[key])))
        out['max_complexity_std'].append(float(np.std(per_seed['max_complexity'])))

    result = {key: np.array(val) for key, val in out.items()}
    result['sizes'] = np.array(sizes)
    result['grain_size'] = grain_size
    return result


# ---------------------------------------------------------------------------
# Main: quick demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Coffee Automaton (Aaronson, Carroll, Ouellette 2014)")
    print("=" * 55)

    # Quick interacting simulation
    print("\nRunning interacting model (50x50, 10000 batches of swaps)...")
    results = run_simulation(grid_size=50, total_steps=10000, num_snapshots=50)

    peak_idx = np.argmax(results['complexity'])
    print(f"\nResults:")
    print(f"  Peak complexity at t = {results['times'][peak_idx]} batches "
          f"({results['paper_time'][peak_idx]:.2e} swaps)")
    print(f"  Peak complexity value = {results['complexity'][peak_idx]} bytes")
    print(f"  Initial / final complexity = {results['complexity'][0]} / "
          f"{results['complexity'][-1]} bytes")
    print(f"  Final entropy = {results['entropy'][-1]} bytes")
    print(f"  Fraction mixed at end = {results['fraction_mixed'][-1]:.3f}")

    # [PATCH 2026-10-18] the old check compared the means of the two halves, which a
    # flat curve passes; ask for an interior peak and a fall at least halfway back.
    c = results['complexity']
    fell = c[-1] <= c[0] + 0.5 * (c[peak_idx] - c[0])
    if 0 < peak_idx < len(c) - 1 and fell:
        print("\n  Complexity shows rise-then-fall pattern [OK]")
    else:
        print("\n  [NOTE] No clear rise and fall; try more steps or another grain size")
