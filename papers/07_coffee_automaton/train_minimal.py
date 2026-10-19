"""
train_minimal.py — Run the coffee automaton experiments from the command line.

Usage:
    # Basic run: interacting model, entropy + complexity over time
    # (50x50, 4 n^2 = 10,000 batches of swaps, grain 7, 7 buckets + adjustment)
    python train_minimal.py

    # Compare interacting vs non-interacting (Figures 2 and 10)
    python train_minimal.py --compare-models

    # Section 5 (3 buckets) vs Section 6 (7 buckets + row-majority adjustment)
    python train_minimal.py --compare-methods

    # Scaling experiment (Figures 6-8): n = 20..60, 2 seeds, ~1 min
    python train_minimal.py --scaling --no-plot
    # Larger grids, n = 20..120 (opt-in, ~10 min on one core)
    python train_minimal.py --scaling --large --no-plot

    # The repo's earlier non-overlapping block averaging, for comparison
    python train_minimal.py --scaling --coarse block --no-plot

Paper: Aaronson, Carroll, Ouellette (2014), arXiv:1405.6903
[PATCH 2026-10-18] Coarse-graining is now the paper's centred window (Sec. 5.1) with
one grain size for all n; run lengths default to the mixing time; --grain, --coarse,
--seeds and --large added.
"""

import argparse
import sys
import os
import numpy as np

# Add parent dir to path so we can import implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implementation import (
    CoffeeAutomaton, run_simulation, compare_models,
    scaling_experiment, measure_entropy, measure_complexity,
    DEFAULT_GRAIN, default_total_steps, fit_r2
)


def plot_single_run(results, model_name='interacting', save_path=None):
    """Plot entropy and complexity vs time for a single simulation."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Entropy and complexity on same time axis
    ax1.plot(results['times'], results['entropy'], 'b-', label='Entropy (gzip fine-grained)')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Compressed size (bytes)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title(f'{model_name.replace("_", "-").title()} Model')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(results['times'], results['complexity'], 'r-', label='Complexity (gzip coarse-grained)')
    ax1_twin.set_ylabel('Complexity (bytes)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    # Complexity alone with peak marked
    ax2.plot(results['times'], results['complexity'], 'r-', linewidth=2)
    peak_idx = np.argmax(results['complexity'])
    ax2.axvline(results['times'][peak_idx], color='gray', linestyle='--', alpha=0.5)
    ax2.annotate(f'Peak at t={results["times"][peak_idx]}',
                 xy=(results['times'][peak_idx], results['complexity'][peak_idx]),
                 xytext=(10, 10), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Apparent complexity (bytes)')
    ax2.set_title('Complexity Rise and Fall')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_model_comparison(interacting, non_interacting, save_path=None):
    """
    Compare interacting vs non-interacting — replicates Figure 2 / Figure 10.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Interacting entropy
    axes[0, 0].plot(interacting['times'], interacting['entropy'], 'b-')
    axes[0, 0].set_title('Interacting: Entropy')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('gzip size (bytes)')

    # Interacting complexity
    axes[0, 1].plot(interacting['times'], interacting['complexity'], 'r-')
    axes[0, 1].set_title('Interacting: Complexity')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('gzip size (bytes)')

    # Non-interacting entropy
    axes[1, 0].plot(non_interacting['times'], non_interacting['entropy'], 'b-')
    axes[1, 0].set_title('Non-Interacting: Entropy')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('gzip size (bytes)')

    # Non-interacting complexity
    axes[1, 1].plot(non_interacting['times'], non_interacting['complexity'], 'r-')
    axes[1, 1].set_title('Non-Interacting: Complexity')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('gzip size (bytes)')

    plt.suptitle('Coffee Automaton: Interacting vs Non-Interacting\n'
                 '(Aaronson, Carroll, Ouellette 2014 — Figures 2/10)',
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_method_comparison(results_3bucket, results_7bucket, save_path=None):
    """
    Compare Section 5 (3-bucket) vs Section 6 (7-bucket adjusted).
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(results_3bucket['times'], results_3bucket['complexity'],
             'r-', label='3-bucket (Section 5)')
    ax1.plot(results_7bucket['times'], results_7bucket['complexity'],
             'b-', label='7-bucket adjusted (Section 6)')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Apparent complexity (bytes)')
    ax1.set_title('Interacting Model')
    ax1.legend()

    # If we have non-interacting data...
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Apparent complexity (bytes)')
    ax2.set_title('Non-Interacting Model')
    ax2.text(0.5, 0.5, 'Run with --compare-models\nfor non-interacting data',
             ha='center', va='center', transform=ax2.transAxes, fontsize=10, color='gray')

    plt.suptitle('Section 5 vs Section 6 Coarse-Graining Methods')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_scaling(scaling_results, save_path=None):
    """
    Plot scaling results — replicates Figures 6-8.
    """
    import matplotlib.pyplot as plt

    sizes = scaling_results['sizes']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Max entropy vs n (should be quadratic ~ n^2)
    axes[0].plot(sizes, scaling_results['max_entropy'], 'bo-')
    # Quadratic fit
    coeffs = np.polyfit(sizes, scaling_results['max_entropy'], 2)
    x_fit = np.linspace(sizes[0], sizes[-1], 50)
    axes[0].plot(x_fit, np.polyval(coeffs, x_fit), 'b--', alpha=0.5,
                 label=f'Quadratic fit')
    axes[0].set_xlabel('Grid size (n)')
    axes[0].set_ylabel('Max entropy (bytes)')
    axes[0].set_title('Max Entropy vs Grid Size\n(Expected: ~ n^2)')
    axes[0].legend()

    # Max complexity vs n (should be linear ~ n)
    axes[1].plot(sizes, scaling_results['max_complexity'], 'ro-')
    # Linear fit
    coeffs = np.polyfit(sizes, scaling_results['max_complexity'], 1)
    axes[1].plot(x_fit, np.polyval(coeffs, x_fit), 'r--', alpha=0.5,
                 label=f'Linear fit')
    axes[1].set_xlabel('Grid size (n)')
    axes[1].set_ylabel('Max complexity (bytes)')
    axes[1].set_title('Max Complexity vs Grid Size\n(Expected: ~ n)')
    axes[1].legend()

    # Time to max complexity vs n (should be quadratic ~ n^2)
    axes[2].plot(sizes, scaling_results['peak_time'], 'go-')
    coeffs = np.polyfit(sizes, scaling_results['peak_time'], 2)
    axes[2].plot(x_fit, np.polyval(coeffs, x_fit), 'g--', alpha=0.5,
                 label=f'Quadratic fit')
    axes[2].set_xlabel('Grid size (n)')
    axes[2].set_ylabel('Time to peak complexity')
    axes[2].set_title('Time to Peak vs Grid Size\n(Expected: ~ n^2)')
    axes[2].legend()

    plt.suptitle('Scaling Analysis (Figures 6-8 from Paper)', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def describe_curve(c):
    """[PATCH 2026-10-18] 'rise and fall' = interior peak and a final value at least halfway back down."""
    c = np.asarray(c, dtype=float)
    k = int(np.argmax(c))
    hump = 0 < k < len(c) - 1 and c[-1] <= c[0] + 0.5 * (c[k] - c[0])
    return (f"{c[0]:.0f} -> peak {c[k]:.0f} -> final {c[-1]:.0f} bytes "
            f"({'rises and falls' if hump else 'no clear rise and fall'})")


def main():
    parser = argparse.ArgumentParser(
        description='Coffee Automaton — Aaronson, Carroll, Ouellette (2014)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_minimal.py
  python train_minimal.py --compare-models
  python train_minimal.py --scaling --no-plot
  python train_minimal.py --scaling --large --no-plot
        """
    )

    parser.add_argument('--grid-size', type=int, default=50,
                        help='Grid side length N (default: 50)')
    # [PATCH 2026-10-18] default run length follows the mixing time (was 100000 batches)
    parser.add_argument('--steps', type=int, default=None,
                        help='Total simulation steps (default: 4 N^2 batches for the interacting '
                             'model, N^2/2 steps for the non-interacting one)')
    parser.add_argument('--snapshots', type=int, default=100,
                        help='Number of measurement points (default: 100)')
    parser.add_argument('--model', type=str, default='interacting',
                        choices=['interacting', 'non_interacting'],
                        help='Which model to run (default: interacting)')
    parser.add_argument('--buckets', type=int, default=7,
                        choices=[3, 7],
                        help='Threshold buckets: 3 (Section 5) or 7 (Section 6)')
    parser.add_argument('--no-adjust', action='store_true',
                        help='Disable row-majority adjustment (use raw thresholding)')
    parser.add_argument('--grain', type=int, default=None,
                        help=f'Grain size g, side of the averaging window (default: {DEFAULT_GRAIN})')
    parser.add_argument('--coarse', type=str, default='window', choices=['window', 'block'],
                        help="Coarse-graining: 'window' = paper Sec. 5.1 (centred g x g average per "
                             "cell); 'block' = non-overlapping blocks (repo's earlier method)")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Mode flags
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare interacting vs non-interacting')
    parser.add_argument('--compare-methods', action='store_true',
                        help='Compare Section 5 (3-bucket) vs Section 6 (7-bucket)')
    parser.add_argument('--scaling', action='store_true',
                        help='Run scaling experiment (Figures 6-8)')
    parser.add_argument('--sizes', type=int, nargs='+', default=None,
                        help='Grid sizes for scaling experiment (default: 20 30 40 50 60)')
    parser.add_argument('--large', action='store_true',
                        help='Scaling on larger grids, n = 20..120 (~10 min on one core)')
    parser.add_argument('--seeds', type=int, default=2,
                        help='Scaling: runs per grid size, averaged (default: 2)')

    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of showing')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting (print results only)')

    args = parser.parse_args()
    grain = args.grain if args.grain is not None else DEFAULT_GRAIN

    if args.scaling:
        # Scaling experiment (Figures 6-8)
        print("Scaling experiment (Figures 6-8 from paper)")
        print("=" * 50)
        sizes = args.sizes or ([20, 40, 60, 80, 100, 120] if args.large else [20, 30, 40, 50, 60])
        print(f"interacting model, {args.coarse} coarse-graining, grain {grain} for every n, "
              f"4 n^2 batches per run, {args.seeds} seed(s) per n")
        results = scaling_experiment(sizes=sizes, seed=args.seed, grain_size=grain,
                                     method=args.coarse, seeds=args.seeds)

        print("\nResults (means over seeds; max complexity = Sec. 5 metric, 3 buckets;"
              " adjusted = Sec. 6 metric, 7 buckets + row majority):")
        print(f"  {'Size':>5} {'Max entropy':>11} {'Max cx (S5)':>11} {'Max cx (S6)':>11}"
              f" {'Peak (batches)':>14} {'Peak (swaps)':>12}")
        for i, n in enumerate(results['sizes']):
            print(f"  {n:>5} {results['max_entropy'][i]:>11.1f} {results['max_complexity'][i]:>11.1f}"
                  f" {results['max_complexity_adjusted'][i]:>11.1f} {results['peak_time'][i]:>14.0f}"
                  f" {results['peak_swaps'][i]:>12.3g}")

        # [PATCH 2026-10-18] the fits the paper reports (Figs. 6-8), plus power-law exponents
        n_arr = results['sizes']
        _, r2_e = fit_r2(n_arr, results['max_entropy'], 2)
        c5, r2_5 = fit_r2(n_arr, results['max_complexity'], 1)
        c6, r2_6 = fit_r2(n_arr, results['max_complexity_adjusted'], 1)
        _, r2_t = fit_r2(n_arr, results['peak_swaps'], 2)
        # slope of log(y) vs log(n) = exponent p in y ~ n^p
        p_b = np.polyfit(np.log(n_arr), np.log(results['peak_time']), 1)[0]
        p_s = np.polyfit(np.log(n_arr), np.log(results['peak_swaps']), 1)[0]
        p_5 = np.polyfit(np.log(n_arr), np.log(results['max_complexity']), 1)[0]
        p_6 = np.polyfit(np.log(n_arr), np.log(results['max_complexity_adjusted']), 1)[0]
        print(f"\n  Fig. 6  max entropy, quadratic fit:            r^2 = {r2_e:.4f}  (paper 0.9999)")
        print(f"  Fig. 7  max complexity (S5), linear fit:       slope {c5[0]:+.2f} bytes per unit n,"
              f" r^2 = {r2_5:.4f}  (paper 0.9798)")
        print(f"          max complexity (S6 adjusted), linear:  slope {c6[0]:+.2f} bytes per unit n,"
              f" r^2 = {r2_6:.4f}")
        print(f"          power-law exponents of max complexity: n^{p_5:.2f} (S5), n^{p_6:.2f} (S6)")
        print(f"  Fig. 8  time to peak (swaps), quadratic fit:   r^2 = {r2_t:.4f}  (paper 0.9878)")
        print(f"          power-law exponents of time to peak:   n^{p_b:.2f} in batches, n^{p_s:.2f} in swaps")

        if not args.no_plot:
            plot_scaling(results, save_path=args.save)

    elif args.compare_models:
        # Interacting vs non-interacting (Figures 2 / 10)
        print("Comparing interacting vs non-interacting models")
        print("=" * 50)
        inter, non_inter = compare_models(
            args.grid_size, args.steps, args.snapshots, args.seed,
            grain_size=grain, num_buckets=args.buckets,
            adjust=not args.no_adjust, method=args.coarse)

        inter_peak = np.argmax(inter['complexity'])
        non_peak = np.argmax(non_inter['complexity'])
        print(f"\n(grain {grain}, {args.buckets} buckets, adjustment "
              f"{'off' if args.no_adjust else 'on'})")
        print(f"Interacting:     {describe_curve(inter['complexity'])}; peak at t = "
              f"{inter['times'][inter_peak]} batches = {inter['paper_time'][inter_peak]:.3g} swaps")
        print(f"Non-interacting: {describe_curve(non_inter['complexity'])}; peak at t = "
              f"{non_inter['times'][non_peak]} steps")

        if not args.no_plot:
            plot_model_comparison(inter, non_inter, save_path=args.save)

    elif args.compare_methods:
        # Section 5 vs Section 6 methods
        print("Comparing coarse-graining methods")
        print("=" * 50)

        print("Running with 3-bucket thresholding (Section 5)...")
        results_3 = run_simulation(
            args.grid_size, args.steps, args.snapshots,
            args.model, grain_size=grain, num_buckets=3, adjust=False,
            seed=args.seed, method=args.coarse)

        print("Running with 7-bucket adjusted (Section 6)...")
        results_7 = run_simulation(
            args.grid_size, args.steps, args.snapshots,
            args.model, grain_size=grain, num_buckets=7, adjust=True,
            seed=args.seed, method=args.coarse)

        peak_3 = np.argmax(results_3['complexity'])
        peak_7 = np.argmax(results_7['complexity'])
        print(f"\n3-bucket:          {describe_curve(results_3['complexity'])}, peak at t={results_3['times'][peak_3]}")
        print(f"7-bucket adjusted: {describe_curve(results_7['complexity'])}, peak at t={results_7['times'][peak_7]}")

        if not args.no_plot:
            plot_method_comparison(results_3, results_7, save_path=args.save)

    else:
        # Single model run
        steps = args.steps if args.steps is not None else default_total_steps(args.grid_size, args.model)
        print(f"Running {args.model} model ({args.grid_size}x{args.grid_size}, "
              f"{steps} steps, grain {grain} ({args.coarse}), {args.buckets} buckets, "
              f"adjust={'off' if args.no_adjust else 'on'})")
        print("=" * 50)

        results = run_simulation(
            args.grid_size, steps, args.snapshots,
            args.model, grain_size=grain, num_buckets=args.buckets,
            adjust=not args.no_adjust, seed=args.seed, method=args.coarse)

        peak_idx = np.argmax(results['complexity'])
        print(f"\nResults:")
        print(f"  Complexity: {describe_curve(results['complexity'])}")
        print(f"  Peak at t = {results['times'][peak_idx]} steps"
              + (f" = {results['paper_time'][peak_idx]:.3g} swaps (paper's unit)"
                 if args.model == 'interacting' else ""))
        print(f"  Entropy: {results['entropy'][0]} -> {results['entropy'][-1]} bytes")
        print(f"  Mixing fraction: {results['fraction_mixed'][-1]:.3f}")

        if not args.no_plot:
            plot_single_run(results, args.model, save_path=args.save)


if __name__ == '__main__':
    main()
