import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portwine.analyzers.base import Analyzer

class BootstrapAnalyzer(Analyzer):
    """
    1) Expects strategy_returns & benchmark_returns of the SAME length.
    2) For each of n_sims, we build a single random list of overlapping-block starts for both
       strategy & benchmark, ensuring eqs_strat[i], eqs_bench[i] reflect the same block indices.
    3) final_returns_strat[i] and final_returns_bench[i] are thus aligned, letting us do a direct
       distribution-of-differences (strat - bench).
    4) The .plot(...) method shows:
       - Top-left: eq paths & mean lines
       - Top-right: horizontally oriented side-by-side histogram of final returns
       - Bottom (spanning 2 columns): CDF of (strategy - benchmark) final returns
    """

    def analyze(self, results, n_sims=1000, n_days=252, block_size=5, seed=42):
        """
        Overlapping block bootstrap with alignment:
          - 'strategy_returns' & 'benchmark_returns' must be same length.
          - For each path, we pick blocks from the same indices for strategy & bench.
        Returns:
          {
            'eqs_strat': (n_sims,n_days),
            'eqs_bench': (n_sims,n_days),
            'final_returns_strat': (n_sims,),
            'final_returns_bench': (n_sims,)
          }
        """
        strat_full = results.get('strategy_returns', pd.Series(dtype=float)).dropna()
        bench_full = results.get('benchmark_returns', pd.Series(dtype=float)).dropna()

        if strat_full.empty or bench_full.empty:
            print("Strategy or benchmark daily returns empty. Aborting.")
            return {}

        if len(strat_full) != len(bench_full):
            print(f"Lengths differ: strategy={len(strat_full)}, bench={len(bench_full)}. Must match.")
            return {}

        L = len(strat_full)
        if L < block_size:
            print(f"Not enough data (L={L}) for block_size={block_size}.")
            return {}

        rng = np.random.default_rng(seed)
        arr_strat = strat_full.values
        arr_bench = bench_full.values
        possible_starts = np.arange(L - block_size + 1)

        eqs_strat = []
        eqs_bench = []

        for _ in range(n_sims):
            path_s = []
            path_b = []
            while len(path_s) < n_days:
                start_i = rng.choice(possible_starts)
                block_s = arr_strat[start_i : start_i + block_size]
                block_b = arr_bench[start_i : start_i + block_size]
                path_s.extend(block_s)
                path_b.extend(block_b)

            path_s = path_s[:n_days]
            path_b = path_b[:n_days]

            eq_s = np.cumprod(1.0 + np.array(path_s))
            eq_b = np.cumprod(1.0 + np.array(path_b))
            eqs_strat.append(eq_s)
            eqs_bench.append(eq_b)

        eqs_strat = np.array(eqs_strat)  # (n_sims, n_days)
        eqs_bench = np.array(eqs_bench)

        final_s = eqs_strat[:, -1] - 1.0
        final_b = eqs_bench[:, -1] - 1.0

        return {
            'eqs_strat': eqs_strat,
            'eqs_bench': eqs_bench,
            'final_returns_strat': final_s,
            'final_returns_bench': final_b
        }

    def plot(self, results, n_sims=1000, n_days=252, block_size=5,
             seed=42, bins=30, alpha_paths=0.08, figsize=(12,10)):
        """
        Creates a 2-row figure:
          Row 0 => 2 columns
             (0,0): eq paths & mean lines (strategy=blue, bench=orange)
             (0,1): horizontally oriented side-by-side histogram of final returns
          Row 1 => single subplot spanning both columns => CDF of (strategy - benchmark)
        """
        data = self.analyze(results, n_sims=n_sims, n_days=n_days,
                            block_size=block_size, seed=seed)
        if not data:
            return

        eqs_s = data['eqs_strat']
        eqs_b = data['eqs_bench']
        final_s = data['final_returns_strat']
        final_b = data['final_returns_bench']

        # Create a figure with a 2x2 grid, but bottom row merges columns
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.7])

        ax_eq = fig.add_subplot(gs[0,0])
        ax_hist = fig.add_subplot(gs[0,1])
        ax_cdf = fig.add_subplot(gs[1,:])

        fig.suptitle(
            f"Aligned Overlapping Block Bootstrap\n"
            f"(n_days={n_days}, block_size={block_size}, n_sims={n_sims})",
            fontsize=13
        )

        x_axis = np.arange(n_days)
        # Mean lines
        mean_bench = eqs_b.mean(axis=0)
        mean_strat = eqs_s.mean(axis=0)
        ax_eq.plot(x_axis, mean_bench, color='orange', linewidth=1, label="Benchmark")
        ax_eq.plot(x_axis, mean_strat, color='blue', linewidth=1, label="Strategy")

        ########################################
        # TOP-LEFT => eq paths
        ########################################

        # Plot each path in low alpha
        for i in range(eqs_s.shape[0]):
            ax_eq.plot(x_axis, eqs_b[i], color='orange', alpha=alpha_paths, linewidth=0.5)

        for i in range(eqs_s.shape[0]):
            ax_eq.plot(x_axis, eqs_s[i], color='blue', alpha=alpha_paths, linewidth=0.5)


        ax_eq.set_title("Bootstrap eq Paths + Means (Aligned)")
        ax_eq.set_xlabel("Day index (0..n_days-1)")
        ax_eq.set_ylabel("Equity (start=1)")
        ax_eq.legend(loc='best')
        ax_eq.grid(True)

        ########################################
        # TOP-RIGHT => horizontally oriented histogram
        ########################################
        min_val = min(final_s.min(), final_b.min())
        max_val = max(final_s.max(), final_b.max())
        bin_edges = np.linspace(min_val, max_val, bins+1)

        counts_s, _ = np.histogram(final_s, bins=bin_edges)
        counts_b, _ = np.histogram(final_b, bins=bin_edges)

        # For a horizontal bar chart: x=counts, y=bin center
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        bar_height = bin_edges[1:] - bin_edges[:-1]  # each bin's "height" along the y-axis
        offset = 0.4 * bar_height
        y_strat = bin_centers - offset/2
        y_bench = bin_centers + offset/2

        # Strategy bars
        ax_hist.barh(
            y_strat, counts_s, height=offset,
            color='blue', alpha=1, edgecolor='black',
            label='Strategy'
        )
        # Benchmark bars
        ax_hist.barh(
            y_bench, counts_b, height=offset,
            color='orange', alpha=1, edgecolor='black',
            label='Benchmark'
        )

        ax_hist.set_title("Side-by-Side Final Returns (Horizontal)")
        ax_hist.set_xlabel("Count (#Paths)")
        ax_hist.set_ylabel("Final Return")
        ax_hist.grid(True)
        ax_hist.legend(loc='best')

        ########################################
        # BOTTOM => single subplot for CDF of (final_s - final_b)
        ########################################
        final_diff = final_s - final_b
        sorted_diff = np.sort(final_diff)
        cdf_vals = np.linspace(0, 1, len(sorted_diff))

        frac_above_zero = 100.0 * np.mean(final_diff > 0)

        ax_cdf.plot(sorted_diff, cdf_vals, color='purple', linewidth=2,
                    label="CDF: (Strat - Bench)")
        ax_cdf.axvline(0.0, color='red', linestyle='--',
                       label=f"0 difference\n(Strategy outperforms ~{frac_above_zero:.1f}%)")
        ax_cdf.set_title("Distribution of Differences (CDF)")
        ax_cdf.set_xlabel("(Strategy - Benchmark) Final Return")
        ax_cdf.set_ylabel("Cumulative Probability")
        ax_cdf.legend(loc='best')
        ax_cdf.grid(True)

        plt.tight_layout()
        plt.show()
