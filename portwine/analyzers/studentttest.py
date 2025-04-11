import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_ind
from portwine.analyzers.base import Analyzer

class StudentsTTestAnalyzer(Analyzer):
    """
    A single-column analyzer that can optionally display:
      1) Equity curve (if with_equity_curve=True)
      2) Histogram of daily returns (strategy vs. benchmark)
      3) A stats table with significance tests (mean vs zero + vs benchmark),
         plus color-coded cells.

    No train/test split logic. The entire dataset is analyzed as one period.
    """

    def analyze(self, results):
        """
        1) Retrieves 'strategy_returns' from results, plus 'benchmark_returns' if present.
        2) T-test vs zero => (t_stat_vs_zero, p_value_vs_zero)
        3) T-test vs benchmark => (t_stat_vs_bench, p_value_vs_bench) if present
        4) Return a dictionary with the stats for the final table.

        Returns
        -------
        dict
          {
            'mean_ret': float,
            't_stat_vs_zero': float,
            'p_value_vs_zero': float,
            't_stat_vs_bench': float,
            'p_value_vs_bench': float,
            'translation': str
          }
        """
        strat_ret = results.get('strategy_returns', pd.Series(dtype=float)).dropna()
        bench_ret = results.get('benchmark_returns', pd.Series(dtype=float)).dropna()

        if strat_ret.empty:
            print("No 'strategy_returns' found or they're empty. Nothing to analyze.")
            return {}

        # Basic daily stats
        mean_ret = strat_ret.mean()

        # T-test vs zero
        t_stat_zero, p_val_zero = ttest_1samp(strat_ret, 0.0)

        # T-test vs bench if present
        t_stat_bench, p_val_bench = (np.nan, np.nan)
        if not bench_ret.empty:
            # Align on common dates if desired. We'll do intersection
            common_dates = strat_ret.index.intersection(bench_ret.index)
            sret = strat_ret.loc[common_dates]
            bret = bench_ret.loc[common_dates]
            if len(sret) > 1 and len(bret) > 1:
                t_stat_bench, p_val_bench = ttest_ind(sret, bret, equal_var=False)

        translation_str = self._build_translation(mean_ret, p_val_zero, p_val_bench)

        return {
            'mean_ret': mean_ret,
            't_stat_vs_zero': t_stat_zero,
            'p_value_vs_zero': p_val_zero,
            't_stat_vs_bench': t_stat_bench,
            'p_value_vs_bench': p_val_bench,
            'translation': translation_str
        }

    def plot(self, results, bins=30, figsize=(7, 10), with_equity_curve=False):
        """
        Plots either a 3-row figure (if with_equity_curve=True)
        or a 2-row figure (if with_equity_curve=False).

        Rows if with_equity_curve=True:
          - row0: equity curve
          - row1: histogram
          - row2: stats table

        Rows if with_equity_curve=False:
          - row0: histogram
          - row1: stats table

        bins : int
            number of bins for the histogram
        figsize : (width, height)
            figure size
        """
        stats_dict = self.analyze(results)
        if not stats_dict:
            return

        # We'll fetch returns for plotting the histogram, and possibly build an equity curve if needed
        strat_ret = results.get('strategy_returns', pd.Series(dtype=float)).dropna()
        bench_ret = results.get('benchmark_returns', pd.Series(dtype=float)).dropna()

        # Decide how many rows
        if with_equity_curve:
            nrows = 3
        else:
            nrows = 2

        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize)

        # In the case of nrows=2, axes might be a single-dimensional array
        # We'll unify the indexing for code clarity:
        if nrows == 2:
            ax_hist = axes[0]
            ax_table = axes[1]
        else:
            # 3 rows => row0 eq, row1 hist, row2 table
            ax_eq = axes[0]
            ax_hist = axes[1]
            ax_table = axes[2]

        fig.suptitle("Student's T-Test", fontsize=14)

        #######################################################################
        # If with_equity_curve => row0 => equity curve
        #######################################################################
        if with_equity_curve:
            if not strat_ret.empty:
                eq_strat = (1.0 + strat_ret).cumprod()
                ax_eq.plot(eq_strat.index, eq_strat.values, label="Strategy")
            if not bench_ret.empty:
                eq_bench = (1.0 + bench_ret).cumprod()
                ax_eq.plot(eq_bench.index, eq_bench.values, label="Benchmark")
            ax_eq.set_title("Equity Curve")
            ax_eq.set_ylabel("Equity")
            ax_eq.grid(True)
            ax_eq.legend(loc='best')

        #######################################################################
        # Next row => histogram of daily returns
        #######################################################################
        if not strat_ret.empty:
            ax_hist.hist(strat_ret, bins=bins, alpha=0.6, label="Strategy")
        if not bench_ret.empty:
            ax_hist.hist(bench_ret, bins=bins, alpha=0.6, label="Benchmark")
        ax_hist.set_title("Daily Returns Distribution")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend(loc='best')
        ax_hist.grid(True)

        #######################################################################
        # Final row => stats table
        #######################################################################
        ax_table.axis('off')

        data_rows = [
            ["Mean Returns", f"{stats_dict['mean_ret']:.4%}"],
            ["t-stat vs Zero", f"{stats_dict['t_stat_vs_zero']:.3f}"],
            ["p-value vs Zero", f"{stats_dict['p_value_vs_zero']:.4f}"],
            ["t-stat vs Bench", f"{stats_dict['t_stat_vs_bench']:.3f}"],
            ["p-value vs Bench", f"{stats_dict['p_value_vs_bench']:.4f}"],
            ["Translation", stats_dict['translation']],
        ]
        table = ax_table.table(
            cellText=data_rows,
            colLabels=["Key", "Value"],
            loc="upper left",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        # highlight
        def highlight_cell(table_obj, row, is_good):
            cell = table_obj.get_celld()[(row, 1)]
            if is_good:
                cell.set_facecolor('lightgreen')
                cell.set_alpha(0.5)
            else:
                cell.set_facecolor('lightcoral')
                cell.set_alpha(0.5)

        def highlight_table(table_obj, stats_d):
            mean_ret = stats_d['mean_ret']
            p_val_zero = stats_d['p_value_vs_zero']
            p_val_bench = stats_d['p_value_vs_bench']

            # row indexes excluding header row=0
            row_mean = 1
            row_tstat_zero = 2
            row_pval_zero = 3
            row_tstat_bench = 4
            row_pval_bench = 5

            # mean > 0 => green
            highlight_cell(table_obj, row_mean, mean_ret > 0)

            # p_val_zero < 0.05 => highlight row2 and row3
            pass_zero = (p_val_zero < 0.05)
            highlight_cell(table_obj, row_tstat_zero, pass_zero)
            highlight_cell(table_obj, row_pval_zero, pass_zero)

            # p_val_bench < 0.05 => highlight row4 and row5
            if np.isnan(p_val_bench):
                pass_bench = False
            else:
                pass_bench = (p_val_bench < 0.05)
            highlight_cell(table_obj, row_tstat_bench, pass_bench)
            highlight_cell(table_obj, row_pval_bench, pass_bench)

        highlight_table(table, stats_dict)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def _build_translation(self, mean_ret, p_val_zero, p_val_bench):
        """
        If no benchmark => 'No benchmark'
        If p_val_bench < 0.05 => 'Significant vs. benchmark'
        else => 'No significance vs. benchmark'
        """
        if np.isnan(p_val_bench):
            return "No benchmark"
        elif p_val_bench < 0.05:
            return "Significant vs. benchmark"
        else:
            return "No significance vs. benchmark"
