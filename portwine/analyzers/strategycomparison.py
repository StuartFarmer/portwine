import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from portwine.analyzers.base import Analyzer


class StrategyComparisonAnalyzer(Analyzer):
    """
    Compares two backtest result dictionaries by:
      1) Computing a suite of performance stats for each (CAGR, Sharpe, Sortino, etc.)
      2) Running difference tests on daily returns (t-test)
      3) Computing rolling correlation, alpha, beta
      4) Plotting:
         - equity curves on top (with fill in between)
         - rolling correlation, alpha, and beta in three subplots below
    """

    def __init__(self, rolling_window=60, ann_factor=252, alpha=0.05):
        super().__init__()
        self.rolling_window = rolling_window
        self.ann_factor = ann_factor
        self.alpha = alpha
        self.analysis_results = {}

    def analyze(self, results, comparison_results):
        """
        Analyzes two sets of backtester results, each containing 'strategy_returns'.

        Returns a dictionary of comparative metrics, stored in self.analysis_results.
        """
        daily_returns_a = results['strategy_returns']
        daily_returns_b = comparison_results['strategy_returns']

        # 1) Compute stats for each strategy individually
        stats_a = self._compute_strategy_stats(daily_returns_a, self.ann_factor)
        stats_b = self._compute_strategy_stats(daily_returns_b, self.ann_factor)

        # 2) Align daily returns by common dates and run difference test
        common_idx = daily_returns_a.dropna().index.intersection(daily_returns_b.dropna().index)
        dr_a = daily_returns_a.loc[common_idx]
        dr_b = daily_returns_b.loc[common_idx]
        t_stat, p_val = stats.ttest_ind(dr_a, dr_b, equal_var=False)

        difference_tests = {
            'MeanReturns_A': dr_a.mean(),
            'MeanReturns_B': dr_b.mean(),
            'MeanDifference': dr_a.mean() - dr_b.mean(),
            't_stat': t_stat,
            'p_value': p_val,
            'significant_at_alpha': (p_val < self.alpha)
        }

        # 3) Rolling correlation
        rolling_corr = dr_a.rolling(self.rolling_window).corr(dr_b)

        # 4) Rolling alpha/beta
        rolling_alpha_beta = self._compute_rolling_alpha_beta(dr_a, dr_b, self.rolling_window)

        # Store and return results
        self.analysis_results = {
            'stats_A': stats_a,
            'stats_B': stats_b,
            'difference_tests': difference_tests,
            'rolling_corr': rolling_corr,
            'rolling_alpha_beta': rolling_alpha_beta
        }
        return self.analysis_results

    def plot(self, results, comparison_results,
             label_main="Strategy", label_compare="Benchmark"):
        """
        Creates a 4-row figure:
          - Top row: equity curves + fill (green if main>compare, red otherwise)
          - Next 3 rows: rolling correlation, alpha, beta

        Also includes a second y-axis in the top subplot to show the percentage
        difference between strategy and benchmark, with a legend entry.
        """
        # --- 1) If analyze() wasn't called before, do so here ---
        if not self.analysis_results:
            self.analyze(results, comparison_results)

        # --- 2) Extract data for plotting ---
        equity_main = (1.0 + results['strategy_returns'].fillna(0)).cumprod()
        equity_compare = (1.0 + comparison_results['strategy_returns'].fillna(0)).cumprod()

        rolling_corr = self.analysis_results['rolling_corr']
        alpha_series = self.analysis_results['rolling_alpha_beta']['alpha']
        beta_series = self.analysis_results['rolling_alpha_beta']['beta']

        # --- 3) Create figure with 4 subplots ---
        fig, axes = plt.subplots(nrows=4, ncols=1,
                                 figsize=(12, 8),
                                 sharex=True,
                                 gridspec_kw={'height_ratios': [6, 1, 1, 1]})
        ax_main = axes[0]
        ax_corr = axes[1]
        ax_alpha = axes[2]
        ax_beta = axes[3]

        # --- 4) Plot the main equity curves in the top subplot ---
        line_main, = ax_main.plot(equity_main.index, equity_main.values,
                                  label=label_main, color='k')
        line_compare, = ax_main.plot(equity_compare.index, equity_compare.values,
                                     label=label_compare, alpha=0.8, color='k',
                                     linestyle='dashed', linewidth=1)

        ax_main.set_title("Strategy Comparison: Equity Curves")
        ax_main.set_ylabel("Cumulative Equity")
        ax_main.grid(True)

        # ---- 4A) Fill area in between curves ----
        ax_main.fill_between(
            equity_main.index,
            equity_main.values,
            equity_compare.values,
            where=(equity_main.values >= equity_compare.values),
            color='green', alpha=0.2, interpolate=True
        )
        ax_main.fill_between(
            equity_main.index,
            equity_main.values,
            equity_compare.values,
            where=(equity_main.values < equity_compare.values),
            color='red', alpha=0.2, interpolate=True
        )

        # ---- 4B) Compute and plot the % difference on a second y-axis ----
        ax_diff = ax_main.twinx()
        pct_diff = (equity_main / equity_compare - 1.0) * 100.0
        line_diff, = ax_diff.plot(pct_diff.index, pct_diff.values,
                                  label="Pct Diff vs. Benchmark", color='b', linewidth=0.5)
        ax_diff.set_ylabel("Difference (%)")

        # ---- Create a single legend for both lines on the top subplot ----
        lines_main, labels_main = ax_main.get_legend_handles_labels()
        lines_diff, labels_diff = ax_diff.get_legend_handles_labels()
        ax_main.legend(lines_main + [line_diff], labels_main + [labels_diff[-1]], loc='best')

        # --- 5) Rolling correlation in the 2nd subplot (RED) ---
        ax_corr.plot(rolling_corr.index, rolling_corr.values,
                     label='Rolling Correlation', color='red')
        ax_corr.set_ylabel("Corr")
        ax_corr.legend(loc='best')
        ax_corr.grid(True)

        # --- 6) Rolling alpha in the 3rd subplot (BLUE) ---
        ax_alpha.plot(alpha_series.index, alpha_series.values,
                      label='Rolling Alpha', color='blue')
        ax_alpha.set_ylabel("Alpha")
        ax_alpha.legend(loc='best')
        ax_alpha.grid(True)

        # --- 7) Rolling beta in the 4th subplot (GREEN) ---
        ax_beta.plot(beta_series.index, beta_series.values,
                     label='Rolling Beta', color='green')
        ax_beta.set_ylabel("Beta")
        ax_beta.legend(loc='best')
        ax_beta.grid(True)

        # --- 8) Final styling and show ---
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------------
    # HELPER METHODS
    # ------------------------------------------------------------------------
    def _compute_strategy_stats(self, daily_returns, ann_factor=252):
        """
        Computes a set of performance stats for a single strategy's daily returns.
        """
        dr = daily_returns.dropna()
        if dr.empty:
            return {
                'TotalReturn': np.nan,
                'CAGR': np.nan,
                'AnnualVol': np.nan,
                'Sharpe': np.nan,
                'Sortino': np.nan,
                'MaxDrawdown': np.nan,
                'Calmar': np.nan
            }

        total_return = (1 + dr).prod() - 1.0
        n_days = len(dr)
        years = n_days / ann_factor

        cagr = (1 + total_return) ** (1 / years) - 1.0 if years > 0 else np.nan
        ann_vol = dr.std() * np.sqrt(ann_factor)

        if ann_vol > 1e-9:
            sharpe = cagr / ann_vol
        else:
            sharpe = np.nan

        negative_returns = dr[dr < 0]
        neg_vol = negative_returns.std() * np.sqrt(ann_factor) if len(negative_returns) > 1 else np.nan
        if neg_vol and neg_vol > 1e-9:
            sortino = cagr / neg_vol
        else:
            sortino = np.nan

        equity = (1 + dr).cumprod()
        running_max = equity.cummax()
        dd_series = (equity - running_max) / running_max
        max_dd = dd_series.min()
        calmar = cagr / abs(max_dd) if (max_dd < 0) else np.nan

        return {
            'TotalReturn': total_return,
            'CAGR': cagr,
            'AnnualVol': ann_vol,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'MaxDrawdown': max_dd,
            'Calmar': calmar
        }

    def _compute_rolling_alpha_beta(self, dr_a, dr_b, window=60):
        """
        Computes rolling alpha/beta by regressing A on B over a rolling window:
        A_t = alpha + beta * B_t.

        Returns a DataFrame with columns ['alpha', 'beta'].
        """
        alpha_list = []
        beta_list = []
        idx_list = dr_a.index

        for i in range(len(idx_list)):
            if i < window:
                alpha_list.append(np.nan)
                beta_list.append(np.nan)
            else:
                window_a = dr_a.iloc[i - window + 1: i + 1]
                window_b = dr_b.iloc[i - window + 1: i + 1]
                var_b = np.var(window_b, ddof=1)
                if var_b < 1e-12:
                    alpha_list.append(np.nan)
                    beta_list.append(np.nan)
                else:
                    cov_ab = np.cov(window_a, window_b, ddof=1)[0, 1]
                    beta_i = cov_ab / var_b
                    alpha_i = window_a.mean() - beta_i * window_b.mean()
                    alpha_list.append(alpha_i)
                    beta_list.append(beta_i)

        df = pd.DataFrame({'alpha': alpha_list, 'beta': beta_list}, index=dr_a.index)
        return df
