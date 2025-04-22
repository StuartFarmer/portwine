import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
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

        Parameters
        ----------
        results : dict
            Dictionary containing strategy_returns for the main strategy.
        comparison_results : dict
            Dictionary containing strategy_returns for the comparison strategy.

        Returns
        -------
        dict
            Dictionary of comparative metrics, stored in self.analysis_results.
        """
        daily_returns_a = results['strategy_returns']
        daily_returns_b = comparison_results['strategy_returns']

        # 1) Compute stats for each strategy individually (using raw returns)
        stats_a = self._compute_strategy_stats(daily_returns_a, self.ann_factor)
        stats_b = self._compute_strategy_stats(daily_returns_b, self.ann_factor)

        # 2) Align returns by date and fill missing values with 0
        # Create a common index that starts at the later of the two start dates
        start_a = daily_returns_a.index.min()
        start_b = daily_returns_b.index.min()
        common_start = max(start_a, start_b)

        # Filter returns to start at the common start date
        dr_a = daily_returns_a[daily_returns_a.index >= common_start].copy()
        dr_b = daily_returns_b[daily_returns_b.index >= common_start].copy()

        # Create a union of all dates after the common start
        all_dates = dr_a.index.union(dr_b.index)

        # Reindex both series to include all dates and fill missing values with 0
        dr_a = dr_a.reindex(all_dates, fill_value=0)
        dr_b = dr_b.reindex(all_dates, fill_value=0)

        # Run difference test on aligned returns
        t_stat, p_val = stats.ttest_ind(dr_a, dr_b, equal_var=False)

        difference_tests = {
            'MeanReturns_A': dr_a.mean(),
            'MeanReturns_B': dr_b.mean(),
            'MeanDifference': dr_a.mean() - dr_b.mean(),
            't_stat': t_stat,
            'p_value': p_val,
            'significant_at_alpha': (p_val < self.alpha)
        }

        # 3) Rolling correlation (using aligned returns)
        rolling_corr = dr_a.rolling(self.rolling_window).corr(dr_b)

        # 4) Rolling alpha/beta (using aligned returns)
        rolling_alpha_beta = self._compute_rolling_alpha_beta(dr_a, dr_b, self.rolling_window)

        # Store and return results
        self.analysis_results = {
            'stats_A': stats_a,
            'stats_B': stats_b,
            'difference_tests': difference_tests,
            'rolling_corr': rolling_corr,
            'rolling_alpha_beta': rolling_alpha_beta,
            'aligned_returns_A': dr_a,
            'aligned_returns_B': dr_b
        }
        return self.analysis_results

    def plot(self, results, comparison_results=None, label_main="Strategy", label_compare=None, tearsheet=False,
             benchmark=False):
        """
        Creates a figure with equity curves, rolling statistics, and optional metrics.

        Parameters
        ----------
        results : dict
            Results dictionary for the main strategy.
        comparison_results : dict, optional
            Results dictionary for the comparison strategy. If None, uses benchmark from results.
        label_main : str, default="Strategy"
            Label for the main strategy.
        label_compare : str, optional
            Label for the comparison strategy. If None, automatically determines label.
        tearsheet : bool, default=False
            If True, adds a performance metrics table to the plot.
        benchmark : bool, default=False
            If True and comparison_results is provided, also plots the benchmark from results.
        """
        # If comparison_results not provided, use benchmark from results
        if comparison_results is None:
            # Create a comparison results dict with benchmark returns from the first results
            comparison_results = {
                'strategy_returns': results['benchmark_returns']
            }

            # Auto-set label if not explicitly provided
            if label_compare is None:
                label_compare = "Benchmark"

            # Set benchmark flag to False since we're already comparing against the benchmark
            benchmark = False
        else:
            # When comparing two strategies, default to "Baseline Strategy" if not specified
            if label_compare is None:
                label_compare = "Baseline Strategy"
        # --- 1) If analyze() wasn't called before, do so here ---
        if not self.analysis_results:
            self.analyze(results, comparison_results)

        # --- 2) Extract data for plotting ---
        aligned_returns_a = self.analysis_results['aligned_returns_A']
        aligned_returns_b = self.analysis_results['aligned_returns_B']

        # Build equity curves from aligned returns
        equity_main = (1.0 + aligned_returns_a).cumprod()
        equity_compare = (1.0 + aligned_returns_b).cumprod()

        # If benchmark flag is True and we're comparing two strategies, also get benchmark data
        if benchmark and comparison_results.get('strategy_returns') is not results.get('benchmark_returns'):
            # Get benchmark returns
            benchmark_returns = results.get('benchmark_returns')
            if benchmark_returns is not None:
                # Create benchmark equity curve starting from the common date
                common_start = max(aligned_returns_a.index[0], aligned_returns_b.index[0], benchmark_returns.index[0])
                benchmark_returns_aligned = benchmark_returns[benchmark_returns.index >= common_start]

                # Fill in any missing dates with zeros to align with the two strategies
                all_dates = aligned_returns_a.index.union(aligned_returns_b.index)
                benchmark_returns_aligned = benchmark_returns_aligned.reindex(
                    all_dates, fill_value=0).loc[aligned_returns_a.index[0]:]

                # Create equity curve
                equity_benchmark = (1.0 + benchmark_returns_aligned).cumprod()
            else:
                benchmark = False  # Disable benchmark if data is not available
        else:
            benchmark = False  # Disable benchmark flag if not needed

        rolling_corr = self.analysis_results['rolling_corr']
        alpha_series = self.analysis_results['rolling_alpha_beta']['alpha']
        beta_series = self.analysis_results['rolling_alpha_beta']['beta']

        # --- 3) Create figure with appropriate subplots ---
        if tearsheet:
            fig = plt.figure(figsize=(12, 14))
            gs = GridSpec(5, 1, figure=fig, height_ratios=[6, 1, 1, 1, 4])

            ax_main = fig.add_subplot(gs[0])
            ax_corr = fig.add_subplot(gs[1], sharex=ax_main)
            ax_alpha = fig.add_subplot(gs[2], sharex=ax_main)
            ax_beta = fig.add_subplot(gs[3], sharex=ax_main)
            ax_table = fig.add_subplot(gs[4])
        else:
            fig, axes = plt.subplots(nrows=4, ncols=1,
                                     figsize=(12, 8),
                                     sharex=True,
                                     gridspec_kw={'height_ratios': [6, 1, 1, 1]})
            ax_main = axes[0]
            ax_corr = axes[1]
            ax_alpha = axes[2]
            ax_beta = axes[3]

        # --- 4) Plot the main equity curves ---
        line_main, = ax_main.plot(equity_main.index, equity_main.values,
                                  label=label_main, color='k')
        line_compare, = ax_main.plot(equity_compare.index, equity_compare.values,
                                     label=label_compare, alpha=0.8, color='k',
                                     linestyle='dashed', linewidth=1)

        # Plot benchmark if requested
        if benchmark:
            line_benchmark, = ax_main.plot(equity_benchmark.index, equity_benchmark.values,
                                           label="Benchmark", alpha=0.7, color='gray',
                                           linestyle='dotted', linewidth=1)

        ax_main.set_title("Strategy Comparison: Equity Curves")
        ax_main.set_ylabel("Cumulative Equity")
        ax_main.grid(True)

        # Fill area between curves
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

        # Plot percentage difference on a second y-axis
        ax_diff = ax_main.twinx()
        pct_diff = (equity_main / equity_compare - 1.0) * 100.0

        # Adjust the label based on what we're comparing against
        if comparison_results.get('strategy_returns') is results.get('benchmark_returns'):
            diff_label = "Pct Diff vs. Benchmark"
        else:
            diff_label = "Pct Diff vs. Baseline"

        line_diff, = ax_diff.plot(pct_diff.index, pct_diff.values,
                                  label=diff_label, color='b', linewidth=0.5)
        ax_diff.set_ylabel("Difference (%)")

        # Create a single legend
        lines_main, labels_main = ax_main.get_legend_handles_labels()
        lines_diff, labels_diff = ax_diff.get_legend_handles_labels()

        # Combine all lines for the legend
        all_lines = lines_main + [line_diff]
        all_labels = labels_main + [labels_diff[-1]]

        ax_main.legend(all_lines, all_labels, loc='best')

        # --- 5) Plot correlation, alpha, beta ---
        ax_corr.plot(rolling_corr.index, rolling_corr.values,
                     label='Rolling Correlation', color='red')
        ax_corr.set_ylabel("Corr")
        ax_corr.legend(loc='best')
        ax_corr.grid(True)

        ax_alpha.plot(alpha_series.index, alpha_series.values,
                      label='Rolling Alpha', color='blue')
        ax_alpha.set_ylabel("Alpha")
        ax_alpha.legend(loc='best')
        ax_alpha.grid(True)

        ax_beta.plot(beta_series.index, beta_series.values,
                     label='Rolling Beta', color='green')
        ax_beta.set_ylabel("Beta")
        ax_beta.legend(loc='best')
        ax_beta.grid(True)

        # --- 6) Create performance metrics table if tearsheet=True ---
        if tearsheet:
            ax_table.axis('tight')
            ax_table.axis('off')

            # Calculate strategy stats
            # Use original returns for individual strategy statistics
            stats_a = self._compute_strategy_stats(results['strategy_returns'], self.ann_factor)
            stats_b = self._compute_strategy_stats(comparison_results['strategy_returns'], self.ann_factor)

            # Add additional risk metrics
            # -- For Strategy A
            eq_a = (1 + results['strategy_returns']).cumprod()
            dd_a = self._compute_drawdown(eq_a)
            max_dd_a = dd_a.min()

            dr_a = results['strategy_returns'].dropna()
            neg_returns_a = dr_a[dr_a < 0]
            neg_vol_a = neg_returns_a.std() * np.sqrt(self.ann_factor) if len(neg_returns_a) > 1 else np.nan
            sortino_a = stats_a['CAGR'] / neg_vol_a if neg_vol_a and neg_vol_a > 1e-9 else np.nan
            calmar_a = stats_a['CAGR'] / abs(max_dd_a) if (max_dd_a < 0) else np.nan

            # -- For Strategy B
            eq_b = (1 + comparison_results['strategy_returns']).cumprod()
            dd_b = self._compute_drawdown(eq_b)
            max_dd_b = dd_b.min()

            dr_b = comparison_results['strategy_returns'].dropna()
            neg_returns_b = dr_b[dr_b < 0]
            neg_vol_b = neg_returns_b.std() * np.sqrt(self.ann_factor) if len(neg_returns_b) > 1 else np.nan
            sortino_b = stats_b['CAGR'] / neg_vol_b if neg_vol_b and neg_vol_b > 1e-9 else np.nan
            calmar_b = stats_b['CAGR'] / abs(max_dd_b) if (max_dd_b < 0) else np.nan

            # Dictionary of metrics indicating if positive difference is good
            positive_is_good = {
                'TotalReturn': True,
                'CAGR': True,
                'AnnualVol': False,
                'Sharpe': True,
                'MaxDrawdown': True,  # Less negative is better
                'Sortino': True,
                'Calmar': True
            }

            # Prepare table data
            table_data = []
            cell_colors = []  # To store cell colors for each row

            # Add header
            table_data.append(["Metric", label_main, label_compare, "Difference"])
            cell_colors.append([None, None, None, None])  # No special color for header

            # Add performance metrics
            table_data.append(["Performance Metrics", "", "", ""])
            cell_colors.append([None, None, None, None])  # No special color for section header

            metrics_to_display = [
                ('TotalReturn', 'Total Return', '{:.2%}'),
                ('CAGR', 'CAGR', '{:.2%}'),
                ('AnnualVol', 'Annual Volatility', '{:.2%}'),
                ('Sharpe', 'Sharpe Ratio', '{:.2f}'),
                ('MaxDrawdown', 'Max Drawdown', '{:.2%}')
            ]

            for key, display_name, format_str in metrics_to_display:
                if key == 'MaxDrawdown':
                    strat_val = max_dd_a
                    bench_val = max_dd_b
                else:
                    strat_val = stats_a.get(key, np.nan)
                    bench_val = stats_b.get(key, np.nan)

                if np.isfinite(strat_val) and np.isfinite(bench_val):
                    diff = strat_val - bench_val

                    strat_text = format_str.format(strat_val)
                    bench_text = format_str.format(bench_val)
                    # Add + sign for positive differences
                    if diff > 0:
                        diff_text = "+" + format_str.format(diff)
                    else:
                        diff_text = format_str.format(diff)

                    # Determine if the difference is good or bad
                    is_good = (diff > 0 and positive_is_good.get(key, True)) or (
                                diff < 0 and not positive_is_good.get(key, True))
                    cell_color = '#d8f3dc' if is_good else '#ffcccb' if diff != 0 else None  # Pastel green or red
                else:
                    strat_text = "N/A" if not np.isfinite(strat_val) else format_str.format(strat_val)
                    bench_text = "N/A" if not np.isfinite(bench_val) else format_str.format(bench_val)
                    diff_text = "N/A"
                    cell_color = None

                table_data.append([display_name, strat_text, bench_text, diff_text])
                cell_colors.append([None, None, None, cell_color])

            # Add risk metrics
            table_data.append(["Risk Metrics", "", "", ""])
            cell_colors.append([None, None, None, None])

            risk_metrics_to_display = [
                ('Sortino', 'Sortino Ratio', '{:.2f}', sortino_a, sortino_b),
                ('Calmar', 'Calmar Ratio', '{:.2f}', calmar_a, calmar_b)
            ]

            for key, display_name, format_str, strat_val, bench_val in risk_metrics_to_display:
                if np.isfinite(strat_val) and np.isfinite(bench_val):
                    diff = strat_val - bench_val

                    strat_text = format_str.format(strat_val)
                    bench_text = format_str.format(bench_val)
                    diff_text = format_str.format(diff)

                    # Determine if the difference is good or bad
                    is_good = (diff > 0 and positive_is_good.get(key, True)) or (
                                diff < 0 and not positive_is_good.get(key, True))
                    cell_color = '#d8f3dc' if is_good else '#ffcccb' if diff != 0 else None
                else:
                    strat_text = "N/A" if not np.isfinite(strat_val) else format_str.format(strat_val)
                    bench_text = "N/A" if not np.isfinite(bench_val) else format_str.format(bench_val)
                    diff_text = "N/A"
                    cell_color = None

                table_data.append([display_name, strat_text, bench_text, diff_text])
                cell_colors.append([None, None, None, cell_color])

            # Add comparative metrics from the analysis
            table_data.append(["Comparative Metrics", "", "", ""])
            cell_colors.append([None, None, None, None])

            difference_tests = self.analysis_results.get('difference_tests', {})

            if difference_tests:
                mean_diff = difference_tests.get('MeanDifference', np.nan)
                t_stat = difference_tests.get('t_stat', np.nan)
                p_value = difference_tests.get('p_value', np.nan)
                significant = difference_tests.get('significant_at_alpha', False)

                if np.isfinite(mean_diff):
                    if mean_diff > 0:
                        mean_diff_text = "+{:.4%}".format(mean_diff)
                    else:
                        mean_diff_text = "{:.4%}".format(mean_diff)

                    is_good = mean_diff > 0
                    cell_color = '#d8f3dc' if is_good else '#ffcccb' if mean_diff != 0 else None

                    table_data.append(["Mean Daily Return Diff.", mean_diff_text, "", ""])
                    cell_colors.append([None, cell_color, None, None])

                if np.isfinite(t_stat) and np.isfinite(p_value):
                    t_stat_text = "{:.2f}".format(t_stat)
                    p_value_text = "{:.4f}".format(p_value)
                    sig_text = "Yes" if significant else "No"

                    table_data.append(["t-statistic", t_stat_text, "", ""])
                    cell_colors.append([None, None, None, None])

                    table_data.append(["p-value", p_value_text, "", ""])
                    cell_colors.append([None, None, None, None])

                    table_data.append(["Statistically Significant", sig_text, "", ""])
                    cell_colors.append([None, None, None, None])

            # Create the table
            table = ax_table.table(
                cellText=table_data,
                cellLoc='left',
                loc='center',
                bbox=[0.05, 0.05, 0.9, 0.9]  # Adjust as needed
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)

            # Import FontProperties for bold text
            from matplotlib.font_manager import FontProperties

            # Set cell properties including colors
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('darkblue')
                elif j == 0:  # Metric column
                    cell.set_text_props(weight='bold')

                # Apply custom cell colors
                if j == 3 and i > 0 and i < len(cell_colors) and cell_colors[i][j] is not None:
                    cell.set_facecolor(cell_colors[i][j])
                elif j == 1 and i > 0 and i < len(cell_colors) and cell_colors[i][j] is not None:
                    cell.set_facecolor(cell_colors[i][j])

                # Category headers
                if j == 0 and table_data[i][j] in [
                    "Performance Metrics",
                    "Risk Metrics",
                    "Comparative Metrics"
                ]:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('gray')

                    # Make category headers span all columns
                    for col in range(1, 4):
                        table[i, col].set_text_props(weight='bold', color='white')
                        table[i, col].set_facecolor('gray')

        # Show the plot
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

    def _compute_drawdown(self, equity_series):
        """
        Computes percentage drawdown for a given equity curve.
        """
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        return drawdown