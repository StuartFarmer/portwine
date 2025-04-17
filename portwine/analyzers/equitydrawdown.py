import numpy as np
import matplotlib.pyplot as plt
from portwine.analyzers.base import Analyzer


class EquityDrawdownAnalyzer(Analyzer):
    """
    Provides common analysis functionality, including drawdown calculation,
    summary stats, and plotting.
    """

    def compute_drawdown(self, equity_series):
        """
        Computes percentage drawdown for a given equity curve.

        Parameters
        ----------
        equity_series : pd.Series
            The cumulative equity values over time (e.g., starting at 1.0).

        Returns
        -------
        drawdown : pd.Series
            The percentage drawdown at each point in time.
        """
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        return drawdown

    def analyze_returns(self, daily_returns, ann_factor=252):
        """
        Computes summary statistics from daily returns.

        Parameters
        ----------
        daily_returns : pd.Series
            Daily returns of a strategy or benchmark.
        ann_factor : int
            Annualization factor, typically 252 for daily data.

        Returns
        -------
        stats : dict
            {
                'TotalReturn': ...,
                'CAGR': ...,
                'AnnualVol': ...,
                'Sharpe': ...,
                'MaxDrawdown': ...
            }
        """
        dr = daily_returns.dropna()
        if len(dr) < 2:
            return {}

        total_ret = (1 + dr).prod() - 1.0
        n_days = len(dr)
        years = n_days / ann_factor
        cagr = (1 + total_ret) ** (1 / years) - 1.0

        ann_vol = dr.std() * np.sqrt(ann_factor)
        sharpe = cagr / ann_vol if ann_vol > 1e-9 else 0.0

        eq = (1 + dr).cumprod()
        dd = self.compute_drawdown(eq)
        max_dd = dd.min()

        return {
            'TotalReturn': total_ret,
            'CAGR': cagr,
            'AnnualVol': ann_vol,
            'Sharpe': sharpe,
            'MaxDrawdown': max_dd,
        }

    def analyze(self, results, ann_factor=252):
        strategy_stats = self.analyze_returns(results['strategy_returns'], ann_factor)
        benchmark_stats = self.analyze_returns(results['benchmark_returns'], ann_factor)

        return {
            'strategy_stats': strategy_stats,
            'benchmark_stats': benchmark_stats
        }

    def plot(self, results, benchmark_label="Benchmark"):
        """
        Plots the strategy equity curve (and benchmark if given) plus drawdowns.
        Also prints summary stats.

        Parameters
        ----------
        results : dict
            Results from the backtest. Will have signals_df, tickers_returns,
            strategy_returns, benchmark_returns, which are all Pandas DataFrames

        benchmark_label : str
            Label to use for benchmark in plot legend and summary stats.
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

        strategy_equity_curve = (1.0 + results['strategy_returns']).cumprod()
        benchmark_equity_curve = (1.0 + results['benchmark_returns']).cumprod()

        # Plot equity curves with specified colors and line widths
        ax1.plot(
            strategy_equity_curve.index,
            strategy_equity_curve.values,
            label="Strategy",
            color='mediumblue',  # deeper blue
            linewidth=1,  # a bit thicker
            alpha=0.6
        )
        ax1.plot(
            benchmark_equity_curve.index,
            benchmark_equity_curve.values,
            label=benchmark_label,
            color='black',  # black
            linewidth=0.5,  # a bit thinner
            alpha=0.5
        )
        ax1.set_title("Equity Curve (relative, starts at 1.0)")
        ax1.legend(loc='best')
        ax1.grid(True)

        # Fill between the strategy and benchmark lines
        ax1.fill_between(
            strategy_equity_curve.index,
            strategy_equity_curve.values,
            benchmark_equity_curve.values,
            where=(strategy_equity_curve.values >= benchmark_equity_curve.values),
            interpolate=True,
            color='green',
            alpha=0.1
        )
        ax1.fill_between(
            strategy_equity_curve.index,
            strategy_equity_curve.values,
            benchmark_equity_curve.values,
            where=(strategy_equity_curve.values < benchmark_equity_curve.values),
            interpolate=True,
            color='red',
            alpha=0.1
        )

        # Plot drawdowns
        strat_dd = self.compute_drawdown(strategy_equity_curve) * 100.0
        bm_dd = self.compute_drawdown(benchmark_equity_curve) * 100.0

        ax2.plot(
            strat_dd.index,
            strat_dd.values,
            label="Strategy DD (%)",
            color='mediumblue',  # deeper blue
            linewidth=1,  # a bit thicker
            alpha=0.6
        )
        ax2.plot(
            bm_dd.index,
            bm_dd.values,
            label=f"{benchmark_label} DD (%)",
            color='black',  # black
            linewidth=0.5,  # a bit thinner
            alpha=0.5
        )
        ax2.set_title("Drawdown (%)")
        ax2.legend(loc='best')
        ax2.grid(True)

        # Fill between drawdown lines: red where strategy is below, green where strategy is above
        ax2.fill_between(
            strat_dd.index,
            strat_dd.values,
            bm_dd.values,
            where=(strat_dd.values <= bm_dd.values),
            interpolate=True,
            color='red',
            alpha=0.1
        )
        ax2.fill_between(
            strat_dd.index,
            strat_dd.values,
            bm_dd.values,
            where=(strat_dd.values > bm_dd.values),
            interpolate=True,
            color='green',
            alpha=0.1
        )

        plt.tight_layout()
        plt.show()

    def generate_report(self, results, ann_factor=252, benchmark_label="Benchmark"):
        """
        Generates and returns a styled DataFrame with strategy stats, benchmark stats,
        and the percentage difference between them. Highlights rows based on performance
        using pastel colors.

        Parameters
        ----------
        results : dict
            Results from the backtest containing strategy and benchmark returns.
        ann_factor : int
            Annualization factor, typically 252 for daily data.
        benchmark_label : str
            Label to use for the benchmark in the report.

        Returns
        -------
        pandas.io.formats.style.Styler
            Styled DataFrame with metrics as rows and three columns:
            - Strategy values
            - Benchmark values
            - Percentage difference (highlighted based on performance)
        """
        import pandas as pd
        import numpy as np

        stats = self.analyze(results, ann_factor)

        strategy_stats = stats['strategy_stats']
        benchmark_stats = stats['benchmark_stats']

        # Create DataFrame with metrics as rows using raw numeric values
        df = pd.DataFrame({
            'Strategy': pd.Series(strategy_stats),
            benchmark_label: pd.Series(benchmark_stats)
        })

        # Calculate percentage difference (Strategy vs. Benchmark)
        diff = []
        for metric in df.index:
            strat_val = df.loc[metric, 'Strategy']
            bench_val = df.loc[metric, benchmark_label]

            if isinstance(strat_val, (int, float)) and isinstance(bench_val, (int, float)):
                if abs(bench_val) > 1e-15:
                    diff_val = (strat_val - bench_val) / abs(bench_val)
                else:
                    diff_val = float('nan')  # Use NaN for division by zero
            else:
                diff_val = float('nan')  # Use NaN for non-numeric values

            diff.append(diff_val)

        df['Difference (%)'] = [d * 100 for d in diff]  # Convert to percentage

        # Create a copy of the DataFrame with formatted values for display
        display_df = df.copy()

        # Format values
        for col in ['Strategy', benchmark_label]:
            for idx in display_df.index:
                val = display_df.loc[idx, col]
                if isinstance(val, (int, float)):
                    if idx in ["CAGR", "AnnualVol", "MaxDrawdown", "TotalReturn"]:
                        display_df.loc[idx, col] = f"{val:.2%}"
                    elif idx == "Sharpe":
                        display_df.loc[idx, col] = f"{val:.2f}"

        # Format difference column
        display_df['Difference (%)'] = display_df['Difference (%)'].apply(
            lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
        )

        # Create a styled version of the DataFrame
        styled_df = display_df.style

        # Define styling function for conditional formatting with pastel colors
        def highlight_difference(row):
            styles = [''] * 3  # No highlighting by default

            # Get the difference value from the original DataFrame to avoid string parsing
            metric = row.name
            if metric in df.index:
                diff_val = df.loc[metric, 'Difference (%)']

                if pd.isna(diff_val):
                    return styles

                # For volatility, negative difference is good
                if metric == 'AnnualVol':
                    if diff_val < 0:
                        styles[2] = 'background-color: #d6f5d6'  # Pastel green
                    elif diff_val > 0:
                        styles[2] = 'background-color: #f8d7da'  # Pastel red
                # For all other metrics, positive difference is good
                else:
                    if diff_val > 0:
                        styles[2] = 'background-color: #d6f5d6'  # Pastel green
                    elif diff_val < 0:
                        styles[2] = 'background-color: #f8d7da'  # Pastel red

            return styles

        # Apply the styling
        styled_df = styled_df.apply(highlight_difference, axis=1)

        # Add a title to the styled DataFrame
        styled_df = styled_df.set_caption("Performance Comparison: Strategy vs. Benchmark")

        return styled_df