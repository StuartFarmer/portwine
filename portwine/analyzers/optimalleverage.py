import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portwine.analyzers.base import Analyzer


def calculate_integral_drawdown(equity_series):
    """
    Calculate the integrated drawdown (area under the absolute drawdown curve)
    for a given equity series.

    Parameters
    ----------
    equity_series : pd.Series
        The cumulative equity curve.

    Returns
    -------
    float
        Integrated drawdown.
    """
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    abs_dd = -drawdown.clip(upper=0)
    if isinstance(equity_series.index, pd.DatetimeIndex):
        # Use days difference for integration.
        x = (equity_series.index - equity_series.index[0]).days
        integral = np.trapezoid(abs_dd, x=x)
    else:
        integral = np.trapezoid(abs_dd, dx=1)
    return integral


def leveraged_integral_drawdown(strategy_returns, leverage=1.0):
    """
    Computes the integrated drawdown for a strategy after applying the leverage multiplier.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy returns.
    leverage : float
        The leverage multiplier.

    Returns
    -------
    float
        Integrated drawdown for the leveraged equity curve.
    """
    leveraged_returns = strategy_returns * leverage
    equity_curve = (1 + leveraged_returns).cumprod()
    return calculate_integral_drawdown(equity_curve)


class OptimalLeverageAnalyzer(Analyzer):
    """
    This Analyzer produces a three-panel figure:

      Panel 1: Equity Curves
         - Plots the benchmark equity curve and the leveraged strategy equity curve (using the optimal leverage)
           in a style similar to the EquityDrawdownAnalyzer.

      Panel 2: Drawdown
         - Plots the percentage drawdown curves.
           **Important:** The drawdown is now computed from the leveraged strategy equity curve,
           so you can see the effect of applying leverage.

      Panel 3: Integrated Drawdown vs. Leverage
         - Plots the integrated drawdown of the strategy as a function of leverage (using a grid search)
           along with a horizontal line for the benchmark integrated drawdown.
         - The x-axis for this panel is the leverage multiplier (not linked to dates).
    """

    def __init__(self, start_leverage=1.0, end_leverage=3.0, step=0.01, benchmark_label="Benchmark"):
        self.start_leverage = start_leverage
        self.end_leverage = end_leverage
        self.step = step
        self.benchmark_label = benchmark_label

    def compute_drawdown(self, equity_series):
        """
        Computes the percentage drawdown of an equity series.

        Parameters
        ----------
        equity_series : pd.Series
            The cumulative equity curve.

        Returns
        -------
        pd.Series
            The percentage drawdown.
        """
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        return drawdown

    def _grid_search_leverage(self, strategy_returns, benchmark_returns):
        """
        Computes integrated drawdown for the leveraged strategy over a grid of leverage values,
        and compares it with the benchmark integrated drawdown.

        Returns
        -------
        dict
            Contains:
              - 'leverage_values': np.array of leverage multipliers.
              - 'leveraged_ids': array of integrated drawdown for each leverage.
              - 'benchmark_id': Integrated drawdown for the benchmark.
              - 'optimal_leverage': The leverage value that minimizes the difference.
              - 'optimal_strategy_id': The integrated drawdown at the optimal leverage.
        """
        # Benchmark equity curve and integrated drawdown.
        benchmark_equity = (1 + benchmark_returns).cumprod()
        benchmark_id = calculate_integral_drawdown(benchmark_equity)

        leverage_values = np.arange(self.start_leverage, self.end_leverage + self.step, self.step)
        leveraged_ids = []
        for L in leverage_values:
            L_returns = strategy_returns * L
            equity_curve = (1 + L_returns).cumprod()
            L_id = calculate_integral_drawdown(equity_curve)
            leveraged_ids.append(L_id)
        leveraged_ids = np.array(leveraged_ids)

        diffs = np.abs(leveraged_ids - benchmark_id)
        opt_index = np.argmin(diffs)
        optimal_leverage = leverage_values[opt_index]
        optimal_strategy_id = leveraged_ids[opt_index]

        return {
            'leverage_values': leverage_values,
            'leveraged_ids': leveraged_ids,
            'benchmark_id': benchmark_id,
            'optimal_leverage': optimal_leverage,
            'optimal_strategy_id': optimal_strategy_id
        }

    def analyze(self, results, ann_factor=252):
        # This Analyzer mainly works in the plotting routine.
        return results

    def plot(self, results, **kwargs):
        """
        Produces a three-panel figure.

        Panel 1: Equity Curves (date x-axis)
          - Plots the benchmark equity curve and the leveraged strategy equity curve (using optimal leverage).

        Panel 2: Drawdown (date x-axis)
          - Plots the percentage drawdown curves computed from the leveraged strategy equity curve.
          - (For reference, benchmark drawdown is also plotted.)

        Panel 3: Integrated Drawdown vs. Leverage (leverage x-axis)
          - Plots the integrated drawdown for the leveraged strategy for a grid of leverage values, along with a horizontal line for the benchmark.
          - The optimal leverage is highlighted.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        strategy_returns = results.get('strategy_returns')
        benchmark_returns = results.get('benchmark_returns')
        if strategy_returns is None or benchmark_returns is None:
            raise ValueError("Results must contain both 'strategy_returns' and 'benchmark_returns'.")

        # Compute equity curves.
        # For Panel 1, we'll show the benchmark and the leveraged strategy (using optimal leverage).
        grid_res = self._grid_search_leverage(strategy_returns, benchmark_returns)
        optimal_leverage = grid_res['optimal_leverage']
        leveraged_returns = strategy_returns * optimal_leverage
        leveraged_equity = (1 + leveraged_returns).cumprod()
        benchmark_equity = (1 + benchmark_returns).cumprod()

        # Compute drawdown curves.
        # Panel 2: Drawdown from the leveraged strategy equity curve.
        leveraged_dd = self.compute_drawdown(leveraged_equity) * 100.0
        benchmark_dd = self.compute_drawdown(benchmark_equity) * 100.0

        # Create three separate panels (do not share x-axis across all).
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=False)

        # Panel 1: Equity Curves (date x-axis)
        ax1.plot(benchmark_equity.index, benchmark_equity.values,
                 label=self.benchmark_label, color='black', linewidth=0.5, alpha=0.5)
        ax1.plot(leveraged_equity.index, leveraged_equity.values,
                 label=f"Strategy (Leveraged: {optimal_leverage:.2f}x)", linestyle='-.', color='darkorange',
                 linewidth=1.5, alpha=0.8)
        ax1.set_title("Equity Curves (Relative, starts at 1.0)")
        ax1.legend(loc="best")
        ax1.grid(True)
        ax1.fill_between(benchmark_equity.index, leveraged_equity.values, benchmark_equity.values,
                         where=(leveraged_equity.values >= benchmark_equity.values),
                         interpolate=True, color='green', alpha=0.1)
        ax1.fill_between(benchmark_equity.index, leveraged_equity.values, benchmark_equity.values,
                         where=(leveraged_equity.values < benchmark_equity.values),
                         interpolate=True, color='red', alpha=0.1)

        # Panel 2: Drawdown Curves (date x-axis) for leveraged strategy.
        ax2.plot(leveraged_equity.index, leveraged_dd,
                 label="Strategy Drawdown (%)", color='darkorange', linewidth=1.5, alpha=0.8)
        ax2.plot(benchmark_equity.index, benchmark_dd,
                 label=f"{self.benchmark_label} Drawdown (%)", color='black', linewidth=0.5, alpha=0.5)
        ax2.set_title("Drawdown (%) (Leveraged Strategy)")
        ax2.legend(loc="best")
        ax2.grid(True)
        ax2.fill_between(leveraged_equity.index, leveraged_dd, benchmark_dd,
                         where=(leveraged_dd <= benchmark_dd),
                         interpolate=True, color='red', alpha=0.1)
        ax2.fill_between(leveraged_equity.index, leveraged_dd, benchmark_dd,
                         where=(leveraged_dd > benchmark_dd),
                         interpolate=True, color='green', alpha=0.1)

        # Panel 3: Integrated Drawdown vs. Leverage (leverage x-axis, independent axis)
        ax3.plot(grid_res['leverage_values'], grid_res['leveraged_ids'],
                 label="Strategy Integrated Drawdown", color='blue', linewidth=2)
        ax3.axhline(grid_res['benchmark_id'], color='red', linestyle="--", linewidth=2,
                    label="Benchmark Integrated Drawdown")
        ax3.plot(grid_res['optimal_leverage'], grid_res['optimal_strategy_id'],
                 marker='o', markersize=8, color='black',
                 label=f"Optimal Leverage: {grid_res['optimal_leverage']:.2f}x")
        ax3.set_title("Integrated Drawdown vs Leverage")
        ax3.set_xlabel("Leverage")
        ax3.set_ylabel("Integrated Drawdown")
        ax3.legend(loc="best")
        ax3.grid(True)

        plt.tight_layout()
        plt.show()
        return fig
