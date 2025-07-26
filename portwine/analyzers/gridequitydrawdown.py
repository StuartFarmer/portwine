import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from portwine.analyzers.base import Analyzer

class GridEquityDrawdownAnalyzer(Analyzer):
    """
    Plots multiple strategy results in a grid layout, where *each cell* of the grid
    is treated as one "element" or "basket." Internally, each basket is composed of
    two vertically-stacked plots:
      - Top: Equity curves (strategy vs. benchmark)
      - Bottom: Drawdown curves (strategy vs. benchmark)

    Each basket has its own title (placed above the top subplot), and the equity and
    drawdown subplots share the same x-axis so only the bottom chart shows the date
    tick labels. We adjust legend font size and inter-plot spacing so that the final
    figure is more visually appealing, without overlapping elements.

    Usage Example
    -------------
    analyzer = GridEquityDrawdownAnalyzer()
    results_list = [
        {'strategy_returns': pd.Series(...), 'benchmark_returns': pd.Series(...)},
        ...
    ]
    titles = ["Basket A", "Basket B", "Basket C", ...]  # same length as results_list
    analyzer.plot(results_list, titles, ncols=2, benchmark_label="Benchmark")
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
        pd.Series
            The percentage drawdown at each point in time.
        """
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        return drawdown

    def plot(self, results_list, titles, ncols=2, benchmark_label="Benchmark"):
        """
        Creates a grid of 'baskets' (nrows x ncols). Each basket is a small layout of
        two subplots: top = equity curve, bottom = drawdown curve.

        Parameters
        ----------
        results_list : list of dict
            Each dict must have 'strategy_returns' and 'benchmark_returns' as pd.Series.
            Example:
                {
                    'strategy_returns': pd.Series([...], index=dates),
                    'benchmark_returns': pd.Series([...], index=dates)
                }
        titles : list of str
            Basket titles. Must match len(results_list).
        ncols : int
            Number of columns in the top-level grid. The number of rows is computed
            automatically.
        benchmark_label : str
            Label for the benchmark lines in equity/drawdown plots.
        """
        if len(results_list) != len(titles):
            raise ValueError("Length of results_list and titles must match.")

        # Number of baskets and how many rows of baskets we need:
        n_baskets = len(results_list)
        nrows = math.ceil(n_baskets / ncols)

        # Create a figure. Each basket is 1 cell in an nrows x ncols grid,
        # but each cell is subdivided into two subplots (top=equity, bottom=drawdown).
        fig = plt.figure(figsize=(6 * ncols, 4 * nrows))

        # Create the top-level GridSpec for the entire figure, with nrows x ncols 'cells'.
        # Reduce spacing between cells to avoid excessive padding
        main_gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.3, wspace=0.3)

        for i, results in enumerate(results_list):
            # Determine which row & column this basket belongs to in the top-level grid
            row = i // ncols
            col = i % ncols

            # For each cell in the main grid, we create a sub-GridSpec with 2 rows (top & bottom).
            # The top is for equity, the bottom is for drawdown, and they share the x-axis.
            basket_gs = main_gs[row, col].subgridspec(
                2, 1,
                height_ratios=[2, 1],  # top subplot is a bit taller than bottom
                hspace=0.03  # Reduce space between top & bottom inside this basket
            )

            # Create the two subplots for this basket
            ax_equity = fig.add_subplot(basket_gs[0, 0])
            ax_drawdown = fig.add_subplot(basket_gs[1, 0], sharex=ax_equity)

            # ----- Top: Equity Curves -----
            ax_equity.set_title(titles[i], pad=8)  # Reduce title padding
            strat_equity = (1.0 + results['strategy_returns']).cumprod()
            bench_equity = (1.0 + results['benchmark_returns']).cumprod()

            ax_equity.plot(
                strat_equity.index, strat_equity.values,
                label="Strategy",
                color='darkblue',
                linewidth=0.5  # Thinner line
            )
            ax_equity.plot(
                bench_equity.index, bench_equity.values,
                label=benchmark_label,
                color='black',
                linewidth=0.5  # Even thinner for benchmark
            )
            ax_equity.set_ylabel("Equity")
            ax_equity.grid(True, alpha=0.3)  # Lighter grid

            # Fill between strategy & benchmark equity
            ax_equity.fill_between(
                strat_equity.index,
                strat_equity.values,
                bench_equity.values,
                where=(strat_equity.values >= bench_equity.values),
                color='green',
                alpha=0.2,  # Lighter fill
                interpolate=True
            )
            ax_equity.fill_between(
                strat_equity.index,
                strat_equity.values,
                bench_equity.values,
                where=(strat_equity.values < bench_equity.values),
                color='red',
                alpha=0.2,  # Lighter fill
                interpolate=True
            )

            # Smaller legend with reduced size
            ax_equity.legend(fontsize=7, loc='best')

            # ----- Bottom: Drawdown Curves -----
            strat_dd = self.compute_drawdown(strat_equity) * 100.0
            bench_dd = self.compute_drawdown(bench_equity) * 100.0

            ax_drawdown.plot(
                strat_dd.index, strat_dd.values,
                label="Strategy DD (%)",
                color='darkblue',
                linewidth=0.5  # Thinner line
            )
            ax_drawdown.plot(
                bench_dd.index, bench_dd.values,
                label=f"{benchmark_label} DD (%)",
                color='black',
                linewidth=0.5  # Even thinner for benchmark
            )
            ax_drawdown.set_ylabel("Drawdown (%)")
            ax_drawdown.grid(True, alpha=0.3)  # Lighter grid

            # Fill between strategy & benchmark drawdown
            ax_drawdown.fill_between(
                strat_dd.index,
                strat_dd.values,
                bench_dd.values,
                where=(strat_dd.values <= bench_dd.values),
                color='red',
                alpha=0.2,  # Lighter fill
                interpolate=True
            )
            ax_drawdown.fill_between(
                strat_dd.index,
                strat_dd.values,
                bench_dd.values,
                where=(strat_dd.values > bench_dd.values),
                color='green',
                alpha=0.2,  # Lighter fill
                interpolate=True
            )

            ax_drawdown.legend(fontsize=7, loc='best')

            # We only want the x-axis tick labels to appear on the bottom chart
            # (sharex=...) automatically aligns them, so remove from top:
            plt.setp(ax_equity.get_xticklabels(), visible=False)

            # Make x-axis tick labels smaller
            ax_drawdown.tick_params(axis='x', labelsize=8)

            # Rotate x-axis tick labels for better readability
            plt.setp(ax_drawdown.get_xticklabels(), rotation=30, ha='right')

        plt.show()
