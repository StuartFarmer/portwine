import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portwine.analyzers.base import Analyzer


class TransactionCostAnalyzer(Analyzer):
    """
    Analyzes the impact of transaction costs on strategy performance.

    This analyzer models transaction costs for portfolio allocation strategies
    and shows how different cost levels affect performance metrics.

    Parameters
    ----------
    cost_levels : list of float, optional
        List of transaction cost levels to analyze (e.g., [0.0005, 0.001, 0.002])
        Default is [0, 0.0005, 0.001, 0.002, 0.005] (0 to 50 bps)
    cost_model : str, optional
        Transaction cost model to use ('proportional' or 'fixed')
        Default is 'proportional'
    """

    def __init__(self, cost_levels=None, cost_model='proportional'):
        self.cost_levels = cost_levels if cost_levels is not None else [0, 0.0005, 0.001, 0.002, 0.005]
        self.cost_model = cost_model

    def calculate_transaction_costs(self, signals_df, cost_per_trade):
        """
        Calculate transaction costs from a dataframe of portfolio weights

        Parameters
        ----------
        signals_df : pd.DataFrame
            DataFrame with dates as index and assets as columns, containing allocation weights
        cost_per_trade : float or dict
            Either a flat cost rate (e.g., 0.001 for 10bps) or a dictionary mapping tickers to specific costs

        Returns
        -------
        pd.Series
            Series of transaction costs for each date
        """
        # Calculate weight changes (absolute value of difference)
        weight_changes = signals_df.diff().abs()

        # For the first row, consider the initial allocation as a change from zero
        weight_changes.iloc[0] = signals_df.iloc[0].abs()

        # Apply costs to each ticker
        if isinstance(cost_per_trade, dict):
            # Apply different costs per ticker
            costs_per_ticker = pd.DataFrame({ticker: weight_changes[ticker] * rate
                                             for ticker, rate in cost_per_trade.items()
                                             if ticker in weight_changes.columns})
        else:
            # Apply uniform cost to all tickers
            costs_per_ticker = weight_changes.multiply(cost_per_trade)

        # Sum costs across all tickers for each date
        daily_costs = costs_per_ticker.sum(axis=1)

        return daily_costs

    def calculate_turnover(self, signals_df):
        """
        Calculate the average portfolio turnover (sum of weight changes per period)

        Parameters
        ----------
        signals_df : pd.DataFrame
            DataFrame with dates as index and assets as columns, containing allocation weights

        Returns
        -------
        float
            Average one-way turnover per period
        """
        # Calculate weight changes (absolute value of difference)
        weight_changes = signals_df.diff().abs()

        # For the first row, consider the initial allocation as a change from zero
        weight_changes.iloc[0] = signals_df.iloc[0].abs()

        # Sum weight changes across all tickers for each date
        daily_turnover = weight_changes.sum(axis=1)

        # Average turnover per period
        avg_turnover = daily_turnover.mean()

        return avg_turnover

    def apply_transaction_costs(self, strategy_returns, transaction_costs):
        """
        Apply transaction costs to strategy returns

        Parameters
        ----------
        strategy_returns : pd.Series
            Original strategy returns
        transaction_costs : pd.Series
            Transaction costs for each period

        Returns
        -------
        pd.Series
            Strategy returns after transaction costs
        """
        # Align the indexes
        aligned_costs = transaction_costs.reindex(strategy_returns.index, fill_value=0)

        # Subtract costs from returns
        net_returns = strategy_returns - aligned_costs

        return net_returns

    def analyze_performance(self, returns, ann_factor=252):
        """
        Calculate key performance metrics for a return series

        Parameters
        ----------
        returns : pd.Series
            Return series to analyze
        ann_factor : int, optional
            Annualization factor (default: 252 for daily data)

        Returns
        -------
        dict
            Dictionary of performance metrics
        """
        # Calculate total return
        total_return = (1 + returns).prod() - 1

        # Calculate annualized return
        n_periods = len(returns)
        years = n_periods / ann_factor
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Calculate volatility
        volatility = returns.std() * np.sqrt(ann_factor)

        # Calculate Sharpe ratio
        sharpe = cagr / volatility if volatility > 0 else 0

        # Calculate maximum drawdown
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns / peak) - 1
        max_drawdown = drawdown.min()

        # Calculate win rate
        win_rate = (returns > 0).mean()

        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }

    def analyze(self, results, ann_factor=252):
        """
        Analyze the impact of various transaction cost levels on strategy performance

        Parameters
        ----------
        results : dict
            Results dictionary from backtester containing:
            - signals_df: DataFrame of portfolio weights
            - strategy_returns: Series of strategy returns
            - benchmark_returns: Series of benchmark returns (optional)
        ann_factor : int, optional
            Annualization factor (default: 252 for daily data)

        Returns
        -------
        dict
            Dictionary with analysis results for each cost level
        """
        signals_df = results.get('signals_df')
        strategy_returns = results.get('strategy_returns')
        benchmark_returns = results.get('benchmark_returns')

        if signals_df is None or strategy_returns is None:
            raise ValueError("Missing required data in results dictionary")

        # Calculate average turnover
        avg_turnover = self.calculate_turnover(signals_df)

        # Analyze performance for each cost level
        analysis = {
            'avg_turnover': avg_turnover,
            'cost_levels': self.cost_levels,
            'performance': {},
            'equity_curves': {},
            'transaction_costs': {},
            'benchmark_returns': benchmark_returns  # Store for plot method
        }

        # Include benchmark performance if available
        if benchmark_returns is not None:
            analysis['benchmark'] = self.analyze_performance(benchmark_returns, ann_factor)

        for cost_level in self.cost_levels:
            # Calculate transaction costs
            transaction_costs = self.calculate_transaction_costs(signals_df, cost_level)

            # Apply costs to returns
            net_returns = self.apply_transaction_costs(strategy_returns, transaction_costs)

            # Analyze performance
            performance = self.analyze_performance(net_returns, ann_factor)

            # Store results
            analysis['performance'][cost_level] = performance
            analysis['equity_curves'][cost_level] = (1 + net_returns).cumprod()
            analysis['transaction_costs'][cost_level] = transaction_costs

        return analysis

    def plot(self, results, figsize=(15, 20), ann_factor=252, save_figure_to=None):
        """
        Create visualizations showing the impact of transaction costs
        with color-coded regions for different cost scenarios.

        Parameters
        ----------
        results : dict
            Results from the backtester (containing signals_df, strategy_returns, etc.)
        figsize : tuple, optional
            Figure size (default: (15, 20))
        ann_factor : int, optional
            Annualization factor (default: 252 for daily data)
        save_figure_to : str, optional
            Filename to save the figure to. If None, the plot is shown instead.
        """
        # Always run analysis on the provided backtester results
        analysis_results = self.analyze(results, ann_factor=ann_factor)

        # Create a single figure
        fig = plt.figure(figsize=figsize)

        # Define a layout with 4 rows
        # Row 0: Equity curves (tallest)
        # Row 1: Rolling turnover (half height of equity curves)
        # Row 2: Performance metrics and drawdown side by side
        # Row 3: Breakeven analyses side by side
        gs = plt.GridSpec(4, 2, height_ratios=[3, 1.5, 1.5, 1.5], figure=fig, hspace=0.3, wspace=0.4)

        # Extract signals_df and calculate turnover
        signals_df = results.get('signals_df')
        daily_turnover = None

        if signals_df is not None:
            weight_changes = signals_df.diff().abs()
            weight_changes.iloc[0] = signals_df.iloc[0].abs()
            daily_turnover = weight_changes.sum(axis=1)

        # Extract common data for plots
        cost_levels = analysis_results['cost_levels']
        colors = plt.cm.viridis(np.linspace(0, 1, len(cost_levels)))

        # -------------------------------------------------------------------------
        # Plot 1: Equity curves - spans full width
        # -------------------------------------------------------------------------
        ax_equity = fig.add_subplot(gs[0, :])

        for i, cost in enumerate(cost_levels):
            equity_curve = analysis_results['equity_curves'][cost]
            ax_equity.plot(equity_curve.index, equity_curve.values,
                           label=f"Cost: {cost * 10000:.0f} bps",
                           color=colors[i])

        # Plot benchmark if available
        if 'benchmark' in analysis_results and 'benchmark_returns' in analysis_results:
            benchmark_returns = analysis_results['benchmark_returns']
            if benchmark_returns is not None:
                benchmark_equity = (1 + benchmark_returns).cumprod()
                ax_equity.plot(benchmark_equity.index, benchmark_equity.values,
                               label="Benchmark", color='black', linestyle='--')

        ax_equity.set_title("Equity Curves with Different Transaction Costs", fontsize=14)
        ax_equity.set_ylabel("Equity (Starting at 1.0)")
        ax_equity.grid(True, alpha=0.3)
        ax_equity.legend(loc='upper left')

        # -------------------------------------------------------------------------
        # Plot 2: Rolling turnover over time - spans full width
        # -------------------------------------------------------------------------
        ax_turnover = fig.add_subplot(gs[1, :])

        if daily_turnover is not None:
            # Calculate rolling metrics
            window_size = min(30, len(daily_turnover) // 10) if len(daily_turnover) > 30 else 10
            rolling_mean = daily_turnover.rolling(window=window_size).mean()
            rolling_std = daily_turnover.rolling(window=window_size).std()

            # Calculate confidence bands (mean ± 1 std)
            upper_band = rolling_mean + rolling_std
            lower_band = (rolling_mean - rolling_std).clip(0)  # Don't go below 0

            # Plot raw turnover with low alpha for context
            ax_turnover.plot(daily_turnover.index, daily_turnover,
                             color='lightblue', alpha=0.3, linewidth=0.5, label='Daily Turnover')

            # Plot confidence bands
            ax_turnover.fill_between(daily_turnover.index, lower_band, upper_band,
                                     color='blue', alpha=0.2, label='±1σ Band')

            # Plot rolling average line (on top of the bands)
            ax_turnover.plot(daily_turnover.index, rolling_mean,
                             color='blue', linewidth=2, label=f'{window_size}-day Rolling Avg')

            # Add horizontal line for overall average
            avg_turnover = daily_turnover.mean()
            ax_turnover.axhline(y=avg_turnover, color='red', linestyle='--',
                                label=f'Overall Avg: {avg_turnover:.2%}')

            # Add turnover stats as text in the upper right
            turnover_text = (
                f"Average Portfolio One-Way Turnover: {avg_turnover:.2%} per period\n"
                f"Implied Round-Trip Turnover: {avg_turnover * 2:.2%} per period\n"
                f"Annualized One-Way Turnover: {avg_turnover * ann_factor:.2%}"
            )

            ax_turnover.text(0.98, 0.95, turnover_text,
                             transform=ax_turnover.transAxes,
                             verticalalignment='top',
                             horizontalalignment='right',
                             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                             fontsize=10,
                             family='monospace')

            ax_turnover.set_ylabel('Portfolio Turnover')
            ax_turnover.set_title('Rolling Portfolio Turnover Over Time', fontsize=14)
            ax_turnover.legend(loc='upper left')
            ax_turnover.grid(True, alpha=0.3)
            ax_turnover.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        else:
            # If no signals_df, show information message
            ax_turnover.text(0.5, 0.5, "Turnover analysis not available\nNo portfolio weights found in results",
                             horizontalalignment='center', verticalalignment='center',
                             transform=ax_turnover.transAxes, fontsize=12)
            ax_turnover.set_title("Portfolio Turnover Analysis", fontsize=14)
            ax_turnover.axis('off')

        # Define cost scenario bands parameters (used in multiple plots)
        optimistic_threshold = 0.0005  # 5 bps
        realistic_threshold = 0.0015  # 15 bps
        x_min = 0
        x_max = max(cost_levels) * 1.1  # Add some padding

        # -------------------------------------------------------------------------
        # Plot 3a: Performance metrics vs cost level (left column)
        # -------------------------------------------------------------------------
        ax_metrics = fig.add_subplot(gs[2, 0])

        # Add background color bands
        ax_metrics.axvspan(0, optimistic_threshold, alpha=0.2, color='green', label='Optimistic (0-5 bps)')
        ax_metrics.axvspan(optimistic_threshold, realistic_threshold, alpha=0.2, color='yellow',
                           label='Realistic (5-15 bps)')
        ax_metrics.axvspan(realistic_threshold, x_max, alpha=0.2, color='pink', label='Conservative (15+ bps)')
        ax_metrics.set_xlim(x_min, x_max)

        # Extract metrics
        sharpes = [analysis_results['performance'][cost]['sharpe'] for cost in cost_levels]
        returns = [analysis_results['performance'][cost]['cagr'] for cost in cost_levels]

        # Plot lines
        ax_metrics.plot(cost_levels, returns, 'o-', label="CAGR", linewidth=2, zorder=5)
        ax_metrics2 = ax_metrics.twinx()
        ax_metrics2.plot(cost_levels, sharpes, 'o-', color='red', label="Sharpe", linewidth=2, zorder=5)

        # Add value labels (every other point if there are many)
        for i, cost in enumerate(cost_levels):
            if len(cost_levels) <= 8 or i % 2 == 0:
                ax_metrics.annotate(f"{returns[i]:.2%}",
                                    (cost, returns[i]),
                                    textcoords="offset points",
                                    xytext=(0, 10),
                                    ha='center',
                                    fontsize=8,
                                    zorder=6)

                ax_metrics2.annotate(f"{sharpes[i]:.2f}",
                                     (cost, sharpes[i]),
                                     textcoords="offset points",
                                     xytext=(0, -10),
                                     ha='center',
                                     fontsize=8,
                                     zorder=6)

        # Configure axes
        ax_metrics.set_xlabel("Transaction Cost (one-way)")
        ax_metrics.set_ylabel("CAGR")
        ax_metrics2.set_ylabel("Sharpe Ratio")
        ax_metrics.set_title("Performance Metrics vs Transaction Costs", fontsize=12)
        ax_metrics.grid(True, alpha=0.3, zorder=0)

        # Combine legends
        handles1, labels1 = ax_metrics.get_legend_handles_labels()
        handles2, labels2 = ax_metrics2.get_legend_handles_labels()
        by_label = dict(zip(labels1 + labels2, handles1 + handles2))
        ax_metrics.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)

        # Format x-axis as bps
        ax_metrics.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 10000:.0f} bps'))

        # -------------------------------------------------------------------------
        # Plot 3b: Max Drawdown vs cost level (right column)
        # -------------------------------------------------------------------------
        ax_drawdown = fig.add_subplot(gs[2, 1])

        # Add cost scenario bands
        ax_drawdown.axvspan(0, optimistic_threshold, alpha=0.2, color='green')
        ax_drawdown.axvspan(optimistic_threshold, realistic_threshold, alpha=0.2, color='yellow')
        ax_drawdown.axvspan(realistic_threshold, x_max, alpha=0.2, color='pink')
        ax_drawdown.set_xlim(x_min, x_max)

        # Extract drawdowns
        drawdowns = [analysis_results['performance'][cost]['max_drawdown'] for cost in cost_levels]

        # Plot line
        ax_drawdown.plot(cost_levels, drawdowns, 'o-', color='red', linewidth=2, zorder=5)

        # Add value labels (every other point if there are many)
        for i, cost in enumerate(cost_levels):
            if len(cost_levels) <= 8 or i % 2 == 0:
                ax_drawdown.annotate(f"{drawdowns[i]:.2%}",
                                     (cost, drawdowns[i]),
                                     textcoords="offset points",
                                     xytext=(0, 10),
                                     ha='center',
                                     fontsize=8,
                                     zorder=6)

        # Configure axes
        ax_drawdown.set_xlabel("Transaction Cost (one-way)")
        ax_drawdown.set_ylabel("Maximum Drawdown")
        ax_drawdown.set_title("Maximum Drawdown vs Transaction Costs", fontsize=12)
        ax_drawdown.grid(True, alpha=0.3, zorder=0)

        # Format x-axis as bps
        ax_drawdown.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 10000:.0f} bps'))
        ax_drawdown.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        # -------------------------------------------------------------------------
        # Plot 4a: Return Breakeven Analysis (left column)
        # -------------------------------------------------------------------------
        ax_return_be = fig.add_subplot(gs[3, 0])

        # Extract data for total returns
        total_returns = [analysis_results['performance'][cost]['total_return'] for cost in cost_levels]

        # Add cost scenario bands
        ax_return_be.axvspan(0, optimistic_threshold, alpha=0.2, color='green')
        ax_return_be.axvspan(optimistic_threshold, realistic_threshold, alpha=0.2, color='yellow')
        ax_return_be.axvspan(realistic_threshold, x_max, alpha=0.2, color='pink')
        ax_return_be.set_xlim(x_min, x_max)

        # Plot return vs cost
        ax_return_be.plot(cost_levels, total_returns, 'o-', color='blue', linewidth=2, zorder=5)
        ax_return_be.axhline(y=0, color='r', linestyle='--', label='Breakeven', zorder=4)

        # Add value labels (every other point if there are many)
        for i, cost in enumerate(cost_levels):
            if len(cost_levels) <= 8 or i % 2 == 0:
                ax_return_be.annotate(f"{total_returns[i]:.2%}",
                                      (cost, total_returns[i]),
                                      textcoords="offset points",
                                      xytext=(0, 10),
                                      ha='center',
                                      fontsize=8,
                                      zorder=6)

        # Find breakeven point (where total return crosses zero)
        if min(total_returns) < 0 < max(total_returns):
            from scipy.interpolate import interp1d
            interp_func = interp1d(total_returns, cost_levels)
            try:
                breakeven_cost = float(interp_func(0))
                ax_return_be.axvline(x=breakeven_cost, color='g', linestyle='--',
                                     label=f'Breakeven: {breakeven_cost * 10000:.1f} bps', zorder=4)
            except:
                pass

        ax_return_be.set_xlabel("Transaction Cost (one-way)")
        ax_return_be.set_ylabel("Total Return")
        ax_return_be.set_title("Return Breakeven Analysis", fontsize=12)
        ax_return_be.grid(True, alpha=0.3, zorder=0)

        # Format axes
        ax_return_be.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 10000:.0f} bps'))
        ax_return_be.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        # Add legend
        handles, labels = ax_return_be.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_return_be.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)

        # -------------------------------------------------------------------------
        # Plot 4b: Sharpe Breakeven Analysis (right column)
        # -------------------------------------------------------------------------
        ax_sharpe_be = fig.add_subplot(gs[3, 1])

        # Add cost scenario bands
        ax_sharpe_be.axvspan(0, optimistic_threshold, alpha=0.2, color='green')
        ax_sharpe_be.axvspan(optimistic_threshold, realistic_threshold, alpha=0.2, color='yellow')
        ax_sharpe_be.axvspan(realistic_threshold, x_max, alpha=0.2, color='pink')
        ax_sharpe_be.set_xlim(x_min, x_max)

        # Plot Sharpe vs cost
        ax_sharpe_be.plot(cost_levels, sharpes, 'o-', color='red', linewidth=2, zorder=5)
        ax_sharpe_be.axhline(y=0, color='r', linestyle='--', label='Sharpe = 0', zorder=4)

        # Add value labels (every other point if there are many)
        for i, cost in enumerate(cost_levels):
            if len(cost_levels) <= 8 or i % 2 == 0:
                ax_sharpe_be.annotate(f"{sharpes[i]:.2f}",
                                      (cost, sharpes[i]),
                                      textcoords="offset points",
                                      xytext=(0, 10),
                                      ha='center',
                                      fontsize=8,
                                      zorder=6)

        # Find where Sharpe crosses zero
        if min(sharpes) < 0 < max(sharpes):
            from scipy.interpolate import interp1d
            interp_func = interp1d(sharpes, cost_levels)
            try:
                breakeven_cost = float(interp_func(0))
                ax_sharpe_be.axvline(x=breakeven_cost, color='g', linestyle='--',
                                     label=f'Breakeven: {breakeven_cost * 10000:.0f} bps', zorder=4)
            except:
                pass

        ax_sharpe_be.set_xlabel("Transaction Cost (one-way)")
        ax_sharpe_be.set_ylabel("Sharpe Ratio")
        ax_sharpe_be.set_title("Sharpe Breakeven Analysis", fontsize=12)
        ax_sharpe_be.grid(True, alpha=0.3, zorder=0)

        # Format x-axis
        ax_sharpe_be.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 10000:.0f} bps'))

        # Add legend (only with non-background items)
        handles, labels = ax_sharpe_be.get_legend_handles_labels()
        filtered_handles = [h for h, l in zip(handles, labels) if
                            not l.startswith(('Optimistic', 'Realistic', 'Conservative'))]
        filtered_labels = [l for l in labels if not l.startswith(('Optimistic', 'Realistic', 'Conservative'))]
        ax_sharpe_be.legend(filtered_handles, filtered_labels, loc='best', fontsize=9)

        # Save or show the figure
        if save_figure_to is not None:
            plt.savefig(save_figure_to, dpi=150, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
        else:
            plt.show()

    def plot_breakeven_analysis(self, results, figsize=(15, 8)):
        """
        This method is no longer used as the breakeven analysis is included
        in the main plot. Kept for backwards compatibility.
        """
        pass

    def generate_report(self, results, ann_factor=252):
        """
        Generate a text report summarizing the transaction cost analysis

        Parameters
        ----------
        results : dict
            Results from the backtester (containing signals_df, strategy_returns, etc.)
        ann_factor : int, optional
            Annualization factor (default: 252 for daily data)

        Returns
        -------
        str
            Text report of the analysis
        """
        # Always run analysis on the provided backtester results
        analysis_results = self.analyze(results, ann_factor=ann_factor)

        lines = []
        lines.append("\n==== TRANSACTION COST ANALYSIS ====\n")

        # Portfolio turnover
        avg_turnover = analysis_results['avg_turnover']
        lines.append(f"Average Portfolio One-Way Turnover: {avg_turnover:.2%} per period")
        lines.append(f"Implied Round-Trip Turnover: {avg_turnover * 2:.2%} per period")

        # Annualized turnover
        annual_turnover = avg_turnover * ann_factor
        lines.append(f"Annualized One-Way Turnover: {annual_turnover:.2%}")
        lines.append("")

        # Performance for each cost level
        lines.append("--- Performance Metrics by Cost Level ---")
        header = f"{'Cost Level':12s} | {'CAGR':8s} | {'Sharpe':8s} | {'Max DD':8s} | {'Volatility':8s}"
        lines.append(header)
        lines.append("-" * len(header))

        for cost in analysis_results['cost_levels']:
            perf = analysis_results['performance'][cost]
            cost_str = f"{cost * 10000:.1f} bps"
            cagr_str = f"{perf['cagr']:.2%}"
            sharpe_str = f"{perf['sharpe']:.2f}"
            dd_str = f"{perf['max_drawdown']:.2%}"
            vol_str = f"{perf['volatility']:.2%}"

            lines.append(f"{cost_str:12s} | {cagr_str:8s} | {sharpe_str:8s} | {dd_str:8s} | {vol_str:8s}")

        lines.append("")

        # Breakeven analysis
        cost_levels = analysis_results['cost_levels']
        returns = [analysis_results['performance'][cost]['total_return'] for cost in cost_levels]
        sharpes = [analysis_results['performance'][cost]['sharpe'] for cost in cost_levels]

        lines.append("--- Breakeven Analysis ---")

        # Return breakeven
        if min(returns) < 0 < max(returns):
            from scipy.interpolate import interp1d
            try:
                interp_func = interp1d(returns, cost_levels)
                breakeven_cost = float(interp_func(0))
                lines.append(f"Return Breakeven Point: {breakeven_cost * 10000:.1f} bps")
            except:
                lines.append("Could not calculate return breakeven point")
        else:
            if min(returns) >= 0:
                lines.append("Strategy remains profitable at all tested cost levels")
            else:
                lines.append("Strategy is unprofitable at all tested cost levels")

        # Sharpe breakeven
        if min(sharpes) < 0 < max(sharpes):
            from scipy.interpolate import interp1d
            try:
                interp_func = interp1d(sharpes, cost_levels)
                breakeven_cost = float(interp_func(0))
                lines.append(f"Sharpe Ratio Breakeven Point: {breakeven_cost * 10000:.1f} bps")
            except:
                lines.append("Could not calculate Sharpe breakeven point")
        else:
            if min(sharpes) >= 0:
                lines.append("Sharpe ratio remains positive at all tested cost levels")
            else:
                lines.append("Sharpe ratio is negative at all tested cost levels")

        return "\n".join(lines)