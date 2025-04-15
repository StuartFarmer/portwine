import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from portwine.analyzers.base import Analyzer


class RegimeChangeAnalyzer(Analyzer):
    """
    Analyzes strategy performance across different market regimes.

    This analyzer segments market data into different regimes (bull, bear, volatile, etc.)
    and evaluates how a strategy performs in each regime. It helps identify potential
    vulnerabilities in different market conditions.

    Parameters
    ----------
    regime_definitions : dict, optional
        Custom regime definitions. Default is None, which uses built-in definitions.
    lookback_window : int, optional
        Window for rolling metrics calculation (default: 20 days)
    """

    def __init__(self, regime_definitions=None, lookback_window=20):
        self.lookback_window = lookback_window

        # Default regime definitions
        self.default_regimes = {
            'bull': {'description': 'Moderate upward trend, normal volatility'},
            'strong_bull': {'description': 'Strong upward trend, normal volatility'},
            'bear': {'description': 'Moderate downward trend, normal volatility'},
            'strong_bear': {'description': 'Strong downward trend, normal volatility'},
            'sideways': {'description': 'Low trend, low volatility range-bound market'},
            'volatile_up': {'description': 'Upward bias with high volatility'},
            'volatile_down': {'description': 'Downward bias with high volatility'}
        }

        # Use custom definitions if provided, otherwise use defaults
        self.regime_definitions = regime_definitions if regime_definitions is not None else self.default_regimes

    def identify_regimes(self, benchmark_returns, method='combined'):
        """
        Identifies market regimes based on benchmark returns.

        Parameters
        ----------
        benchmark_returns : pd.Series
            Daily returns of the benchmark
        method : str, optional
            Method to use for regime identification:
            - 'trend': Based on rolling returns only
            - 'volatility': Based on rolling volatility only
            - 'combined': Using both trend and volatility (default)

        Returns
        -------
        pd.Series
            Series with regime labels for each date
        """
        # Ensure we have enough data
        if len(benchmark_returns) < self.lookback_window * 2:
            raise ValueError(f"Need at least {self.lookback_window * 2} data points")

        # Calculate rolling metrics
        rolling_returns = benchmark_returns.rolling(window=self.lookback_window).mean() * self.lookback_window
        rolling_vol = benchmark_returns.rolling(window=self.lookback_window).std() * np.sqrt(self.lookback_window)

        # Calculate longer-term metrics for comparison
        long_window = min(self.lookback_window * 3, len(benchmark_returns) // 2)
        rolling_long_vol = benchmark_returns.rolling(window=long_window).std() * np.sqrt(long_window)

        # Initialize regimes Series
        regimes = pd.Series(index=benchmark_returns.index, dtype='object')

        # Set volatility thresholds
        high_vol_threshold = rolling_long_vol.quantile(0.7)  # Top 30% volatility
        low_vol_threshold = rolling_long_vol.quantile(0.3)  # Bottom 30% volatility

        # Set trend thresholds
        positive_trend_threshold = 0.01  # 1% over lookback period
        negative_trend_threshold = -0.01  # -1% over lookback period
        strong_trend_threshold = 0.03  # 3% over lookback period

        # Loop through dates and assign regimes
        if method == 'trend':
            # Simplified trend-based regimes
            regimes.loc[rolling_returns > positive_trend_threshold] = 'bull'
            regimes.loc[rolling_returns < negative_trend_threshold] = 'bear'
            regimes.loc[(rolling_returns >= negative_trend_threshold) &
                        (rolling_returns <= positive_trend_threshold)] = 'sideways'

        elif method == 'volatility':
            # Simplified volatility-based regimes
            regimes.loc[rolling_vol > high_vol_threshold] = 'volatile'
            regimes.loc[rolling_vol <= high_vol_threshold] = 'normal'
            regimes.loc[rolling_vol < low_vol_threshold] = 'calm'

        else:  # combined approach (default)
            # Bull market
            regimes.loc[(rolling_returns > positive_trend_threshold) &
                        (rolling_vol <= high_vol_threshold)] = 'bull'

            # Strong bull market
            regimes.loc[(rolling_returns > strong_trend_threshold) &
                        (rolling_vol <= high_vol_threshold)] = 'strong_bull'

            # Bear market
            regimes.loc[(rolling_returns < negative_trend_threshold) &
                        (rolling_vol <= high_vol_threshold)] = 'bear'

            # Strong bear market
            regimes.loc[(rolling_returns < -strong_trend_threshold) &
                        (rolling_vol <= high_vol_threshold)] = 'strong_bear'

            # Sideways market (low trend, low volatility)
            regimes.loc[(rolling_returns >= negative_trend_threshold) &
                        (rolling_returns <= positive_trend_threshold) &
                        (rolling_vol <= low_vol_threshold)] = 'sideways'

            # Volatile market with upward bias
            regimes.loc[(rolling_returns > 0) & (rolling_vol > high_vol_threshold)] = 'volatile_up'

            # Volatile market with downward bias
            regimes.loc[(rolling_returns <= 0) & (rolling_vol > high_vol_threshold)] = 'volatile_down'

            # Fill any remaining NaN with 'undefined'
            regimes.fillna('undefined', inplace=True)

        return regimes

    def _extract_continuous_regime_periods(self, regimes):
        """
        Extract continuous periods for each regime.

        Parameters
        ----------
        regimes : pd.Series
            Series with regime labels for each date

        Returns
        -------
        dict
            Dictionary of regime periods: {regime: [(start_date, end_date), ...]}
        """
        regime_periods = {}
        current_regime = None
        start_date = None

        for date, regime in regimes.items():
            if regime != current_regime:
                # Save the previous period if it exists
                if current_regime is not None and start_date is not None:
                    if current_regime not in regime_periods:
                        regime_periods[current_regime] = []
                    regime_periods[current_regime].append((start_date, date))

                # Start a new period
                current_regime = regime
                start_date = date

        # Add the last regime period
        if current_regime is not None and start_date is not None:
            if current_regime not in regime_periods:
                regime_periods[current_regime] = []
            regime_periods[current_regime].append((start_date, regimes.index[-1]))

        return regime_periods

    def calculate_regime_metrics(self, strategy_returns, benchmark_returns, regimes, ann_factor=252):
        """
        Calculate performance metrics for each regime.

        Parameters
        ----------
        strategy_returns : pd.Series
            Daily returns of the strategy
        benchmark_returns : pd.Series
            Daily returns of the benchmark
        regimes : pd.Series
            Series with regime labels for each date
        ann_factor : int, optional
            Annualization factor (default: 252 for daily data)

        Returns
        -------
        dict
            Dictionary with metrics for each regime
        """
        metrics = {}

        # Extract continuous periods for each regime
        regime_periods = self._extract_continuous_regime_periods(regimes)

        # Get unique regimes
        unique_regimes = regimes.unique()

        for regime in unique_regimes:
            # Skip undefined regime
            if regime == 'undefined':
                continue

            # Get returns during this regime
            regime_mask = (regimes == regime)
            strat_regime_returns = strategy_returns.loc[regime_mask]
            bench_regime_returns = benchmark_returns.loc[regime_mask]

            # Skip if too few data points
            if len(strat_regime_returns) < 5:
                continue

            # Calculate basic metrics
            cumulative_strat = (1 + strat_regime_returns).prod() - 1
            cumulative_bench = (1 + bench_regime_returns).prod() - 1

            # Annualized metrics with better handling
            days_in_regime = len(strat_regime_returns)

            # Only annualize if the regime has enough data points
            if days_in_regime >= 20:  # At least ~1 month of data
                years = days_in_regime / ann_factor
                annualized_strat = (1 + cumulative_strat) ** (1 / years) - 1
                annualized_bench = (1 + cumulative_bench) ** (1 / years) - 1
            else:
                # For short regimes, don't try to annualize, just use the cumulative return
                annualized_strat = cumulative_strat
                annualized_bench = cumulative_bench

            # Cap annualized returns at reasonable levels (e.g., +/-1000%)
            max_return = 10.0  # 1000%
            annualized_strat = max(min(annualized_strat, max_return), -max_return)
            annualized_bench = max(min(annualized_bench, max_return), -max_return)

            # Volatility
            vol_strat = strat_regime_returns.std() * np.sqrt(min(ann_factor, days_in_regime))
            vol_bench = bench_regime_returns.std() * np.sqrt(min(ann_factor, days_in_regime))

            # Sharpe ratio
            sharpe_strat = (strat_regime_returns.mean() / strat_regime_returns.std()) * np.sqrt(
                min(ann_factor, days_in_regime)) if strat_regime_returns.std() > 0 else 0
            sharpe_bench = (bench_regime_returns.mean() / bench_regime_returns.std()) * np.sqrt(
                min(ann_factor, days_in_regime)) if bench_regime_returns.std() > 0 else 0

            # Maximum drawdown
            strat_equity = (1 + strat_regime_returns).cumprod()
            bench_equity = (1 + bench_regime_returns).cumprod()

            strat_peak = strat_equity.cummax()
            bench_peak = bench_equity.cummax()

            strat_drawdown = (strat_equity / strat_peak) - 1
            bench_drawdown = (bench_equity / bench_peak) - 1

            max_dd_strat = strat_drawdown.min()
            max_dd_bench = bench_drawdown.min()

            # Correlation
            correlation = strat_regime_returns.corr(bench_regime_returns)

            # Extract continuous periods data for this regime
            periods = regime_periods.get(regime, [])
            period_equity_curves = []

            for start_date, end_date in periods:
                # Extract returns for this period
                period_strat_returns = strategy_returns.loc[start_date:end_date]
                period_bench_returns = benchmark_returns.loc[start_date:end_date]

                # Calculate equity curves starting at 1.0
                period_strat_equity = (1 + period_strat_returns).cumprod()
                period_bench_equity = (1 + period_bench_returns).cumprod()

                # Store if period is long enough
                if len(period_strat_equity) >= 5:
                    period_equity_curves.append({
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration': len(period_strat_equity),
                        'strat_equity': period_strat_equity,
                        'bench_equity': period_bench_equity
                    })

            # Store metrics
            metrics[regime] = {
                'count': days_in_regime,
                'period_fraction': days_in_regime / len(strategy_returns),
                'cumulative_return': cumulative_strat,
                'benchmark_return': cumulative_bench,
                'annualized_return': annualized_strat,
                'benchmark_annualized': annualized_bench,
                'outperformance': cumulative_strat - cumulative_bench,
                'annualized_outperformance': annualized_strat - annualized_bench,
                'volatility': vol_strat,
                'benchmark_volatility': vol_bench,
                'sharpe': sharpe_strat,
                'benchmark_sharpe': sharpe_bench,
                'max_drawdown': max_dd_strat,
                'benchmark_max_drawdown': max_dd_bench,
                'correlation': correlation,
                'win_rate': (strat_regime_returns > 0).mean(),
                'returns': strat_regime_returns,
                'benchmark_returns': bench_regime_returns,
                'periods': period_equity_curves,
                'n_periods': len(period_equity_curves)
            }

        return metrics

    def analyze(self, results, method='combined', ann_factor=252):
        """
        Analyze strategy performance across different market regimes.

        Parameters
        ----------
        results : dict
            Results dictionary from backtester containing:
            - strategy_returns: Series of strategy returns
            - benchmark_returns: Series of benchmark returns
        method : str, optional
            Method for regime identification (default: 'combined')
        ann_factor : int, optional
            Annualization factor (default: 252 for daily data)

        Returns
        -------
        dict
            Dictionary with regime analysis results
        """
        strategy_returns = results.get('strategy_returns')
        benchmark_returns = results.get('benchmark_returns')

        if strategy_returns is None or benchmark_returns is None:
            raise ValueError("Missing required return data in results dictionary")

        # Align the return series
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        # Identify market regimes
        regimes = self.identify_regimes(benchmark_returns, method=method)

        # Calculate metrics for each regime
        regime_metrics = self.calculate_regime_metrics(
            strategy_returns, benchmark_returns, regimes, ann_factor=ann_factor
        )

        # Calculate market state transitions
        transitions = self._calculate_regime_transitions(regimes)

        # Return analysis results
        return {
            'regimes': regimes,
            'metrics': regime_metrics,
            'transitions': transitions,
            'method': method,
            'lookback_window': self.lookback_window,
            'regime_counts': regimes.value_counts().to_dict()
        }

    def _calculate_regime_transitions(self, regimes):
        """
        Calculate transition probabilities between regimes.

        Parameters
        ----------
        regimes : pd.Series
            Series with regime labels for each date

        Returns
        -------
        pd.DataFrame
            Matrix of transition probabilities
        """
        # Create shifted series to calculate transitions
        current = regimes
        previous = regimes.shift(1)

        # Get unique regimes
        unique_regimes = sorted(regimes.unique())

        # Create transition matrix with float dtype
        transitions = pd.DataFrame(0.0, index=unique_regimes, columns=unique_regimes)

        # Count transitions
        for prev, curr in zip(previous.dropna(), current.loc[previous.dropna().index]):
            transitions.loc[prev, curr] += 1.0

        # Convert to probabilities
        for regime in unique_regimes:
            row_sum = transitions.loc[regime].sum()
            if row_sum > 0:
                transitions.loc[regime] = transitions.loc[regime] / row_sum

        return transitions

    def plot(self, results, analysis=None, figsize=(15, 16), ncols=2, benchmark_label="Benchmark"):
        """
        Create visualizations of regime analysis with regime equity curves in a grid.
        Each grid cell shows a single consolidated equity curve per regime.

        Parameters
        ----------
        results : dict
            Results from the backtester (containing strategy_returns, benchmark_returns)
        analysis : dict, optional
            Analysis results (if None, will run analyze() first)
        figsize : tuple, optional
            Figure size (default: (15, 16))
        ncols : int, optional
            Number of columns in the grid layout (default: 2)
        benchmark_label : str, optional
            Label for benchmark in plots (default: "Benchmark")
        """
        if analysis is None:
            analysis = self.analyze(results)

        # Extract key components
        regimes = analysis['regimes']
        metrics = analysis['metrics']
        strategy_returns = results.get('strategy_returns')
        benchmark_returns = results.get('benchmark_returns')

        # Align returns with regimes
        common_index = strategy_returns.index.intersection(regimes.index)
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        regimes = regimes.loc[common_index]

        # Create cumulative return series
        strategy_equity = (1 + strategy_returns).cumprod()
        benchmark_equity = (1 + benchmark_returns).cumprod()

        # Set up colors for different regimes
        regime_colors = {
            'bull': 'green',
            'strong_bull': 'darkgreen',
            'bear': 'red',
            'strong_bear': 'darkred',
            'sideways': 'gray',
            'volatile_up': 'lightgreen',
            'volatile_down': 'lightcoral',
            'undefined': 'lightgray'
        }

        # Add any missing regimes with default colors
        for regime in regimes.unique():
            if regime not in regime_colors:
                regime_colors[regime] = 'blue'

            # Add description if not in regime_definitions
            if regime not in self.regime_definitions:
                self.regime_definitions[regime] = {'description': 'Custom regime'}

        # Create figure with overall equity plot at the top and regime grid below
        fig = plt.figure(figsize=figsize)

        # Define a GridSpec with 2 rows - top for main equity plot, bottom for regime grid
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])

        # Plot 1: Overall equity curve with regime backgrounds
        ax_equity = fig.add_subplot(gs[0, 0])

        # Plot equity curves
        ax_equity.plot(strategy_equity.index, strategy_equity.values,
                       label='Strategy', linewidth=2, color='blue')
        ax_equity.plot(benchmark_equity.index, benchmark_equity.values,
                       label=benchmark_label, linewidth=2, color='black', alpha=0.7)

        # Add colored backgrounds for different regimes
        regime_changes = []
        current_regime = None

        for date, regime in regimes.items():
            if regime != current_regime:
                regime_changes.append((date, regime))
                current_regime = regime

        # Add final date
        if regime_changes:
            regime_changes.append((regimes.index[-1], None))

        # Color the background regions
        for i in range(len(regime_changes) - 1):
            start_date = regime_changes[i][0]
            end_date = regime_changes[i + 1][0]
            regime = regime_changes[i][1]

            if regime in regime_colors:
                ax_equity.axvspan(start_date, end_date,
                                  alpha=0.2, color=regime_colors[regime],
                                  label=f"_{regime}" if i == 0 else "")

        # Add labels and legend for equity plot
        ax_equity.set_title('Overall Equity Curve with Market Regimes', fontsize=14)
        ax_equity.set_ylabel('Equity (Starting at 1.0)')
        ax_equity.grid(True, alpha=0.3)
        ax_equity.legend(loc='upper left')

        # Add regime legend (custom patches)
        import matplotlib.patches as mpatches
        regime_patches = []

        for regime, color in regime_colors.items():
            if regime in metrics and regime != 'undefined':
                patch = mpatches.Patch(color=color, alpha=0.2,
                                       label=f"{regime} ({metrics[regime]['count']} days)")
                regime_patches.append(patch)

        # Add second legend for regimes
        if regime_patches:
            ax_equity.legend(handles=regime_patches, loc='upper left',
                             title="Market Regimes")

        # Now create a grid for individual regime equity curves
        # Only include regimes with enough data (at least 5 days)
        valid_regimes = {r: data for r, data in metrics.items()
                         if r != 'undefined' and data['count'] >= 5}

        if len(valid_regimes) > 0:
            # Determine grid dimensions
            n_regimes = len(valid_regimes)
            nrows = math.ceil((n_regimes + 1) / ncols)  # +1 for regime definition table

            # Create a sub-GridSpec for the regime grid with reduced horizontal spacing and increased vertical spacing
            regime_gs = gs[1, 0].subgridspec(nrows=nrows, ncols=ncols, hspace=0.4, wspace=0.05)

            # Sort regimes by count (most data first)
            sorted_regimes = sorted(valid_regimes.items(), key=lambda x: x[1]['count'], reverse=True)

            # Track the position for the regime definitions table
            table_position = None

            for i, (regime, data) in enumerate(sorted_regimes):
                # Calculate grid position
                row = i // ncols
                col = i % ncols

                # Create subplot
                ax = fig.add_subplot(regime_gs[row, col])

                # Get regime returns
                strat_returns = data['returns']
                bench_returns = data['benchmark_returns']

                # Calculate combined equity curve from all regime days
                strat_equity = (1 + strat_returns).cumprod()
                bench_equity = (1 + bench_returns).cumprod()

                # Normalize to trading days - X-axis is day number, not actual date
                days = range(len(strat_equity))

                # Plot equity curves
                ax.plot(days, strat_equity.values,
                        label='Strategy', color='blue', linewidth=2)
                ax.plot(days, bench_equity.values,
                        label=benchmark_label, color='black', linewidth=1.5, alpha=0.7)

                # Fill between strategy & benchmark equity
                ax.fill_between(
                    days,
                    strat_equity.values,
                    bench_equity.values,
                    where=(strat_equity.values >= bench_equity.values),
                    color='green',
                    alpha=0.2,
                    interpolate=True
                )
                ax.fill_between(
                    days,
                    strat_equity.values,
                    bench_equity.values,
                    where=(strat_equity.values < bench_equity.values),
                    color='red',
                    alpha=0.2,
                    interpolate=True
                )

                # Add labels
                regime_color = regime_colors.get(regime, 'blue')
                days_count = data['count']

                # Calculate outperformance based on final cumulative returns (not annualized)
                strat_final_return = strat_equity.iloc[-1]
                bench_final_return = bench_equity.iloc[-1]
                outperf_pct = (strat_final_return / bench_final_return) - 1
                sign = "+" if outperf_pct >= 0 else ""

                # Simplified title with correct outperformance between final equity values
                title = f"{regime} ({days_count} days, {sign}{outperf_pct:.2%})"

                ax.set_title(title, color=regime_color, fontweight='bold')

                # Show y-axis on the leftmost plots and rightmost plots
                if col == 0:  # Left column
                    ax.set_ylabel("Equity (Starting at 1.0)")
                elif col == ncols - 1:  # Right column
                    # Create a twin axis for the rightmost plots with their own scale
                    ax_right = ax.twinx()
                    ax_right.set_ylabel("Equity (Starting at 1.0)")
                    # Make the right y-axis have the same limits as the left
                    ax_right.set_ylim(ax.get_ylim())
                    # Hide tick labels but keep the axis label
                    ax_right.tick_params(axis='y', labelleft=False)
                else:
                    # Middle columns don't need y-axis labels or ticks
                    ax.yaxis.set_ticklabels([])

                # Add a grid
                ax.grid(True, alpha=0.3)

                # Add a small legend
                ax.legend(fontsize=8, loc='best')

                # Format tick labels to be smaller
                ax.tick_params(axis='both', which='major', labelsize=8)

                # Keep track of last position for regime definitions table
                table_position = (row, col)

            # Calculate position for regime definitions table (last lower-right cell)
            last_row = nrows - 1
            last_col = ncols - 1

            # If there's a free cell in the lower right, use it for the table
            if table_position != (last_row, last_col):
                ax_table = fig.add_subplot(regime_gs[last_row, last_col])

                # Create regime definitions table
                regimes_to_show = sorted([r for r in regime_colors if r != 'undefined'])
                colors = [regime_colors.get(r, 'blue') for r in regimes_to_show]

                # Get descriptions from regime definitions
                descs = [self.regime_definitions.get(r, {}).get('description', 'Custom regime') for r in
                         regimes_to_show]

                # Format cell colors
                cell_colors = []
                for r in range(len(regimes_to_show) + 1):  # +1 for header row
                    if r == 0:
                        # Header row has no color
                        cell_colors.append(['none', 'none'])
                    else:
                        # Color first cell of data rows with regime color (at 0.3 alpha)
                        reg_color = colors[r - 1]
                        # Convert color name to rgba with alpha
                        from matplotlib.colors import to_rgba
                        rgba_color = to_rgba(reg_color, 0.3)
                        cell_colors.append([rgba_color, 'none'])

                # Create the table
                ax_table.axis('off')  # Hide the axis
                table = ax_table.table(
                    cellText=[['Regime', 'Description']] + list(zip(regimes_to_show, descs)),
                    colWidths=[0.3, 0.7],
                    cellLoc='left',
                    loc='center',
                    cellColours=cell_colors
                )

                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)  # Make cells a bit taller

                # Make header row bold
                for (i, j), cell in table.get_celld().items():
                    if i == 0:  # Header row
                        cell.set_text_props(fontweight='bold')
                    if j == 0 and i > 0:  # First column (regime names)
                        cell.set_text_props(fontweight='bold')

        plt.tight_layout()
        plt.show()

    def generate_report(self, results, analysis=None, ann_factor=252):
        """
        Generate a text report summarizing the regime analysis.

        Parameters
        ----------
        results : dict
            Results from the backtester
        analysis : dict, optional
            Analysis results (if None, will run analyze() first)
        ann_factor : int, optional
            Annualization factor (default: 252 for daily data)

        Returns
        -------
        str
            Text report of the analysis
        """
        if analysis is None:
            analysis = self.analyze(results, ann_factor=ann_factor)

        regimes = analysis['regimes']
        metrics = analysis['metrics']

        lines = []
        lines.append("\n==== MARKET REGIME ANALYSIS ====\n")

        # Overall regime distribution
        lines.append("--- Regime Distribution ---")
        total_days = len(regimes)

        for regime, count in sorted(analysis['regime_counts'].items(),
                                    key=lambda x: x[1], reverse=True):
            if regime != 'undefined':
                percentage = count / total_days * 100
                n_periods = metrics[regime]['n_periods'] if regime in metrics else 0
                lines.append(f"{regime}: {count} days ({percentage:.1f}%), {n_periods} distinct periods")

        lines.append("")

        # Performance metrics by regime
        lines.append("--- Performance by Regime ---")

        # Create a header for the table
        header = (f"{'Regime':<15} | {'Days':<5} | {'Return':<8} | {'Bench':<8} | "
                  f"{'Alpha':<8} | {'Sharpe':<6} | {'B.Sharpe':<8} | {'Corr':<5} | {'Win %':<5}")
        lines.append(header)
        lines.append("-" * len(header))

        for regime, data in sorted(metrics.items(),
                                   key=lambda x: x[1]['annualized_return'],
                                   reverse=True):
            # Skip undefined regime
            if regime == 'undefined':
                continue

            # Format the metrics
            days = data['count']
            ret = data['annualized_return']
            bench_ret = data['benchmark_annualized']
            alpha = data['annualized_outperformance']
            sharpe = data['sharpe']
            bench_sharpe = data['benchmark_sharpe']
            corr = data['correlation']
            win_rate = data['win_rate']

            # Add the row to the table
            lines.append(
                f"{regime:<15} | {days:<5d} | {ret:>7.2%} | {bench_ret:>7.2%} | "
                f"{alpha:>7.2%} | {sharpe:>5.2f} | {bench_sharpe:>7.2f} | {corr:>4.2f} | {win_rate:>5.1%}"
            )

        lines.append("")

        # Focus on outperformance by regime
        lines.append("--- Regime Outperformance Analysis ---")

        # Sort regimes by outperformance
        outperf_sorted = sorted(
            [(regime, data['annualized_outperformance'], data['count'], data['n_periods'])
             for regime, data in metrics.items() if regime != 'undefined'],
            key=lambda x: x[1],
            reverse=True
        )

        # Display outperformance by regime
        for regime, outperf, days, n_periods in outperf_sorted:
            sign = "+" if outperf >= 0 else ""
            lines.append(f"{regime} ({days} days, {n_periods} periods): {sign}{outperf:.2%} outperformance")

        lines.append("")

        # Regime stability analysis
        lines.append("--- Regime Stability Analysis ---")

        for regime, data in metrics.items():
            if regime != 'undefined':
                n_periods = data['n_periods']
                avg_days = data['count'] / n_periods if n_periods > 0 else 0
                periods = data['periods']

                if periods:
                    max_period = max(periods, key=lambda x: x['duration'])
                    max_days = max_period['duration']
                    lines.append(f"{regime}: {n_periods} periods, avg {avg_days:.1f} days per period, "
                                 f"longest period: {max_days} days")

        lines.append("")

        # Summary
        lines.append("--- Strategy Robustness Summary ---")

        # Calculate regime Sharpe ratio relative to benchmark
        relative_sharpes = {}
        for regime, data in metrics.items():
            if regime != 'undefined':
                strat_sharpe = data['sharpe']
                bench_sharpe = data['benchmark_sharpe']
                relative_sharpes[regime] = strat_sharpe - bench_sharpe

        # Find best and worst regimes
        best_regime = max(relative_sharpes.items(), key=lambda x: x[1])
        worst_regime = min(relative_sharpes.items(), key=lambda x: x[1])

        lines.append(f"Best performing regime: {best_regime[0]} "
                     f"(Sharpe diff: {best_regime[1]:+.2f})")
        lines.append(f"Worst performing regime: {worst_regime[0]} "
                     f"(Sharpe diff: {worst_regime[1]:+.2f})")

        # Calculate volatility of strategy performance across regimes
        if len(metrics) >= 2:
            regime_returns = [data['annualized_return'] for data in metrics.values()]
            regime_perf_vol = np.std(regime_returns)

            lines.append(f"Volatility of performance across regimes: {regime_perf_vol:.2%}")

            # Provide an assessment of robustness
            if regime_perf_vol < 0.05:
                lines.append("Assessment: Highly robust across different market regimes")
            elif regime_perf_vol < 0.10:
                lines.append("Assessment: Good robustness across different market regimes")
            elif regime_perf_vol < 0.15:
                lines.append("Assessment: Moderate robustness, some regime dependence")
            elif regime_perf_vol < 0.25:
                lines.append("Assessment: Limited robustness, significant regime dependence")
            else:
                lines.append("Assessment: Poor robustness, extreme regime dependence")

        return "\n".join(lines)
