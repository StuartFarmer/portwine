# .gitignore

```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# IDEs and editors
.vscode/
.idea/

# macOS system files
.DS_Store

# virtual environments
.venv/
venv/
env/
ENV/

# Poetry
poetry.lock


# Coverage reports
htmlcov/
.coverage
.coverage.*

# Jupyter Notebook
.ipynb_checkpoints

# Logs
*.log

# Build artifacts
build/
dist/
*.egg-info/

```

# portwine/__init__.py

```py
from portwine.strategies.base import StrategyBase
from portwine.backtester import Backtester, BenchmarkTypes, benchmark_equal_weight, benchmark_markowitz

```

# portwine/analyzers/__init__.py

```py
from portwine.analyzers.base import Analyzer
from portwine.analyzers.equitydrawdown import EquityDrawdownAnalyzer
from portwine.analyzers.gridequitydrawdown import GridEquityDrawdownAnalyzer
from portwine.analyzers.montecarlo import MonteCarloAnalyzer
from portwine.analyzers.seasonality import SeasonalityAnalyzer
from portwine.analyzers.correlation import CorrelationAnalyzer
from portwine.analyzers.traintest import TrainTestEquityDrawdownAnalyzer
from portwine.analyzers.strategycomparison import StrategyComparisonAnalyzer
from portwine.analyzers.studentttest import StudentsTTestAnalyzer

```

# portwine/analyzers/base.py

```py
class Analyzer:
    def analyze(self, results, *args, **kwargs):
        # Analyze method. Should return some sort of dataframe, etc
        raise NotImplementedError

    def plot(self, results, *args, **kwargs):
        # Optional visualization method
        raise NotImplementedError

```

# portwine/analyzers/correlation.py

```py
from portwine.analyzers import Analyzer
import matplotlib.pyplot as plt

class CorrelationAnalyzer(Analyzer):
    """
    Computes and plots correlation among the tickers' daily returns.

    Usage:
      1) correlation_dict = analyzer.analyze(results)
      2) analyzer.plot(results)

    'results' should be the dictionary from the backtester, containing:
        'tickers_returns': DataFrame of daily returns for each ticker
                           (columns = ticker symbols, index = dates)
    """

    def __init__(self, method='pearson'):
        """
        Parameters
        ----------
        method : str
            Correlation method (e.g. 'pearson', 'spearman', 'kendall').
        """
        self.method = method

    def analyze(self, results):
        """
        Generates a correlation matrix of the daily returns among all tickers.

        Parameters
        ----------
        results : dict
            {
              'tickers_returns': DataFrame of daily returns per ticker
              ...
            }

        Returns
        -------
        analysis_dict : dict
            {
              'correlation_matrix': DataFrame (square) of correlations
            }
        """
        tickers_returns = results.get('tickers_returns')
        if tickers_returns is None or tickers_returns.empty:
            print("Error: 'tickers_returns' missing or empty in results.")
            return None

        # Compute correlation
        corr_matrix = tickers_returns.corr(method=self.method)

        return {
            'correlation_matrix': corr_matrix
        }

    def plot(self, results):
        """
        Plots a heatmap of the correlation matrix.

        Parameters
        ----------
        results : dict
            The same dictionary used in 'analyze', containing 'tickers_returns'.
        """
        analysis_dict = self.analyze(results)
        if analysis_dict is None:
            print("No correlation data to plot.")
            return

        corr_matrix = analysis_dict['correlation_matrix']
        if corr_matrix.empty:
            print("Correlation matrix is empty. Nothing to plot.")
            return

        # Plot the correlation matrix as a heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr_matrix, aspect='auto')
        fig.colorbar(cax)

        # Set tick marks for each ticker
        tickers = corr_matrix.columns
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=45, ha='left')
        ax.set_yticklabels(tickers)

        ax.set_title("Correlation Matrix", pad=20)
        plt.tight_layout()
        plt.show()

```

# portwine/analyzers/equitydrawdown.py

```py
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
            color='mediumblue',   # deeper blue
            linewidth=1,         # a bit thicker
            alpha=0.6
        )
        ax1.plot(
            benchmark_equity_curve.index,
            benchmark_equity_curve.values,
            label=benchmark_label,
            color='black',      # black
            linewidth=0.5,         # a bit thinner
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
            color='mediumblue',   # deeper blue
            linewidth=1,         # a bit thicker
            alpha=0.6
        )
        ax2.plot(
            bm_dd.index,
            bm_dd.values,
            label=f"{benchmark_label} DD (%)",
            color='black',      # black
            linewidth=0.5,         # a bit thinner
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
        stats = self.analyze(results, ann_factor)

        strategy_stats = stats['strategy_stats']
        benchmark_stats = stats['benchmark_stats']

        print("\n=== Strategy Summary ===")
        for k, v in strategy_stats.items():
            if k in ["CAGR", "AnnualVol", "MaxDrawdown"]:
                print(f"{k}: {v:.2%}")
            elif k == "Sharpe":
                print(f"{k}: {v:.2f}")
            else:
                print(f"{k}: {v:.2%}")

        print(f"\n=== {benchmark_label} Summary ===")
        for k, v in benchmark_stats.items():
            if k in ["CAGR", "AnnualVol", "MaxDrawdown"]:
                print(f"{k}: {v:.2%}")
            elif k == "Sharpe":
                print(f"{k}: {v:.2f}")
            else:
                print(f"{k}: {v:.2%}")

        # Now show a comparison: percentage difference (Strategy vs. Benchmark).
        print("\n=== Strategy vs. Benchmark (Percentage Difference) ===")
        for k in strategy_stats.keys():
            strat_val = strategy_stats.get(k, None)
            bench_val = benchmark_stats.get(k, None)
            if strat_val is None or bench_val is None:
                print(f"{k}: N/A (missing data)")
                continue

            if isinstance(strat_val, (int, float)) and isinstance(bench_val, (int, float)):
                if abs(bench_val) > 1e-15:
                    diff = (strat_val - bench_val) / abs(bench_val)
                    print_val = f"{diff * 100:.2f}%"
                else:
                    print_val = "N/A (benchmark ~= 0)"
            else:
                print_val = "N/A (non-numeric)"

            print(f"{k}: {print_val}")

```

# portwine/analyzers/montecarlo.py

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portwine.analyzers.base import Analyzer

class MonteCarloAnalyzer(Analyzer):
    """
    Runs Monte Carlo simulations on a strategy's returns (with replacement).
    Allows plotting of all paths plus confidence bands (5%-95%) and a mean path,
    on a log scale. Can also plot a benchmark curve for comparison.
    """

    def __init__(self, frequency='ME'):
        assert frequency in ['ME', 'D'], 'Only supports ME (monthly) or D (daily) frequencies'
        self.frequency = frequency
        self.ann_factor = 12 if frequency == 'ME' else 252

    def _compute_drawdown(self, equity):
        rolling_max = equity.cummax()
        dd = (equity - rolling_max) / rolling_max
        return dd

    def _convert_to_monthly(self, daily_returns):
        if not isinstance(daily_returns.index, pd.DatetimeIndex):
            daily_returns.index = pd.to_datetime(daily_returns.index)
        monthly = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        return monthly

    def analyze(self, returns):
        if len(returns) < 2:
            return {
                'CumulativeReturn': np.nan,
                'AnnVol': np.nan,
                'Sharpe': np.nan,
                'MaxDrawdown': np.nan
            }

        cumret = (1 + returns).prod() - 1.0
        vol = returns.std() * np.sqrt(self.ann_factor)
        mean_ret = returns.mean()
        sharpe = (mean_ret / returns.std()) * np.sqrt(self.ann_factor) if returns.std() != 0 else np.nan

        equity = (1 + returns).cumprod()
        dd = self._compute_drawdown(equity)
        max_dd = dd.min()

        return {
            'CumulativeReturn': cumret,
            'AnnVol': vol,
            'Sharpe': sharpe,
            'MaxDrawdown': max_dd
        }

    def mc_with_replacement(self, ret_series, n_sims=100, random_seed=42):
        """
        Example method that bootstraps returns and avoids the repeated insert.
        """
        np.random.seed(random_seed)
        returns_array = ret_series.values
        n = len(returns_array)

        all_equities = []  # We'll store each path's equity Series or array here
        sim_stats = []

        for _ in range(n_sims):
            indices = np.random.choice(range(n), size=n, replace=True)
            sampled = returns_array[indices]
            sim_returns = pd.Series(sampled, index=ret_series.index)
            eq = (1 + sim_returns).cumprod()
            # Instead of adding a column to a DataFrame in each iteration,
            # we collect each path for now:
            all_equities.append(eq)
            sim_stats.append(self.analyze(sim_returns))

        # Now, combine them all at once. For example:
        #   1) convert each path to a DataFrame with a single column
        #   2) pd.concat them horizontally (axis=1)
        sim_equity = pd.concat(
            [path.to_frame(name=f"Sim_{i}") for i, path in enumerate(all_equities)],
            axis=1
        )

        orig_stats = self.analyze(ret_series)
        return {
            'simulated_stats': sim_stats,
            'simulated_equity_curves': sim_equity,
            'original_stats': orig_stats
        }

        # return sim_equity

    def get_periodic_returns(self, results):
        """
        Extract or compute the returns to feed into the Monte Carlo simulations.

        By default, tries monthly if freq='ME'. If you already have monthly
        or daily in 'strategy_daily_returns', you can keep or transform them.

        Parameters
        ----------
        results : dict
            Dictionary from the backtester with keys:
            {
                'strategy_daily_returns': pd.Series (index=dates),
                'equity_curve': pd.Series,
                ...
            }
        freq : str
            Frequency to convert daily returns (e.g. 'ME' for monthly).
            If None, uses daily returns as is.

        Returns
        -------
        pd.Series
            Returns at the desired frequency.
        """
        daily = results.get('strategy_returns', pd.Series(dtype=float))
        if daily.empty:
            print("No strategy_daily_returns found in results.")
            return pd.Series(dtype=float)

        if self.frequency == 'ME':
            return self._convert_to_monthly(daily)
        else:
            # Return daily as is
            return daily

    def plot(self, results, title="Monte Carlo Simulations (Log Scale)"):
        """
        Plots all visualizations on a single figure with 5 subplots:
        - Main plot: Simulated equity paths in black with very low alpha,
          along with a 5%-95% confidence band in shaded area, plus a mean path,
          on a log scale, optionally with a benchmark.
        - Four smaller plots: Histograms showing the distribution of performance metrics:
          - Cumulative Return
          - Annual Vol
          - Sharpe
          - Max Drawdown
          and, if 'benchmark_returns' is provided, a vertical line to compare
          the benchmark's metric in each histogram.

        Parameters
        ----------
        results : dict
            {
                'benchmark_returns': DataFrame or Series with benchmark returns
            }
        title : str
            Chart title.
        """
        # Generate the simulation data
        periodic_returns = self.get_periodic_returns(results)
        mc_results = self.mc_with_replacement(periodic_returns, n_sims=200)

        sim_equity = mc_results['simulated_equity_curves']
        if sim_equity.empty:
            print("No simulation equity curves to plot.")
            return

        # Create a single figure with GridSpec for layout control
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 4)  # 3 rows, 4 columns grid

        # Main equity curve plot takes up the top 2 rows
        ax_main = fig.add_subplot(gs[0:2, :])

        # Plot all paths in black, alpha=0.01
        ax_main.plot(sim_equity.index, sim_equity.values,
                     color='black', alpha=0.01, linewidth=0.8)

        # Confidence bands
        lo5 = sim_equity.quantile(0.05, axis=1)
        hi95 = sim_equity.quantile(0.95, axis=1)
        mean_path = sim_equity.mean(axis=1)

        ax_main.fill_between(sim_equity.index, lo5, hi95,
                             color='blue', alpha=0.2,
                             label='5%-95% Confidence Band')
        ax_main.plot(mean_path.index, mean_path.values,
                     color='red', linewidth=2, label='Mean Path')

        benchmark_equity = None
        if 'benchmark_returns' in results and results['benchmark_returns'] is not None:
            benchmark_equity = (1 + results['benchmark_returns']).cumprod()

        # Plot benchmark if provided
        if benchmark_equity is not None and not benchmark_equity.empty:
            ax_main.plot(benchmark_equity.index, benchmark_equity.values,
                         color='green', linewidth=2, label='Benchmark')

        # Log scale
        ax_main.set_yscale('log')
        ax_main.set_title(title, fontsize=14)
        ax_main.set_ylabel("Equity (log scale)")
        ax_main.legend(loc='best')
        ax_main.grid(True)

        # Get performance stats for histograms
        simulated_stats = mc_results.get('simulated_stats', [])
        if not simulated_stats:
            # If we have no performance stats, there's nothing more to plot
            plt.tight_layout()
            plt.show()
            return

        # Prepare data for histograms
        cumulative_returns = [d['CumulativeReturn'] for d in simulated_stats]
        ann_vols = [d['AnnVol'] for d in simulated_stats]
        sharpes = [d['Sharpe'] for d in simulated_stats]
        max_dds = [d['MaxDrawdown'] for d in simulated_stats]

        # If benchmark_returns is provided, compute same stats
        benchmark_stats = {}
        if 'benchmark_returns' in results and results['benchmark_returns'] is not None and not results[
            'benchmark_returns'].empty:
            benchmark_stats = self.analyze(results['benchmark_returns'])

        # Create the 4 histogram subplots in the bottom row
        axes = [
            fig.add_subplot(gs[2, 0]),  # Cumulative Return
            fig.add_subplot(gs[2, 1]),  # Annual Vol
            fig.add_subplot(gs[2, 2]),  # Sharpe
            fig.add_subplot(gs[2, 3])  # Max Drawdown
        ]

        # 1) Cumulative Return
        axes[0].hist(cumulative_returns, bins=30, color='blue', alpha=0.7)
        axes[0].set_title("Cumulative Return", fontsize=10)
        if 'CumulativeReturn' in benchmark_stats and not np.isnan(benchmark_stats['CumulativeReturn']):
            cr_bench = benchmark_stats['CumulativeReturn']
            axes[0].axvline(cr_bench, color='green', linestyle='--',
                            label=f"Benchmark={cr_bench:.2f}")
            axes[0].legend(fontsize=8)

        # 2) Annual Vol
        axes[1].hist(ann_vols, bins=30, color='blue', alpha=0.7)
        axes[1].set_title("Annual Volatility", fontsize=10)
        if 'AnnVol' in benchmark_stats and not np.isnan(benchmark_stats['AnnVol']):
            av_bench = benchmark_stats['AnnVol']
            axes[1].axvline(av_bench, color='green', linestyle='--',
                            label=f"Benchmark={av_bench:.2f}")
            axes[1].legend(fontsize=8)

        # 3) Sharpe
        axes[2].hist(sharpes, bins=30, color='blue', alpha=0.7)
        axes[2].set_title("Sharpe Ratio", fontsize=10)
        if 'Sharpe' in benchmark_stats and not np.isnan(benchmark_stats['Sharpe']):
            sh_bench = benchmark_stats['Sharpe']
            axes[2].axvline(sh_bench, color='green', linestyle='--',
                            label=f"Benchmark={sh_bench:.2f}")
            axes[2].legend(fontsize=8)

        # 4) Max Drawdown
        axes[3].hist(max_dds, bins=30, color='blue', alpha=0.7)
        axes[3].set_title("Max Drawdown", fontsize=10)
        if 'MaxDrawdown' in benchmark_stats and not np.isnan(benchmark_stats['MaxDrawdown']):
            dd_bench = benchmark_stats['MaxDrawdown']
            axes[3].axvline(dd_bench, color='green', linestyle='--',
                            label=f"Benchmark={dd_bench:.2f}")
            axes[3].legend(fontsize=8)

        # Adjust layout to ensure all elements fit well
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.show()

```

# portwine/analyzers/seasonality.py

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
from scipy import stats
from statsmodels.stats.multitest import multipletests
from portwine.analyzers.base import Analyzer

class SeasonalityAnalyzer(Analyzer):
    """
    A Seasonality Analyzer that:
      1) Adds time-based features (day_of_week, month, quarter, year, etc.)
         and special "turn-of" indicators for month and quarter (±3 days).
      2) Uses a generic seasonality analysis method to handle day_of_week,
         month, quarter, year, or any other time grouping.
      3) Plots each grouping in a temporal (natural) order on the x-axis
         (e.g., Monday->Sunday, Jan->Dec, Q1->Q4, ascending years, etc.).
    """

    def analyze(self, results, significance_level=0.05, benchmark_comparison=True):
        """
        Performs seasonality analysis on strategy & optional benchmark returns.

        Parameters
        ----------
        results : dict
            Backtester results dict with keys:
                'strategy_returns': pd.Series
                'benchmark_returns': pd.Series (optional)
        significance_level : float
            Alpha level for statistical tests (default: 0.05).
        benchmark_comparison : bool
            Whether to compare strategy against benchmark.

        Returns
        -------
        analysis_results : dict
            {
                'day_of_week': { 'stats': DataFrame, ... },
                'month_of_year': { ... },
                'quarter': { ... },
                'year': { ... },
                'turn_of_month': { ... },
                'turn_of_quarter': { ... }
            }
        """
        strategy_returns = results.get('strategy_returns', pd.Series(dtype=float))
        benchmark_returns = (results.get('benchmark_returns', pd.Series(dtype=float))
                             if benchmark_comparison else None)

        # Ensure DatetimeIndex
        if not isinstance(strategy_returns.index, pd.DatetimeIndex):
            strategy_returns.index = pd.to_datetime(strategy_returns.index)
        if benchmark_returns is not None and not isinstance(benchmark_returns.index, pd.DatetimeIndex):
            benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

        # Build DataFrames with features
        strat_df = self._add_time_features(strategy_returns)
        bench_df = self._add_time_features(benchmark_returns) if benchmark_returns is not None else None

        # We define a dictionary mapping each "analysis label" to the column & optional display name
        time_periods = {
            'day_of_week': {'group_col': 'day_of_week', 'display_col': 'day_name'},
            'month_of_year': {'group_col': 'month', 'display_col': 'month_name'},
            'quarter': {'group_col': 'quarter', 'display_col': None},
            'year': {'group_col': 'year', 'display_col': None},
        }

        analysis_results = {}

        # Generic analysis for day_of_week, month_of_year, quarter, year
        for label, info in time_periods.items():
            analysis_results[label] = self._analyze_seasonality(
                df=strat_df,
                column=info['group_col'],
                display_name_column=info['display_col'],
                benchmark_df=bench_df,
                alpha=significance_level
            )

        # Turn-of-month and turn-of-quarter
        analysis_results['turn_of_month'] = self._analyze_turn(
            strat_df, bench_df, prefix='tom', alpha=significance_level
        )
        analysis_results['turn_of_quarter'] = self._analyze_turn(
            strat_df, bench_df, prefix='toq', alpha=significance_level
        )

        return analysis_results

    def plot(self, results, analysis_results=None, figsize=(15, 18)):
        """
        Plot each seasonal grouping in subplots, with a temporal or natural ordering on the x-axis.

        Parameters
        ----------
        results : dict
            Backtester results dict (see analyze).
        analysis_results : dict, optional
            Results from analyze() method (will be computed if None).
        figsize : tuple
            Size of the entire figure.
        """
        if analysis_results is None:
            analysis_results = self.analyze(results)

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)

        # For each row/col, we define a label & chart title
        subplot_map = [
            ('day_of_week', (0, 0), "Day of Week"),
            ('month_of_year', (0, 1), "Month of Year"),
            ('quarter', (1, 0), "Quarter"),
            ('year', (1, 1), "Year"),
            ('turn_of_month', (2, 0), "Turn of Month"),
            ('turn_of_quarter', (2, 1), "Turn of Quarter")
        ]

        for key, (r, c), title in subplot_map:
            ax = axes[r][c]
            self._plot_seasonality(analysis_results.get(key), title=title, ax=ax)

        plt.tight_layout()
        plt.show()

    def generate_report(self, results, analysis_results=None, alpha=0.05):
        """
        Generate a text report summarizing the analysis.

        Parameters
        ----------
        results : dict
            Backtester results dict (see analyze).
        analysis_results : dict, optional
            Results from analyze() method (will be computed if None).
        alpha : float
            Significance level (default: 0.05).

        Returns
        -------
        str
            Multi-line text report.
        """
        if analysis_results is None:
            analysis_results = self.analyze(results, significance_level=alpha)

        lines = []
        lines.append("=== SEASONALITY ANALYSIS REPORT ===\n")

        strat_ret = results.get('strategy_returns', pd.Series(dtype=float))
        if not strat_ret.empty:
            lines.append(f"Overall Mean Return: {strat_ret.mean():.4%}")
            lines.append(f"Overall Positive Days: {(strat_ret > 0).mean():.2%}")
            lines.append(f"Total Days Analyzed: {len(strat_ret)}")
        lines.append("")

        label_map = {
            'day_of_week': "DAY OF WEEK",
            'month_of_year': "MONTH OF YEAR",
            'quarter': "QUARTER",
            'year': "YEAR",
            'turn_of_month': "TURN OF MONTH",
            'turn_of_quarter': "TURN OF QUARTER"
        }

        for key, title in label_map.items():
            lines.append(f"=== {title} ANALYSIS ===")
            result_dict = analysis_results.get(key)
            lines.extend(self._format_seasonality_report(result_dict, label_name=title))
            lines.append("")

        return "\n".join(lines)

    ###########################################################################
    # Internals
    ###########################################################################
    def _add_time_features(self, returns_series):
        """
        Adds standard time-based features to a returns Series (day, month, quarter, etc.)
        Also adds flags for turn-of-month and turn-of-quarter from T-3 to T+3.
        """
        if returns_series is None or returns_series.empty:
            return None

        df = pd.DataFrame({'returns': returns_series})

        # Basic time features
        df['day_of_week'] = df.index.dayofweek  # 0=Monday
        df['day_name'] = df.index.day_name()
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['month_name'] = df.index.month_name()
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Reorder categories for day_of_week and month_name
        # so plots can follow chronological order if we choose to map them
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_name'] = pd.Categorical(df['day_name'], categories=day_order, ordered=True)

        month_order = [calendar.month_name[i] for i in range(1, 13)]
        df['month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)

        # Mark turn-of-month and turn-of-quarter
        self._mark_turn(df, prefix='tom', period_months=range(1, 13))  # All months
        self._mark_turn(df, prefix='toq', period_months=[1, 4, 7, 10])  # Quarter start months
        return df

    def _mark_turn(self, df, prefix, period_months, day_range=range(-3, 4)):
        """
        Mark turn-of-month or turn-of-quarter:
          prefix='tom' or 'toq'
          period_months: which months define a "turn" (1..12 or [1,4,7,10])
          day_range: e.g. -3..+3
        """
        # Identify the 'first day' of each relevant month
        is_start = df.index.month.isin(period_months) & (df.index.day == 1)

        for offset in day_range:
            col_name = f"{prefix}_{offset}"
            df[col_name] = False
            for idx in df.index[is_start]:
                shifted_idx = idx + pd.Timedelta(days=offset)
                if shifted_idx in df.index:
                    df.loc[shifted_idx, col_name] = True

    def _analyze_turn(self, strategy_df, benchmark_df, prefix='tom', alpha=0.05):
        """
        Analyze returns around turn-of-month (tom) or turn-of-quarter (toq) ±3 days
        by grouping them into a single factor column (e.g. 'tom_day').
        """
        if strategy_df is None or strategy_df.empty:
            return None

        offset_labels = {
            f'{prefix}_-3': 'T-3',
            f'{prefix}_-2': 'T-2',
            f'{prefix}_-1': 'T-1',
            f'{prefix}_0': 'T+0',
            f'{prefix}_1': 'T+1',
            f'{prefix}_2': 'T+2',
            f'{prefix}_3': 'T+3'
        }

        strat_frames = []
        bench_frames = []

        for col, label in offset_labels.items():
            if col in strategy_df.columns:
                subset_strat = strategy_df.loc[strategy_df[col], :].copy()
                if not subset_strat.empty:
                    subset_strat[f'{prefix}_day'] = label
                    strat_frames.append(subset_strat)

            if benchmark_df is not None and not benchmark_df.empty and col in benchmark_df.columns:
                subset_bench = benchmark_df.loc[benchmark_df[col], :].copy()
                if not subset_bench.empty:
                    subset_bench[f'{prefix}_day'] = label
                    bench_frames.append(subset_bench)

        if not strat_frames:
            return None

        df_strat_all = pd.concat(strat_frames)
        df_bench_all = pd.concat(bench_frames) if bench_frames else None

        return self._analyze_seasonality(
            df_strat_all,
            column=f'{prefix}_day',
            benchmark_df=df_bench_all,
            alpha=alpha
        )

    def _analyze_seasonality(self,
                             df,
                             column,
                             value_column='returns',
                             benchmark_df=None,
                             alpha=0.05,
                             display_name_column=None):
        """
        Generic seasonality analysis for a given column (e.g., 'day_of_week', 'month', etc.).
        Groups by that column, computes means, medians, stats, t-tests, etc.
        Optionally compares strategy vs. benchmark if benchmark_df is provided.
        """
        if df is None or df.empty:
            return None

        grouped = df.groupby(column)[value_column]
        stats_df = pd.DataFrame({
            'mean': grouped.mean(),
            'median': grouped.median(),
            'std': grouped.std(),
            'count': grouped.count(),
            'positive_pct': grouped.apply(lambda x: (x > 0).mean()),
            'cumulative': grouped.apply(lambda x: (1 + x).prod() - 1),
        })

        # Overall stats
        overall_mean = df[value_column].mean()
        overall_std = df[value_column].std()
        overall_n = len(df)

        # T-tests: group vs. non-group
        t_stats, p_values = [], []
        for group_val in stats_df.index:
            group_data = df.loc[df[column] == group_val, value_column]
            non_group_data = df.loc[df[column] != group_val, value_column]
            t_stat, p_val = stats.ttest_ind(group_data, non_group_data, equal_var=False)
            t_stats.append(t_stat)
            p_values.append(p_val)

        stats_df['t_stat'] = t_stats
        stats_df['p_value'] = p_values
        stats_df['significant'] = stats_df['p_value'] < alpha
        # Effect size
        stats_df['effect_size'] = (stats_df['mean'] - overall_mean) / (overall_std if overall_std != 0 else 1e-9)

        # Multiple testing correction
        _, corr_pvals, _, _ = multipletests(stats_df['p_value'].values, alpha=alpha, method='fdr_bh')
        stats_df['corrected_p_value'] = corr_pvals
        stats_df['significant_corrected'] = stats_df['corrected_p_value'] < alpha

        # If benchmark is provided
        if benchmark_df is not None and not benchmark_df.empty:
            bm_grouped = benchmark_df.groupby(column)[value_column]
            bm_stats = pd.DataFrame({
                'benchmark_mean': bm_grouped.mean(),
                'benchmark_median': bm_grouped.median(),
                'benchmark_cumulative': bm_grouped.apply(lambda x: (1 + x).prod() - 1)
            })
            stats_df = stats_df.merge(bm_stats, left_index=True, right_index=True, how='left')

            # Strategy vs. Benchmark t-test within each group
            strat_vs_bm_t, strat_vs_bm_p = [], []
            for group_val in stats_df.index:
                strat_grp = df.loc[df[column] == group_val, value_column]
                bm_grp = benchmark_df.loc[benchmark_df[column] == group_val, value_column]
                if not strat_grp.empty and not bm_grp.empty:
                    t_stat, p_val = stats.ttest_ind(strat_grp, bm_grp, equal_var=False)
                else:
                    t_stat, p_val = np.nan, np.nan
                strat_vs_bm_t.append(t_stat)
                strat_vs_bm_p.append(p_val)

            stats_df['strat_vs_bm_t'] = strat_vs_bm_t
            stats_df['strat_vs_bm_p'] = strat_vs_bm_p
            stats_df['strat_vs_bm_significant'] = stats_df['strat_vs_bm_p'] < alpha

            # Correct these p-values
            finite_mask = stats_df['strat_vs_bm_p'].notnull()
            if finite_mask.sum() > 0:
                _, corr_bm_pvals, _, _ = multipletests(
                    stats_df.loc[finite_mask, 'strat_vs_bm_p'].values,
                    alpha=alpha, method='fdr_bh'
                )
                stats_df['strat_vs_bm_corrected_p'] = np.nan
                stats_df.loc[finite_mask, 'strat_vs_bm_corrected_p'] = corr_bm_pvals
                stats_df['strat_vs_bm_significant_corrected'] = (
                        stats_df['strat_vs_bm_corrected_p'] < alpha
                )

        # Remap to display names if requested
        if display_name_column and display_name_column in df.columns:
            # Build a map from numeric group_val -> display_name (like day_of_week -> Monday, etc.)
            unique_map = df[[column, display_name_column]].drop_duplicates()
            mapper = dict(zip(unique_map[column], unique_map[display_name_column]))
            # Re-map index to display
            new_index = []
            for val in stats_df.index:
                new_index.append(mapper[val] if val in mapper else val)
            stats_df.index = new_index

        # No longer automatically sorting by 'mean' descending.
        # Instead, we keep the natural/temporal order in the final plot step.

        return {
            'stats': stats_df,
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'overall_n': overall_n
        }

    def _plot_seasonality(self, seasonality_result, title, ax):
        """
        Plot a bar chart of the mean returns for each group in their temporal/natural order.
        If a benchmark is present, show it alongside strategy bars.
        """
        if seasonality_result is None:
            ax.text(0.5, 0.5, "No data available",
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
            return

        df_stats = seasonality_result['stats'].copy()
        if df_stats.empty:
            ax.text(0.5, 0.5, "No data available",
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
            return

        # Reorder the index in a temporal manner depending on 'title'
        df_stats = self._reorder_index_temporally(df_stats, title)

        x_labels = df_stats.index
        x_pos = np.arange(len(x_labels))
        mean_vals = df_stats['mean'].values
        has_benchmark = ('benchmark_mean' in df_stats.columns)

        width = 0.35 if has_benchmark else 0.5

        # Plot strategy bars
        bars_strat = ax.bar(
            x_pos - (width / 2 if has_benchmark else 0),
            mean_vals,
            width=width,
            label='Strategy'
        )

        # Color code significant bars
        if 'significant_corrected' in df_stats.columns:
            for i, sig in enumerate(df_stats['significant_corrected']):
                if sig:
                    bars_strat[i].set_color('green')
                    bars_strat[i].set_alpha(0.7)

        # Plot benchmark bars if available
        if has_benchmark:
            bench_vals = df_stats['benchmark_mean'].values
            bars_bench = ax.bar(
                x_pos + width / 2,
                bench_vals,
                width=width,
                label='Benchmark',
                color='orange',
                alpha=0.7
            )

        # Plot an overall mean line
        overall_mean = seasonality_result['overall_mean']
        ax.axhline(y=overall_mean, color='r', linestyle='--', alpha=0.5,
                   label=f'Overall Mean: {overall_mean:.4%}')

        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_ylabel('Mean Return')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend()

    def _reorder_index_temporally(self, df_stats, title):
        """
        Reorder df_stats.index in a natural or temporal way based on the chart title:
          - Day of Week: Monday->Sunday
          - Month of Year: use abbreviated month names (Jan->Dec)
          - Quarter: map numeric to Q1..Q4 or reorder Q1..Q4
          - Year: use the last two digits (e.g., 2021 -> 21)
          - Turn of Month or Quarter: T-3..T+3
        Only drop rows if their 'mean' is truly NaN, preserving partial data.
        """
        if df_stats is None or df_stats.empty:
            return df_stats

        idx_list = list(df_stats.index)

        # 1) Day of Week
        if "Day of Week" in title:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sorted_order = [d for d in day_order if d in idx_list]
            df_stats = df_stats.reindex(sorted_order)
            df_stats = df_stats.dropna(subset=["mean"])
            return df_stats

        # 2) Month of Year
        if "Month of Year" in title:
            # The DataFrame index currently has full month names like "January", "February", ...
            # We reorder them, then rename them to abbreviated forms.
            full_to_abbrev = {
                "January": "Jan", "February": "Feb", "March": "Mar",
                "April": "Apr", "May": "May", "June": "Jun",
                "July": "Jul", "August": "Aug", "September": "Sep",
                "October": "Oct", "November": "Nov", "December": "Dec"
            }

            month_order = list(full_to_abbrev.keys())  # [January, February, ..., December]
            # Reindex in chronological order
            sorted_order = [m for m in month_order if m in idx_list]
            df_stats = df_stats.reindex(sorted_order)
            df_stats = df_stats.dropna(subset=["mean"])

            # Now rename to abbreviations
            new_index = []
            for m in df_stats.index:
                if m in full_to_abbrev:
                    new_index.append(full_to_abbrev[m])
                else:
                    new_index.append(m)  # fallback if something unrecognized
            df_stats.index = new_index

            return df_stats

        # 3) Quarter
        if "Quarter" in title:
            # We might see numeric 1..4 or strings "Q1","Q2", etc.
            def is_qstring(x):
                return isinstance(x, str) and len(x) == 2 and x.upper().startswith("Q")

            string_items = [x for x in idx_list if isinstance(x, str)]
            all_qstrings = all(is_qstring(s) for s in string_items) and len(string_items) == len(idx_list)

            if all_qstrings:
                quarter_order = ["Q1", "Q2", "Q3", "Q4"]
                sorted_order = [q for q in quarter_order if q in idx_list]
                df_stats = df_stats.reindex(sorted_order)
                df_stats = df_stats.dropna(subset=["mean"])
                return df_stats

            # Otherwise parse numerics
            numeric_map = {}
            for item in idx_list:
                try:
                    numeric_map[item] = int(item)  # e.g. 1..4
                except:
                    numeric_map[item] = None

            valid_items = [k for k, v in numeric_map.items() if v in [1, 2, 3, 4]]
            valid_items.sort(key=lambda x: numeric_map[x])

            if valid_items:
                df_stats = df_stats.loc[valid_items]
                new_index = []
                for old in valid_items:
                    q_num = numeric_map[old]
                    new_index.append(f"Q{q_num}")
                df_stats.index = new_index
                df_stats = df_stats.dropna(subset=["mean"])
                return df_stats

            # Fallback: just drop rows that are NaN in mean
            df_stats = df_stats.dropna(subset=["mean"])
            return df_stats

        # 4) Year
        if "Year" in title:
            # We parse each index item as int => then sort ascending => rename to last two digits
            numeric_map = {}
            for item in idx_list:
                try:
                    numeric_map[item] = int(item)
                except:
                    numeric_map[item] = None

            valid_items = [k for k, v in numeric_map.items() if v is not None]
            # Sort ascending by parsed year
            valid_items.sort(key=lambda x: numeric_map[x])

            if valid_items:
                df_stats = df_stats.loc[valid_items]
                # rename to last two digits
                new_index = []
                for old in valid_items:
                    y = numeric_map[old]
                    # last two digits
                    y2 = y % 100
                    new_index.append(f"{y2:02d}")  # e.g. "21", "22"
                df_stats.index = new_index

            df_stats = df_stats.dropna(subset=["mean"])
            return df_stats

        # 5) Turn of Month / Quarter
        if "Turn of Month" in title or "Turn of Quarter" in title:
            turn_order = ["T-3", "T-2", "T-1", "T+0", "T+1", "T+2", "T+3"]
            sorted_order = [d for d in turn_order if d in idx_list]
            df_stats = df_stats.reindex(sorted_order)
            df_stats = df_stats.dropna(subset=["mean"])
            return df_stats

        # Otherwise, return as is
        df_stats = df_stats.dropna(subset=["mean"])
        return df_stats

    def _format_seasonality_report(self, seasonality_result, label_name):
        """
        Return a list of lines describing the seasonality results for a given period.
        """
        if (seasonality_result is None
                or 'stats' not in seasonality_result
                or seasonality_result['stats'].empty):
            return ["No data available.\n"]

        df_stats = seasonality_result['stats']

        lines = []
        lines.append(f"--- {label_name} ---")

        # We won't reorder the DataFrame here,
        # because the user specifically wanted temporal ordering in the plot,
        # not necessarily for the table. But we can do it here if desired:
        # For textual best/worst we might want them by "mean" descending:
        sorted_df = df_stats.sort_values('mean', ascending=False)

        # Best & worst
        best_idx = sorted_df.index[0]
        worst_idx = sorted_df.index[-1]
        best_val = sorted_df.iloc[0]['mean']
        worst_val = sorted_df.iloc[-1]['mean']

        lines.append(f"Best {label_name}: {best_idx} ({best_val:.4%})")
        lines.append(f"Worst {label_name}: {worst_idx} ({worst_val:.4%})\n")

        # Table header
        header = (f"{label_name:<15s} | {'Mean':>7s} | {'Median':>7s} | "
                  f"{'Win %':>5s} | {'Count':>5s} | {'Signif?'} ")
        lines.append(header)
        lines.append("-" * len(header))

        for idx, row in sorted_df.iterrows():
            is_sig = '*' if row.get('significant_corrected', False) else ''
            lines.append(
                f"{str(idx):<15s} | "
                f"{row['mean']:>7.2%} | "
                f"{row['median']:>7.2%} | "
                f"{row['positive_pct']:>5.2%} | "
                f"{int(row['count']):>5d} | {is_sig}"
            )

        lines.append("")
        lines.append("* indicates statistical significance (after FDR correction)")
        return lines

```

# portwine/analyzers/traintest.py

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from portwine.analyzers.base import Analyzer


class TrainTestEquityDrawdownAnalyzer(Analyzer):
    """
    A train/test analyzer that plots:
      - (Row 0) Strategy & benchmark equity curves with a vertical line marking
                the boundary between train & test sets.
      - (Row 1) Drawdown curves for strategy & benchmark with the same vertical line.
      - (Row 2) two columns:
          * Left: Overlaid histogram of train vs. test daily returns.
          * Right: Summary stats table for [CAGR, Vol, Sharpe, MaxDD, Calmar, Sortino].
            Columns = [Metric, Train, Test, Diff, Overfit Ratio].

    Specifics:
      1) Diff column:
         - For CAGR, Vol, MaxDD, Calmar: show the simple difference (test minus train) as a percentage.
           * Exception: For MaxDD, the difference is computed as the difference in absolute values.
         - For Sharpe and Sortino: show the simple difference as a raw value.
      2) Overfit Ratio:
         - For MaxDD => ratio = abs(testVal) / abs(trainVal).
         - For everything else => ratio = abs(trainVal) / abs(testVal).
         - The ratio is always positive.
         - Color coding: if ratio <= 1.1 => green; if <= 1.25 => yellow; else red.
      3) Extra metrics: Calmar, Sortino.
    """

    def _compute_drawdown(self, equity_series):
        """
        Computes drawdown from peak as a fraction (range: 0 to -1).
        Example: -0.20 => -20% drawdown from the peak.
        """
        rolling_max = equity_series.cummax()
        dd = (equity_series - rolling_max) / rolling_max
        return dd

    def _compute_summary_stats(self, daily_returns):
        """
        Returns a dict with:
          'CAGR'
          'Vol'
          'Sharpe'
          'MaxDD'   (negative, e.g. -0.25 => -25%)
          'Calmar'  (CAGR / abs(MaxDD))
          'Sortino' (annualized mean / annualized downside stdev)
        """
        if len(daily_returns) < 2:
            return {}

        dr = daily_returns.dropna()
        if dr.empty:
            return {}

        ann_factor = 252.0
        eq = (1.0 + dr).cumprod()
        end_val = eq.iloc[-1]
        n_days = len(dr)
        years = n_days / ann_factor

        # CAGR
        if years > 0 and end_val > 0:
            cagr_ = end_val ** (1.0 / years) - 1.0
        else:
            cagr_ = np.nan

        # Volatility
        std_ = dr.std()
        vol_ = std_ * np.sqrt(ann_factor) if std_ > 1e-9 else np.nan

        # Sharpe Ratio
        if vol_ and vol_ > 1e-9:
            sharpe_ = cagr_ / vol_
        else:
            sharpe_ = np.nan

        # Maximum Drawdown (negative)
        dd = self._compute_drawdown(eq)
        max_dd_ = dd.min()  # negative

        # Calmar Ratio = CAGR / |MaxDD|
        if max_dd_ is not None and max_dd_ != 0 and not np.isnan(max_dd_):
            calmar_ = cagr_ / abs(max_dd_)
        else:
            calmar_ = np.nan

        # Sortino Ratio: annualized return / annualized downside volatility
        downside = dr[dr < 0]
        if len(downside) < 2:
            sortino_ = np.nan
        else:
            downside_std_annual = downside.std() * np.sqrt(ann_factor)
            ann_mean = dr.mean() * ann_factor
            if downside_std_annual > 1e-9:
                sortino_ = ann_mean / downside_std_annual
            else:
                sortino_ = np.nan

        return {
            "CAGR": cagr_,
            "Vol": vol_,
            "Sharpe": sharpe_,
            "MaxDD": max_dd_,
            "Calmar": calmar_,
            "Sortino": sortino_
        }

    def plot(self, results, split=0.7, benchmark_label="Benchmark"):
        """
        Creates the figure with 3 rows x 2 columns.
        """
        strategy_returns = results.get("strategy_returns", pd.Series(dtype=float))
        if strategy_returns.empty:
            print("No strategy returns found in results.")
            return

        benchmark_returns = results.get("benchmark_returns", pd.Series(dtype=float))

        # Equity curves
        strat_equity = (1.0 + strategy_returns).cumprod()
        bench_equity = None
        if not benchmark_returns.empty:
            bench_equity = (1.0 + benchmark_returns).cumprod()

        # Split train/test by date
        all_dates = strategy_returns.index.unique().sort_values()
        n = len(all_dates)
        if n < 2:
            print("Not enough data to plot.")
            return

        split_idx = int(n * split)
        if split_idx < 1:
            print("Train set is empty. Increase 'split'.")
            return
        if split_idx >= n:
            print("Test set is empty. Decrease 'split'.")
            return

        train_dates = all_dates[:split_idx]
        test_dates = all_dates[split_idx:]
        split_date = train_dates[-1]

        # Split returns
        strat_train = strategy_returns.loc[train_dates]
        strat_test = strategy_returns.loc[test_dates]

        # Compute summary stats for train and test returns
        train_stats = self._compute_summary_stats(strat_train)
        test_stats = self._compute_summary_stats(strat_test)

        # Layout the figure
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(nrows=3, ncols=2, figure=fig, height_ratios=[2, 2, 2])

        # Row 0: Equity curves
        ax_eq = fig.add_subplot(gs[0, :])
        ax_eq.plot(strat_equity.index, strat_equity.values, label="Strategy")
        if bench_equity is not None:
            ax_eq.plot(bench_equity.index, bench_equity.values, label=benchmark_label, alpha=0.7)
        ax_eq.set_title("Equity Curve")
        ax_eq.legend(loc="best")
        ax_eq.axvline(x=split_date, color="gray", linestyle="--", alpha=0.8)

        # Row 1: Drawdowns
        ax_dd = fig.add_subplot(gs[1, :])
        strat_dd = self._compute_drawdown(strat_equity) * 100.0
        ax_dd.plot(strat_dd.index, strat_dd.values, label="Strategy DD (%)")
        if bench_equity is not None:
            bench_dd = self._compute_drawdown(bench_equity) * 100.0
            ax_dd.plot(bench_dd.index, bench_dd.values, label=f"{benchmark_label} DD (%)", alpha=0.7)
        ax_dd.set_title("Drawdown (%)")
        ax_dd.legend(loc="best")
        ax_dd.axvline(x=split_date, color="gray", linestyle="--", alpha=0.8)

        # Row 2, Left: Histogram of daily returns (train vs. test)
        ax_hist = fig.add_subplot(gs[2, 0])
        ax_hist.hist(strat_train, bins=30, alpha=0.5, label="Train")
        ax_hist.hist(strat_test, bins=30, alpha=0.5, label="Test")
        ax_hist.set_title("Train vs. Test Daily Returns")
        ax_hist.legend(loc="best")

        # Row 2, Right: Summary stats table
        ax_table = fig.add_subplot(gs[2, 1])
        ax_table.axis("off")
        ax_table.set_title("Train vs. Test Stats", pad=10)

        row_labels = ["CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "Sortino"]
        col_labels = ["Metric", "Train", "Test", "Diff", "Overfit Ratio"]
        cell_text = []
        diff_list = []
        ratio_list = []

        def fmt_val(metric, val):
            if pd.isna(val):
                return "NaN"
            if metric in ["CAGR", "Vol", "MaxDD", "Calmar"]:
                return f"{val:,.2%}"
            elif metric in ["Sharpe", "Sortino"]:
                return f"{val:,.2f}"
            else:
                return f"{val:.4f}"

        for metric in row_labels:
            train_val = train_stats.get(metric, np.nan)
            test_val = test_stats.get(metric, np.nan)

            # Compute diff: use test - train for all metrics except MaxDD.
            if pd.isna(train_val) or pd.isna(test_val):
                diff_val = np.nan
            else:
                if metric == "MaxDD":
                    # Use difference in absolute values so that a worse drawdown is positive.
                    diff_val = abs(test_val) - abs(train_val)
                else:
                    diff_val = test_val - train_val

            # Compute Overfit Ratio (always positive).
            if (
                pd.isna(train_val)
                or pd.isna(test_val)
                or abs(train_val) < 1e-12
                or abs(test_val) < 1e-12
            ):
                ratio_val = np.nan
            else:
                if metric == "MaxDD":
                    ratio_val = abs(test_val) / abs(train_val)
                else:
                    ratio_val = abs(train_val) / abs(test_val)

            train_str = fmt_val(metric, train_val)
            test_str = fmt_val(metric, test_val)
            if pd.isna(diff_val):
                diff_str = "NaN"
            else:
                if metric in ["CAGR", "Vol", "MaxDD", "Calmar"]:
                    diff_str = f"{diff_val:,.2%}"
                elif metric in ["Sharpe", "Sortino"]:
                    diff_str = f"{diff_val:,.2f}"
                else:
                    diff_str = f"{diff_val:.4f}"

            ratio_str = "NaN" if pd.isna(ratio_val) else f"{ratio_val:,.2f}"

            cell_text.append([metric, train_str, test_str, diff_str, ratio_str])
            diff_list.append((metric, diff_val))
            ratio_list.append((metric, ratio_val))

        tbl = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        # Bold header row.
        for col_idx in range(len(col_labels)):
            hdr_cell = tbl[(0, col_idx)]
            hdr_cell.get_text().set_weight("bold")

        # Bold first column.
        for row_idx in range(1, len(row_labels) + 1):
            metric_cell = tbl[(row_idx, 0)]
            metric_cell.get_text().set_weight("bold")

        # Color the Diff column (column 3).
        diff_col_idx = 3
        for i, (metric, diff_val) in enumerate(diff_list, start=1):
            if metric == "Vol" or pd.isna(diff_val):
                continue
            diff_cell = tbl.get_celld()[(i, diff_col_idx)]
            if metric == "MaxDD":
                # For MaxDD diff: positive diff (i.e. a larger absolute drawdown in test) should be red.
                if diff_val > 0:
                    diff_cell.set_facecolor("lightcoral")
                else:
                    diff_cell.set_facecolor("lightgreen")
            else:
                if diff_val > 0:
                    diff_cell.set_facecolor("lightgreen")
                else:
                    diff_cell.set_facecolor("lightcoral")

        # Color the Overfit Ratio column (column 4), skipping Vol.
        ratio_col_idx = 4
        for i, (metric, ratio_val) in enumerate(ratio_list, start=1):
            if metric == "Vol" or pd.isna(ratio_val):
                continue
            ratio_cell = tbl.get_celld()[(i, ratio_col_idx)]
            if ratio_val <= 1.1:
                ratio_cell.set_facecolor("lightgreen")
            elif ratio_val <= 1.25:
                ratio_cell.set_facecolor("lightyellow")
            else:
                ratio_cell.set_facecolor("lightcoral")

        plt.tight_layout()
        plt.show()

```

# portwine/backtester.py

```py
# portwine/backtester.py

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import pandas_market_calendars as mcal
from portwine.loaders.base import MarketDataLoader

class InvalidBenchmarkError(Exception):
    """Raised when the requested benchmark is neither a standard name nor a valid ticker."""
    pass

# ----------------------------------------------------------------------
# Built‑in benchmark helpers
# ----------------------------------------------------------------------
def benchmark_equal_weight(ret_df: pd.DataFrame, *_, **__) -> pd.Series:
    return ret_df.mean(axis=1)

def benchmark_markowitz(
    ret_df: pd.DataFrame,
    lookback: int = 60,
    shift_signals: bool = True,
    verbose: bool = False,
) -> pd.Series:
    tickers = ret_df.columns
    n = len(tickers)
    iterator = tqdm(ret_df.index, desc="Markowitz") if verbose else ret_df.index
    w_rows: List[np.ndarray] = []
    for ts in iterator:
        win = ret_df.loc[:ts].tail(lookback)
        if len(win) < 2:
            w = np.ones(n) / n
        else:
            cov = win.cov().values
            w_var = cp.Variable(n, nonneg=True)
            prob = cp.Problem(cp.Minimize(cp.quad_form(w_var, cov)), [cp.sum(w_var) == 1])
            try:
                prob.solve()
                w = w_var.value if w_var.value is not None else np.ones(n) / n
            except Exception:
                w = np.ones(n) / n
        w_rows.append(w)
    w_df = pd.DataFrame(w_rows, index=ret_df.index, columns=tickers)
    if shift_signals:
        w_df = w_df.shift(1).ffill().fillna(1.0 / n)
    return (w_df * ret_df).sum(axis=1)

STANDARD_BENCHMARKS: Dict[str, Callable] = {
    "equal_weight": benchmark_equal_weight,
    "markowitz":    benchmark_markowitz,
}

class BenchmarkTypes:
    STANDARD_BENCHMARK = 0
    TICKER             = 1
    CUSTOM_METHOD      = 2
    INVALID            = 3

# ------------------------------------------------------------------------------
# Backtester
# ------------------------------------------------------------------------------
class Backtester:
    """
    A step‑driven back‑tester that supports intraday bars and,
    optionally, an exchange trading calendar.
    """

    def __init__(
        self,
        market_data_loader: MarketDataLoader,
        alternative_data_loader=None,
        calendar: Optional[Union[str, mcal.ExchangeCalendar]] = None
    ):
        self.market_data_loader      = market_data_loader
        self.alternative_data_loader = alternative_data_loader
        if isinstance(calendar, str):
            self.calendar = mcal.get_calendar(calendar)
        else:
            self.calendar = calendar

    def _split_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        reg, alt = [], []
        for t in tickers:
            if isinstance(t, str) and ":" in t:
                alt.append(t)
            else:
                reg.append(t)
        return reg, alt

    def get_benchmark_type(self, benchmark) -> int:
        if isinstance(benchmark, str):
            if benchmark in STANDARD_BENCHMARKS:
                return BenchmarkTypes.STANDARD_BENCHMARK
            if self.market_data_loader.fetch_data([benchmark]).get(benchmark) is not None:
                return BenchmarkTypes.TICKER
            return BenchmarkTypes.INVALID
        if callable(benchmark):
            return BenchmarkTypes.CUSTOM_METHOD
        return BenchmarkTypes.INVALID

    def run_backtest(
        self,
        strategy,
        shift_signals: bool = True,
        benchmark: Union[str, Callable, None] = "equal_weight",
        start_date=None,
        end_date=None,
        require_all_history: bool = False,
        verbose: bool = False
    ) -> Optional[Dict[str, pd.DataFrame]]:
        # 1) normalize date filters
        sd = pd.Timestamp(start_date) if start_date is not None else None
        ed = pd.Timestamp(end_date)   if end_date   is not None else None
        if sd is not None and ed is not None and sd > ed:
            raise ValueError("start_date must be on or before end_date")

        # 2) split tickers
        reg_tkrs, alt_tkrs = self._split_tickers(strategy.tickers)

        # 3) classify benchmark
        bm_type = self.get_benchmark_type(benchmark)
        if bm_type == BenchmarkTypes.INVALID:
            raise InvalidBenchmarkError(f"{benchmark} is not a valid benchmark.")

        # 4) load regular data
        reg_data = self.market_data_loader.fetch_data(reg_tkrs)
        if not reg_tkrs or len(reg_data) < len(reg_tkrs):
            return None

        # 5) build trading dates
        if self.calendar is not None:
            # data span
            first_dt = min(df.index.min() for df in reg_data.values())
            last_dt  = max(df.index.max() for df in reg_data.values())

            # schedule uses dates only
            sched = self.calendar.schedule(
                start_date=first_dt.date(),
                end_date=last_dt.date()
            )
            closes = sched["market_close"]

            # drop tz if present
            if getattr(getattr(closes, "dt", None), "tz", None) is not None:
                closes = closes.dt.tz_convert(None)

            # restrict to actual data
            closes = closes[(closes >= first_dt) & (closes <= last_dt)]

            # require history
            if require_all_history and reg_tkrs:
                common = max(df.index.min() for df in reg_data.values())
                closes = closes[closes >= common]

            # apply start/end (full timestamp)
            if sd is not None:
                closes = closes[closes >= sd]
            if ed is not None:
                closes = closes[closes <= ed]

            all_ts = list(closes)

            # **raise** on empty calendar range
            if not all_ts:
                raise ValueError("No trading dates after filtering")

        else:
            # legacy union of data indices
            if hasattr(self.market_data_loader, "get_all_dates"):
                all_ts = self.market_data_loader.get_all_dates(reg_tkrs)
            else:
                all_ts = sorted({ts for df in reg_data.values() for ts in df.index})

            # require history
            if require_all_history and reg_tkrs:
                common = max(df.index.min() for df in reg_data.values())
                all_ts = [d for d in all_ts if d >= common]

            # apply start/end
            if sd is not None:
                all_ts = [d for d in all_ts if d >= sd]
            if ed is not None:
                all_ts = [d for d in all_ts if d <= ed]

            if not all_ts:
                raise ValueError("No trading dates after filtering")

        # 6) preload benchmark ticker if needed
        if bm_type == BenchmarkTypes.TICKER:
            bm_data = self.market_data_loader.fetch_data([benchmark])

        # 7) main loop: signals
        sig_rows = []
        iterator = tqdm(all_ts, desc="Backtest") if verbose else all_ts
        for ts in iterator:
            if hasattr(self.market_data_loader, "next"):
                bar = self.market_data_loader.next(reg_tkrs, ts)
            else:
                bar = self._bar_dict(ts, reg_data)

            if self.alternative_data_loader:
                alt_ld = self.alternative_data_loader
                if hasattr(alt_ld, "next"):
                    bar.update(alt_ld.next(alt_tkrs, ts))
                else:
                    for t, df in alt_ld.fetch_data(alt_tkrs).items():
                        bar[t] = self._bar_dict(ts, {t: df})[t]

            sig = strategy.step(ts, bar)
            row = {"date": ts}
            for t in strategy.tickers:
                row[t] = sig.get(t, 0.0)
            sig_rows.append(row)

        # 8) construct signals_df
        sig_df = pd.DataFrame(sig_rows).set_index("date").sort_index()
        sig_df.index.name = None
        sig_reg = ((sig_df.shift(1).ffill() if shift_signals else sig_df)
                   .fillna(0.0)[reg_tkrs])

        # 9) compute returns
        px     = pd.DataFrame({t: reg_data[t]["close"] for t in reg_tkrs})
        px     = px.reindex(sig_reg.index).ffill()
        ret_df = px.pct_change(fill_method=None).fillna(0.0)
        strat_ret = (ret_df * sig_reg).sum(axis=1)

        # 10) benchmark returns
        if bm_type == BenchmarkTypes.CUSTOM_METHOD:
            bm_ret = benchmark(ret_df)
        elif bm_type == BenchmarkTypes.STANDARD_BENCHMARK:
            bm_ret = STANDARD_BENCHMARKS[benchmark](ret_df)
        else:  # TICKER
            ser = bm_data[benchmark]["close"].reindex(sig_reg.index).ffill()
            bm_ret = ser.pct_change(fill_method=None).fillna(0.0)

        # 11) dynamic alternative data update
        if self.alternative_data_loader and hasattr(self.alternative_data_loader, "update"):
            for ts in sig_reg.index:
                raw_sigs = sig_df.loc[ts, strategy.tickers].to_dict()
                raw_rets = ret_df.loc[ts].to_dict()
                self.alternative_data_loader.update(ts, raw_sigs, raw_rets, float(strat_ret.loc[ts]))

        return {
            "signals_df":       sig_reg,
            "tickers_returns":  ret_df,
            "strategy_returns": strat_ret,
            "benchmark_returns": bm_ret
        }

    @staticmethod
    def _bar_dict(ts: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, dict | None]:
        out: Dict[str, dict | None] = {}
        for t, df in data.items():
            if ts in df.index:
                row = df.loc[ts]
                out[t] = {
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "volume": float(row["volume"]),
                }
            else:
                out[t] = None
        return out

```

# portwine/brokers/__init__.py

```py

```

# portwine/brokers/alpaca.py

```py
import requests
import time
from datetime import datetime
from typing import Dict, List

from portwine.brokers.base import (
    BrokerBase,
    Account,
    Position,
    Order,
    OrderExecutionError,
    OrderNotFoundError,
    OrderCancelError,
)


def _parse_datetime(dt_str: str) -> datetime:
    """
    Parse an ISO‑8601 timestamp from Alpaca, handling the trailing 'Z'.
    """
    if dt_str is None:
        return None
    # Alpaca returns times like "2021-04-14T09:30:00Z"
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)


class AlpacaBroker(BrokerBase):
    """
    Alpaca REST API implementation of BrokerBase using the `requests` library.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        """
        Args:
            api_key: Your Alpaca API key ID.
            api_secret: Your Alpaca secret key.
            base_url: Alpaca REST endpoint (paper or live).
        """
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type": "application/json",
        })

    def get_account(self) -> Account:
        url = f"{self._base_url}/v2/account"
        resp = self._session.get(url)
        if not resp.ok:
            raise OrderExecutionError(f"Account fetch failed: {resp.text}")
        data = resp.json()
        return Account(equity=float(data["equity"]), last_updated_at=int(time.time() * 1000))

    def get_positions(self) -> Dict[str, Position]:
        url = f"{self._base_url}/v2/positions"
        resp = self._session.get(url)
        if not resp.ok:
            raise OrderExecutionError(f"Positions fetch failed: {resp.text}")
        positions = {}
        for p in resp.json():
            positions[p["symbol"]] = Position(
                ticker=p["symbol"],
                quantity=float(p["qty"]),
                last_updated_at=int(time.time() * 1000)
            )
        return positions

    def get_position(self, ticker: str) -> Position:
        url = f"{self._base_url}/v2/positions/{ticker}"
        resp = self._session.get(url)
        if resp.status_code == 404:
            return Position(ticker=ticker, quantity=0.0, last_updated_at=int(time.time() * 1000))
        if not resp.ok:
            raise OrderExecutionError(f"Position fetch failed: {resp.text}")
        p = resp.json()
        return Position(ticker=p["symbol"], quantity=float(p["qty"]), last_updated_at=int(time.time() * 1000))

    def get_order(self, order_id: str) -> Order:
        url = f"{self._base_url}/v2/orders/{order_id}"
        resp = self._session.get(url)
        if resp.status_code == 404:
            raise OrderNotFoundError(f"Order {order_id} not found")
        if not resp.ok:
            raise OrderExecutionError(f"Order fetch failed: {resp.text}")
        o = resp.json()
        return Order(
            order_id=o["id"],
            ticker=o["symbol"],
            side=o["side"],
            quantity=float(o["qty"]),
            order_type=o["type"],
            status=o["status"],
            time_in_force=o["time_in_force"],
            average_price=float(o["filled_avg_price"] or 0.0),
            remaining_quantity=float(o["qty"]) - float(o["filled_qty"]),
            created_at=_parse_datetime(o["created_at"]),
            last_updated_at=_parse_datetime(o["updated_at"]),
        )

    def get_orders(self) -> List[Order]:
        url = f"{self._base_url}/v2/orders"
        resp = self._session.get(url)
        if not resp.ok:
            raise OrderExecutionError(f"Orders fetch failed: {resp.text}")
        orders = []
        for o in resp.json():
            orders.append(Order(
                order_id=o["id"],
                ticker=o["symbol"],
                side=o["side"],
                quantity=float(o["qty"]),
                order_type=o["type"],
                status=o["status"],
                time_in_force=o["time_in_force"],
                average_price=float(o["filled_avg_price"] or 0.0),
                remaining_quantity=float(o["qty"]) - float(o["filled_qty"]),
                created_at=_parse_datetime(o["created_at"]),
                last_updated_at=_parse_datetime(o["updated_at"]),
            ))
        return orders

    def submit_order(self, symbol: str, quantity: float) -> Order:
        """
        Execute a market order on Alpaca.
        Positive quantity → buy, negative → sell.
        """
        side = "buy" if quantity > 0 else "sell"
        qty = abs(quantity)
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        url = f"{self._base_url}/v2/orders"
        resp = self._session.post(url, json=payload)
        if not resp.ok:
            raise OrderExecutionError(f"Order submission failed: {resp.text}")
        o = resp.json()
        return Order(
            order_id=o["id"],
            ticker=o["symbol"],
            side=o["side"],
            quantity=float(o["qty"]),
            order_type=o["type"],
            status=o["status"],
            time_in_force=o["time_in_force"],
            average_price=float(o["filled_avg_price"] or 0.0),
            remaining_quantity=float(o["qty"]) - float(o["filled_qty"]),
            created_at=_parse_datetime(o["created_at"]),
            last_updated_at=_parse_datetime(o["updated_at"]),
        )

    def market_is_open(self, timestamp: datetime) -> bool:
        """
        Check if the market is open at the time of the request using the Alpaca clock endpoint.
        """
        url = f"{self._base_url}/v2/clock"
        resp = self._session.get(url)
        if not resp.ok:
            raise OrderExecutionError(f"Clock fetch failed: {resp.text}")
        data = resp.json()
        return bool(data.get("is_open", False))

```

# portwine/brokers/base.py

```py
"""
Broker base class for trading interfaces.

This module provides the base class for all broker implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


@dataclass
class Position:
    """Represents a trading position."""
    ticker: str
    quantity: float
    last_updated_at: int # UNIX timestamp in second for last time the data was updated

class TimeInForce(Enum):
    DAY = "day"
    GOOD_TILL_CANCELLED = "gtc"
    IMMEDIATE_OR_CANCEL = "ioc"
    FILL_OR_KILL = "fok"
    MARKET_ON_OPEN = "opg"
    MARKET_ON_CLOSE = "cls"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    
class OrderStatus(Enum):
    SUBMITTED = "submitted"
    REJECTED = "rejected"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    PENDING = "pending"

class Side(Enum):
    BUY = "buy"
    SELL = "sell"

'''
    Order dataclass. All brokers must adhere to this standard. 
'''
@dataclass
class Order:
    order_id: str               # unique identifier for the order
    ticker: str                 # asset ticker
    side: str                   # buy or sell
    quantity: float             # amount / size of order
    order_type: str             # market, limit, etc
    status: str                 # submitted, rejected, filled, etc
    time_in_force: str          # gtc, fok, etc
    average_price: float        # average fill price of order
    remaining_quantity: float   # how much of the order still needs to be filled
    created_at: int        # when the order was created (UNIX timestamp milliseconds)
    last_updated_at: int   # when the data on this order was last updated with the broker

@dataclass
class Account:
    equity: float         # amount of money available to purchase securities (can include margin)
    last_updated_at: int
    # net_liquidation_value     liquid value, total_equity... all are names for the same thing, equity...
    # buying power includes margin, equity does not


class OrderExecutionError(Exception):
    """Raised when an order fails to execute."""
    pass


class OrderNotFoundError(Exception):
    """Raised when an order cannot be found."""
    pass


class OrderCancelError(Exception):
    """Raised when an order fails to cancel."""
    pass


class BrokerBase(ABC):
    """Base class for all broker implementations."""

    @abstractmethod
    def get_account(self) -> Account:
        """
        Get the current account information.
        
        Returns:
            Account object containing current account state
        """
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.
        
        Returns:
            Dictionary mapping symbol to Position objects
        """
        pass

    @abstractmethod
    def get_position(self, ticker) -> Position:
        # Returns a position object for a given ticker
        # if there is no position, then an empty position is returned
        # with quantity 0
        pass

    @abstractmethod
    def get_order(self, order_id) -> Order:
        pass

    @abstractmethod
    def get_orders(self) -> List[Order]:
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        quantity: float,
    ) -> Order:
        """
        Execute a market order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            quantity: Order quantity
        
        Returns:
            Order object representing the executed order
        
        Raises:
            ValueError: If the order parameters are invalid
            OrderExecutionError: If the order fails to execute
        """
        pass

    @abstractmethod
    def market_is_open(self, timestamp: datetime) -> bool:
        """
        Check if the market is open at the given datetime.
        """
        pass

```

# portwine/brokers/mock.py

```py
import threading
import time
from datetime import datetime
from typing import Dict, List

from portwine.brokers.base import (
    BrokerBase,
    Account,
    Position,
    Order,
    OrderNotFoundError,
)


class MockBroker(BrokerBase):
    """
    A simple in‑memory mock broker for testing.
    Fills all market orders immediately at a fixed price and
    tracks positions and orders in dictionaries.
    """

    def __init__(self, initial_equity: float = 1_000_000.0, fill_price: float = 100.0):
        self._equity = initial_equity
        self._fill_price = fill_price
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        self._lock = threading.Lock()

    def get_account(self) -> Account:
        """
        Return a snapshot of the account with a unix‐timestamp last_updated_at (ms).
        """
        ts = int(time.time() * 1_000)
        return Account(
            equity=self._equity,
            last_updated_at=ts
        )

    def get_positions(self) -> Dict[str, Position]:
        """
        Return all current positions, preserving each position's last_updated_at.
        """
        return {
            symbol: Position(
                ticker=pos.ticker,
                quantity=pos.quantity,
                last_updated_at=pos.last_updated_at
            )
            for symbol, pos in self._positions.items()
        }

    def get_position(self, ticker: str) -> Position:
        """
        Return position for a single ticker (zero if not held),
        with last_updated_at from the stored position or now if none.
        """
        pos = self._positions.get(ticker)
        if pos is None:
            ts = int(time.time() * 1_000)
            return Position(ticker=ticker, quantity=0.0, last_updated_at=ts)
        return Position(
            ticker=pos.ticker,
            quantity=pos.quantity,
            last_updated_at=pos.last_updated_at
        )

    def get_order(self, order_id: str) -> Order:
        """
        Retrieve a single order by ID; raise if not found.
        """
        try:
            return self._orders[order_id]
        except KeyError:
            raise OrderNotFoundError(f"Order {order_id} not found")

    def get_orders(self) -> List[Order]:
        """
        Return a list of all orders submitted so far.
        """
        return list(self._orders.values())

    def submit_order(self, symbol: str, quantity: float) -> Order:
        """
        Simulate a market order fill:
          - Immediately 'fills' at self._fill_price
          - Updates in‑memory positions with a unix‐timestamp last_updated_at (ms)
          - Records the order with status 'filled' and last_updated_at as unix timestamp (ms)
        """
        with self._lock:
            self._order_counter += 1
            oid = str(self._order_counter)

        side = "buy" if quantity > 0 else "sell"
        qty = abs(quantity)
        now_ts = int(time.time() * 1_000)

        # Update position
        prev = self._positions.get(symbol)
        prev_qty = prev.quantity if prev is not None else 0.0
        new_qty = prev_qty + quantity

        if new_qty == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = Position(
                ticker=symbol,
                quantity=new_qty,
                last_updated_at=now_ts
            )

        order = Order(
            order_id=oid,
            ticker=symbol,
            side=side,
            quantity=qty,
            order_type="market",
            status="filled",
            time_in_force="day",
            average_price=self._fill_price,
            remaining_quantity=0.0,
            created_at=now_ts,
            last_updated_at=now_ts
        )

        self._orders[oid] = order
        return order

    def market_is_open(self, timestamp: datetime) -> bool:
        """
        Stub implementation of market hours:
        - Returns True if it's a weekday (Monday=0 … Friday=4)
          between 09:30 and 16:00 in the local system time.
        """
        # Weekday check
        if timestamp.weekday() >= 5:
            return False

        # Hour/minute check: between 9:30 and 16:00
        hm = timestamp.hour * 60 + timestamp.minute
        open_time = 9 * 60 + 30   # 9:30 = 570 minutes
        close_time = 16 * 60      # 16:00 = 960 minutes
        return open_time <= hm < close_time

```

# portwine/execution.py

```py
"""
Execution module for the portwine framework.

This module provides the base classes and interfaces for execution modules,
which connect strategy implementations from the backtester to live trading.
"""

from __future__ import annotations

import abc
import logging
from typing import Dict, List, Optional, Tuple, Iterator
import math
import time
from datetime import datetime

from portwine.loaders.base import MarketDataLoader
from portwine.strategies.base import StrategyBase
from portwine.brokers.base import BrokerBase, Order

# Configure logging
logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Base exception for execution-related errors."""
    pass


class OrderExecutionError(ExecutionError):
    """Exception raised when order execution fails."""
    pass


class DataFetchError(ExecutionError):
    """Exception raised when data fetching fails."""
    pass


class PortfolioExceededError(ExecutionError):
    """Raised when current portfolio weights exceed 100% of portfolio value."""
    pass


class ExecutionBase(abc.ABC):
    """
    Base class for execution implementations.
    
    An execution implementation is responsible for:
    1. Fetching latest market data
    2. Passing data to strategy to get updated weights
    3. Calculating position changes needed
    4. Executing necessary trades using a broker
    """
    
    def __init__(
        self,
        strategy: StrategyBase,
        market_data_loader: MarketDataLoader,
        broker: BrokerBase,
        alternative_data_loader: Optional[MarketDataLoader] = None,
        min_change_pct: float = 0.01,
        min_order_value: float = 1.0,
        timezone: Optional[datetime.tzinfo] = None,
    ):
        """
        Initialize the execution instance.
        
        Parameters
        ----------
        strategy : StrategyBase
            The strategy implementation to use for generating trading signals
        market_data_loader : MarketDataLoader
            Market data loader for price data
        broker : BrokerBase
            Broker implementation for executing trades
        alternative_data_loader : Optional[MarketDataLoader]
            Additional data loader for alternative data
        min_change_pct : float, default 0.01
            Minimum change percentage required to trigger a trade
        min_order_value : float, default 1.0
            Minimum dollar value required for an order
        timezone : Optional[datetime.tzinfo], default None
            Timezone for timestamp conversion
        """
        self.strategy = strategy
        self.market_data_loader = market_data_loader
        self.broker = broker
        self.alternative_data_loader = alternative_data_loader
        self.min_change_pct = min_change_pct
        self.min_order_value = min_order_value
        
        # Store timezone (tzinfo); default to system local timezone
        self.timezone = timezone if timezone is not None else datetime.now().astimezone().tzinfo
        # Initialize ticker list from strategy
        self.tickers = strategy.tickers
        
        logger.info(f"Initialized {self.__class__.__name__} with {len(self.tickers)} tickers")
    
    @staticmethod
    def _split_tickers(tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Split full ticker list into regular and alternative tickers.
        Regular tickers have no ':'; alternative contain ':'
        """
        reg: List[str] = []
        alt: List[str] = []
        for t in tickers:
            if isinstance(t, str) and ":" in t:
                alt.append(t)
            else:
                reg.append(t)
        return reg, alt

    def fetch_latest_data(self, timestamp: Optional[float] = None) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Fetch latest market data for the tickers in the strategy.
        
        Parameters
        ----------
        timestamp : Optional[float]
            UNIX timestamp to get data for, or current time if None
            
        Returns
        -------
        Dict[str, Optional[Dict[str, float]]]
            Dictionary of latest bar data for each ticker
        
        Raises
        ------
        DataFetchError
            If data cannot be fetched
        """
        try:
            # Convert UNIX timestamp to timezone-aware datetime, default to now
            if timestamp is None:
                dt = datetime.now(tz=self.timezone)
            else:
                # timestamp is seconds since epoch
                dt = datetime.fromtimestamp(timestamp, tz=self.timezone)
            # Strip tzinfo for loader to match tz-naive indices
            loader_dt = dt.replace(tzinfo=None)
            # Split tickers into market vs alternative
            reg_tkrs, alt_tkrs = self._split_tickers(self.tickers)
            # Fetch market data only for regular tickers
            data = self.market_data_loader.next(reg_tkrs, loader_dt)
            # Fetch alternative data only for alternative tickers
            if self.alternative_data_loader is not None and alt_tkrs:
                alt_data = self.alternative_data_loader.next(alt_tkrs, loader_dt)
                # Merge alternative entries into result
                data.update(alt_data)
            
            return data
        except Exception as e:
            logger.exception(f"Error fetching latest data: {e}")
            raise DataFetchError(f"Failed to fetch latest data: {e}")
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current closing prices for the specified symbols by querying only market data.

        This method bypasses alternative data and directly uses market_data_loader.next
        with a timezone-naive datetime matching the execution timezone.
        """
        # Build current datetime in execution timezone
        dt = datetime.now(tz=self.timezone)
        # Align to loader timezone (no-op if same) and strip tzinfo
        loader_dt = dt.astimezone(self.timezone).replace(tzinfo=None)
        # Fetch only market data for given symbols
        data = self.market_data_loader.next(symbols, loader_dt)
        prices: Dict[str, float] = {}
        for symbol, bar in data.items():
            if bar is None:
                continue
            price = bar.get('close')
            if price is not None:
                prices[symbol] = price
        return prices
    
    def _get_current_positions(self) -> Tuple[Dict[str, float], float]:
        """
        Get current positions from broker account info.
        
        Returns
        -------
        Tuple[Dict[str, float], float]
            Current position quantities for each ticker and the portfolio value
        """
        positions = self.broker.get_positions()
        account = self.broker.get_account()
        
        current_positions = {symbol: position.quantity for symbol, position in positions.items()}
        portfolio_value = account.equity
        
        return current_positions, portfolio_value

    def _calculate_target_positions(
        self,
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        fractional: bool = True,
    ) -> Dict[str, float]:
        """
        Convert target weights to absolute position sizes.
        
        Optionally prevent fractional shares by rounding down when `fractional=False`.
        
        Parameters
        ----------
        target_weights : Dict[str, float]
            Target allocation weights for each ticker
        portfolio_value : float
            Current portfolio value
        prices : Dict[str, float]
            Current prices for each ticker
        fractional : bool, default True
            If False, positions are floored to the nearest integer
        
        Returns
        -------
        Dict[str, float]
            Target position quantities for each ticker
        """
        target_positions = {}
        for symbol, weight in target_weights.items():
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            target_value = weight * portfolio_value
            raw_qty = target_value / price
            if fractional:
                qty = raw_qty
            else:
                qty = math.floor(raw_qty)
            target_positions[symbol] = qty
        return target_positions

    def _calculate_current_weights(
        self,
        positions: List[Tuple[str, float]],
        portfolio_value: float,
        prices: Dict[str, float],
        raises: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate current weights of positions based on prices and portfolio value.

        Args:
            positions: List of (ticker, quantity) tuples.
            portfolio_value: Total portfolio value.
            prices: Mapping of ticker to current price.
            raises: If True, raise PortfolioExceededError when total weights > 1.

        Returns:
            Dict[ticker, weight] mapping.

        Raises:
            PortfolioExceededError: If raises=True and sum(weights) > 1.
        """
        # Map positions
        pos_map: Dict[str, float] = {t: q for t, q in positions}
        weights: Dict[str, float] = {}
        total: float = 0.0
        for ticker, price in prices.items():
            qty = pos_map.get(ticker, 0.0)
            w = (price * qty) / portfolio_value if portfolio_value else 0.0
            weights[ticker] = w
            total += w
        if raises and total > 1.0:
            raise PortfolioExceededError(
                f"Total weights {total:.2f} exceed 1.0"
            )
        return weights

    def _target_positions_to_orders(
        self,
        target_positions: Dict[str, float],
        current_positions: Dict[str, float],
    ) -> List[Order]:
        """
        Determine necessary orders given target and current positions, returning Order objects.

        Args:
            target_positions: Mapping ticker -> desired quantity
            current_positions: Mapping ticker -> existing quantity

        Returns:
            List of Order dataclasses for each non-zero change.
        """
        orders: List[Order] = []
        for ticker, target_qty in target_positions.items():
            current_qty = current_positions.get(ticker, 0.0)
            diff = target_qty - current_qty
            if diff == 0:
                continue
            side = 'buy' if diff > 0 else 'sell'
            qty = abs(int(diff))
            # Build an Order dataclass with default/trivial values for non-relevant fields
            order = Order(
                order_id="",
                ticker=ticker,
                side=side,
                quantity=float(qty),
                order_type="market",
                status="new",
                time_in_force="day",
                average_price=0.0,
                remaining_quantity=0.0,
                created_at=0,
                last_updated_at=0,
            )
            orders.append(order)
        return orders

    def _execute_orders(self, orders: List[Order]) -> List[Order]:
        """
        Execute a list of Order objects through the broker.

        Parameters
        ----------
        orders : List[Order]
            List of Order dataclasses to submit.

        Returns
        -------
        List[Order]
            List of updated Order objects returned by the broker.
        """
        executed_orders: List[Order] = []
        for order in orders:
            # Determine signed quantity: negative for sell, positive for buy
            qty_arg = order.quantity if order.side == 'buy' else -order.quantity
            # Submit each order and collect the updated result
            updated = self.broker.submit_order(
                symbol=order.ticker,
                quantity=qty_arg,
            )
            # Restore expected positive quantity and original side
            updated.quantity = order.quantity
            updated.side = order.side
            executed_orders.append(updated)
        return executed_orders
    
    def step(self, timestamp_ms: Optional[int] = None) -> List[Order]:
        """
        Execute a single step of the trading strategy.

        Uses a UNIX timestamp in milliseconds; if None, uses current time.

        Returns a list of updated Order objects.
        """

        # Determine timestamp in ms
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        # Convert ms to datetime in execution timezone
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=self.timezone)

        # Check if market is open
        if not self.broker.market_is_open(dt):
            return []

        # Fetch latest market data
        latest_data = self.fetch_latest_data(dt.timestamp())
        # Get target weights from strategy
        target_weights = self.strategy.step(dt, latest_data)
        # Get current positions and portfolio value
        current_positions, portfolio_value = self._get_current_positions()
        # Extract prices from fetched data
        prices = {
            symbol: bar['close']
            for symbol, bar in latest_data.items() if bar and 'close' in bar
        }
        # Compute target positions and optionally current weights
        target_positions = self._calculate_target_positions(
            target_weights, portfolio_value, prices
        )
        _ = self._calculate_current_weights(
            list(current_positions.items()), portfolio_value, prices
        )
        # Determine orders and execute them
        orders = self._target_positions_to_orders(target_positions, current_positions)
        return self._execute_orders(orders)

    def run(self, schedule: Iterator[int]) -> None:
        """
        Continuously execute `step` at each timestamp provided by the schedule iterator,
        waiting until the scheduled time before running.

        Args:
            schedule: An iterator yielding UNIX timestamps in milliseconds for when to run each step.

        The loop terminates when the iterator is exhausted (StopIteration).
        """
        for timestamp_ms in schedule:
            # compute time until next scheduled timestamp
            now_ms = int(time.time() * 1000)
            wait_ms = timestamp_ms - now_ms
            if wait_ms > 0:
                time.sleep(wait_ms / 1000)
            # execute step at or after scheduled time
            self.step(timestamp_ms)

```

# portwine/loaders/__init__.py

```py
from portwine.loaders.base import MarketDataLoader
from portwine.loaders.eodhd import EODHDMarketDataLoader
from portwine.loaders.polygon import PolygonMarketDataLoader
from portwine.loaders.noisy import NoisyMarketDataLoader
from portwine.loaders.fred import FREDMarketDataLoader
from portwine.loaders.barchartindices import BarchartIndicesMarketDataLoader
from portwine.loaders.alternative import AlternativeMarketDataLoader
from portwine.loaders.dailytoopenclose import DailyToOpenCloseLoader
from portwine.loaders.alpaca import AlpacaMarketDataLoader

```

# portwine/loaders/alpaca.py

```py
"""
Alpaca market data loader for the portwine framework.

This module provides a MarketDataLoader implementation for fetching data
from the Alpaca Markets API, both for historical and real-time data.
Uses direct REST API calls instead of the Alpaca Python SDK.
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import requests

import pandas as pd
import pytz

from portwine.loaders.base import MarketDataLoader

# Configure logging
logger = logging.getLogger(__name__)

# API URLs
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


class AlpacaMarketDataLoader(MarketDataLoader):
    """
    Market data loader for Alpaca Markets API.
    
    This loader fetches historical and real-time data from Alpaca Markets API
    using direct REST calls. It supports fetching OHLCV data for stocks and ETFs.
    
    Parameters
    ----------
    api_key : str, optional
        Alpaca API key. If not provided, attempts to read from ALPACA_API_KEY env var.
    api_secret : str, optional
        Alpaca API secret. If not provided, attempts to read from ALPACA_API_SECRET env var.
    start_date : Union[str, datetime], optional
        Start date for historical data, defaults to 2 years ago
    end_date : Union[str, datetime], optional
        End date for historical data, defaults to today
    cache_dir : str, optional
        Directory to cache data to. If not provided, data is not cached.
    paper_trading : bool, default True
        Whether to use paper trading mode (sandbox)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        cache_dir: Optional[str] = None,
        paper_trading: bool = True,
    ):
        """Initialize Alpaca market data loader."""
        super().__init__()
        
        # Use environment variables if not provided
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials not provided. "
                "Either pass as parameters or set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
            )
        
        # Set up API URLs based on paper trading flag
        self.base_url = ALPACA_PAPER_URL if paper_trading else ALPACA_LIVE_URL
        self.data_url = ALPACA_DATA_URL
        
        # Auth headers for API requests
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }
        
        # Create requests session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Cache directory
        self.cache_dir = cache_dir
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Date range for historical data
        today = datetime.now(pytz.UTC)
        if start_date is None:
            # Default to 2 years ago
            self.start_date = today - timedelta(days=365 * 2)
        else:
            self.start_date = pd.Timestamp(start_date).to_pydatetime()
            if self.start_date.tzinfo is None:
                self.start_date = pytz.UTC.localize(self.start_date)
        
        if end_date is None:
            self.end_date = today
        else:
            self.end_date = pd.Timestamp(end_date).to_pydatetime()
            if self.end_date.tzinfo is None:
                self.end_date = pytz.UTC.localize(self.end_date)
        
        # Latest data cache to avoid frequent API calls
        self._latest_data_cache: Dict[str, Dict] = {}
        self._latest_data_timestamp = None
    
    def _api_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Helper method to make authenticated GET requests to Alpaca API
        
        Parameters
        ----------
        url : str
            API endpoint URL (starting with /)
        params : Dict[str, Any], optional
            Query parameters for the request
            
        Returns
        -------
        Any
            JSON response data
            
        Raises
        ------
        Exception
            If API request fails
        """
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if response is not None:
                logger.error(f"Response: {response.text}")
            raise
    
    def _get_cache_path(self, ticker: str) -> str:
        """
        Get path to cached data for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        str
            Path to cached data
        """
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{ticker}.parquet")
    
    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame if data is cached, None otherwise
        """
        if not self.cache_dir:
            return None
        
        cache_path = self._get_cache_path(ticker)
        if os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Error loading cached data for {ticker}: {e}")
        
        return None
    
    def _save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Save data to cache.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
        df : pd.DataFrame
            Data to cache
        """
        if not self.cache_dir:
            return
        
        cache_path = self._get_cache_path(ticker)
        try:
            df.to_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Error caching data for {ticker}: {e}")
    
    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for API requests"""
        return dt.isoformat()
    
    def _fetch_historical_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Alpaca API.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data or None if fetch fails
        """
        try:
            # Format parameters for API request
            params = {
                "symbols": ticker,
                "timeframe": "1Day",
                "start": self._format_datetime(self.start_date),
                "end": self._format_datetime(self.end_date),
                "limit": 10000,  # Maximum allowed by the API
                "adjustment": "raw"
            }
            
            # Make API request for bars
            url = f"{self.data_url}/v2/stocks/bars"
            response = self._api_get(url, params)
            
            # Extract data
            if 'bars' in response and ticker in response['bars']:
                bars = response['bars'][ticker]
                
                # Convert to dataframe
                df = pd.DataFrame(bars)
                
                # Convert timestamp string to datetime index
                df['t'] = pd.to_datetime(df['t'])
                df = df.set_index('t')
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                })
                
                # Drop timezone for compatibility
                df.index = df.index.tz_localize(None)
                
                return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
        
        return None
    
    def _fetch_latest_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch latest data for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        dict or None
            Latest OHLCV data or None if fetch fails
        """
        # Check if we have recent data in memory
        now = datetime.now()
        if (
            self._latest_data_timestamp 
            and (now - self._latest_data_timestamp).total_seconds() < 60
            and ticker in self._latest_data_cache
        ):
            return self._latest_data_cache[ticker]
        
        try:
            # Format parameters for API request
            params = {
                "symbols": ticker
            }
            
            # Make API request for latest bar
            url = f"{self.data_url}/v2/stocks/bars/latest"
            response = self._api_get(url, params)
            
            if 'bars' in response and ticker in response['bars']:
                bar = response['bars'][ticker]
                
                # Update cache
                self._latest_data_cache[ticker] = {
                    "open": float(bar['o']),
                    "high": float(bar['h']),
                    "low": float(bar['l']),
                    "close": float(bar['c']),
                    "volume": float(bar['v']),
                }
                self._latest_data_timestamp = now
                
                return self._latest_data_cache[ticker]
        
        except Exception as e:
            logger.error(f"Error fetching latest data for {ticker}: {e}")
        
        return None
    
    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data or None if load fails
        """
        # Check cache first
        df = self._load_from_cache(ticker)
        
        # Fetch from API if not cached or outdated
        if df is None or df.index.max() < self.end_date.replace(tzinfo=None):
            df_new = self._fetch_historical_data(ticker)
            
            if df_new is not None and not df_new.empty:
                if df is not None:
                    # Append new data
                    df = pd.concat([df[~df.index.isin(df_new.index)], df_new])
                    df = df.sort_index()
                else:
                    df = df_new
                
                # Save to cache
                self._save_to_cache(ticker, df)
        
        return df
    
    def next(self, tickers: List[str], timestamp: pd.Timestamp) -> Dict[str, Dict]:
        """
        Get data for tickers at or immediately before timestamp.
        
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols
        timestamp : pd.Timestamp
            Timestamp to get data for
            
        Returns
        -------
        Dict[str, dict]
            Dictionary mapping tickers to bar data
        """
        result = {}
        
        # If timestamp is close to now, get live data
        now = pd.Timestamp.now()
        if abs((now - timestamp).total_seconds()) < 86400:  # Within 24 hours
            for ticker in tickers:
                bar_data = self._fetch_latest_data(ticker)
                if bar_data:
                    result[ticker] = bar_data
                else:
                    result[ticker] = None
        else:
            # Otherwise use historical data
            for ticker in tickers:
                df = self.fetch_data([ticker]).get(ticker)
                if df is not None:
                    bar = self._get_bar_at_or_before(df, timestamp)
                    if bar is not None:
                        result[ticker] = {
                            "open": float(bar["open"]),
                            "high": float(bar["high"]),
                            "low": float(bar["low"]),
                            "close": float(bar["close"]),
                            "volume": float(bar["volume"]),
                        }
                    else:
                        result[ticker] = None
                else:
                    result[ticker] = None
        
        return result
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close() 
```

# portwine/loaders/alternative.py

```py
from portwine.loaders.base import MarketDataLoader


class AlternativeMarketDataLoader(MarketDataLoader):
    """
    A unified market data loader that routes requests to specialized data loaders
    based on source identifiers in ticker symbols.

    This loader expects tickers in the format "SOURCE:TICKER" where SOURCE matches
    a source identifier from one of the provided data loaders.

    Examples:
    - "FRED:SP500" would fetch SP500 data from the FRED data loader
    - "BARCHARTINDEX:ADDA" would fetch ADDA index from the Barchart Indices loader

    Parameters
    ----------
    loaders : list
        List of MarketDataLoader instances with SOURCE_IDENTIFIER class attributes
    """

    def __init__(self, loaders):
        """
        Initialize the alternative market data loader with a list of specialized loaders.
        """
        super().__init__()

        # Create a dictionary mapping source identifiers to loaders
        self.source_loaders = {}
        for loader in loaders:
            if hasattr(loader, 'SOURCE_IDENTIFIER'):
                self.source_loaders[loader.SOURCE_IDENTIFIER] = loader
            else:
                print(f"Warning: Loader {loader.__class__.__name__} does not have a SOURCE_IDENTIFIER")

    def load_ticker(self, ticker):
        """
        Load data for a specific ticker by routing to the appropriate data loader.

        Parameters
        ----------
        ticker : str
            Ticker in format "SOURCE:TICKER" (e.g., "FRED:SP500")

        Returns
        -------
        pd.DataFrame or None
            DataFrame with daily date index and appropriate columns, or None if load fails
        """
        # Check if it's already in the cache
        if ticker in self._data_cache:
            return self._data_cache[ticker].copy()  # Important: return a copy from cache

        # Parse the ticker to extract source and actual ticker
        if ":" not in ticker:
            print(f"Error: Invalid ticker format for {ticker}. Expected format is 'SOURCE:TICKER'.")
            return None

        source, actual_ticker = ticker.split(":", 1)

        # Find the appropriate loader
        if source in self.source_loaders:
            loader = self.source_loaders[source]
            data = loader.load_ticker(actual_ticker)

            # Cache the result if successful
            if data is not None:
                self._data_cache[ticker] = data.copy()  # Important: store a copy in cache

            return data
        else:
            print(f"Error: No loader found for source identifier '{source}'.")
            print(f"Available sources: {list(self.source_loaders.keys())}")
            return None

    def fetch_data(self, tickers):
        """
        Ensures we have data loaded for each ticker in 'tickers'.
        Routes each ticker to the appropriate specialized data loader.

        Parameters
        ----------
        tickers : list
            Tickers to ensure data is loaded for, in format "SOURCE:TICKER"

        Returns
        -------
        data_dict : dict
            { ticker: DataFrame }
        """
        fetched_data = {}

        for ticker in tickers:
            data = self.load_ticker(ticker)

            # Add to the returned dictionary if load was successful
            if data is not None:
                fetched_data[ticker] = data

        return fetched_data

    def get_available_sources(self):
        """
        Returns a list of available source identifiers.

        Returns
        -------
        list
            List of source identifiers
        """
        return list(self.source_loaders.keys())
```

# portwine/loaders/barchartindices.py

```py
import os
import pandas as pd
from portwine.loaders.base import MarketDataLoader


class BarchartIndicesMarketDataLoader(MarketDataLoader):
    """
    Loads data from local CSV files containing Barchart Indices data.

    This loader is designed to handle CSV files with index data following the format
    of Barchart indices. It reads from a local directory and does not download from
    any online source.

    Parameters
    ----------
    data_path : str
        Directory where index CSV files are stored
    """

    # Source identifier for AlternativeMarketDataLoader
    SOURCE_IDENTIFIER = 'BARCHARTINDEX'

    def __init__(self, data_path):
        """
        Initialize the Barchart Indices market data loader.
        """
        super().__init__()
        self.data_path = data_path

        # Create the data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def load_ticker(self, ticker):
        """
        Load data for a specific Barchart index from a CSV file.

        Parameters
        ----------
        ticker : str
            Barchart index code (e.g., 'ADDA')

        Returns
        -------
        pd.DataFrame
            DataFrame with daily date index and appropriate columns for the portwine framework
        """
        file_path = os.path.join(self.data_path, f"{ticker}.csv")

        if not os.path.isfile(file_path):
            print(f"Warning: CSV file not found for Barchart index {ticker}: {file_path}")
            return None

        try:
            # Load CSV file with automatic date parsing
            df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
            return self._format_dataframe(df)
        except Exception as e:
            print(f"Error loading data for Barchart index {ticker}: {str(e)}")
            return None

    def _format_dataframe(self, df):
        """
        Format the dataframe to match the expected structure for portwine.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to format

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame with appropriate columns
        """
        # Make sure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Make sure the index name is 'date'
        df.index.name = 'date'

        # Sort by date
        df = df.sort_index()

        # Ensure we have all required columns for portwine
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Check if we have a close column, if not look for other possible names
        if 'close' not in df.columns:
            for alt_name in ['Close', 'CLOSE', 'price', 'Price', 'value', 'Value']:
                if alt_name in df.columns:
                    df['close'] = df[alt_name]
                    break
            # If still not found, use the first numeric column
            if 'close' not in df.columns and not df.empty:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    df['close'] = df[numeric_cols[0]]

        # Fill missing required columns
        for col in required_columns:
            if col not in df.columns:
                if col in ['open', 'high', 'low'] and 'close' in df.columns:
                    df[col] = df['close']
                elif col == 'volume':
                    df[col] = 0
                else:
                    # If we can't determine a suitable close value, use NaN
                    df[col] = float('nan')

        return df
```

# portwine/loaders/base.py

```py
import pandas as pd

class MarketDataLoader:
    """
    Base loader. Override load_ticker; fetch_data remains unchanged.
    Adds:
      - get_all_dates: union calendar for any tickers
      - next: returns the bar at or immediately before a given ts via searchsorted
    """

    def __init__(self):
        self._data_cache = {}

    def load_ticker(self, ticker: str) -> pd.DataFrame | None:
        """
        Must be overridden to load and return a DataFrame indexed by pd.Timestamp
        with columns ['open','high','low','close','volume'], or return None.
        """
        raise NotImplementedError

    def fetch_data(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """
        Exactly as before: caches & returns all requested tickers.
        """
        fetched = {}
        for t in tickers:
            if t not in self._data_cache:
                df = self.load_ticker(t)
                if df is not None:
                    self._data_cache[t] = df
            if t in self._data_cache:
                fetched[t] = self._data_cache[t]
        return fetched

    def get_all_dates(self, tickers: list[str]) -> list[pd.Timestamp]:
        """
        Build the *union* of all timestamps across these tickers.
        This is your intraday/daily trading calendar.
        """
        data = self.fetch_data(tickers)
        all_ts = {ts for df in data.values() for ts in df.index}
        return sorted(all_ts)

    def _get_bar_at_or_before(self, df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
        """
        Find the row whose index is <= ts, using searchsorted.
        Returns the row (a pd.Series) or None if ts is before the first index.
        """
        idx = df.index
        pos = idx.searchsorted(ts, side="right") - 1
        if pos >= 0:
            return df.iloc[pos]
        return None

    def next(self,
             tickers: list[str],
             ts: pd.Timestamp
    ) -> dict[str, dict[str, float] | None]:
        """
        For a given timestamp ts, return a dict:
          { ticker: {'open','high','low','close','volume'} }
        where the values come from the bar at or immediately before ts.
        """
        data = self.fetch_data(tickers)
        bar_dict: dict[str, dict[str, float] | None] = {}

        for t, df in data.items():
            row = self._get_bar_at_or_before(df, ts)
            if row is None:
                bar_dict[t] = None
            else:
                bar_dict[t] = {
                    'open':   float(row['open']),
                    'high':   float(row['high']),
                    'low':    float(row['low']),
                    'close':  float(row['close']),
                    'volume': float(row['volume'])
                }

        return bar_dict

```

# portwine/loaders/dailytoopenclose.py

```py
import pandas as pd
import numpy as np
from portwine.loaders.base import MarketDataLoader


class DailyToOpenCloseLoader(MarketDataLoader):
    """
    Wraps a daily OHLCV loader and emits two intraday bars per day:
      - 09:30 bar: OHLC all set to the daily open, volume = 0
      - 16:00 bar: OHLC all set to the daily close, volume = daily volume
    """
    def __init__(self, base_loader: MarketDataLoader):
        self.base_loader = base_loader
        super().__init__()

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        # 1) Load the daily OHLCV from the base loader
        df_daily = self.base_loader.load_ticker(ticker)
        if df_daily is None or df_daily.empty:
            return df_daily

        # 2) Ensure the index is datetime
        df_daily = df_daily.copy()
        df_daily.index = pd.to_datetime(df_daily.index)

        # 3) Build intraday records
        records = []
        for date, row in zip(df_daily.index, df_daily.itertuples()):
            # 09:30 bar using the daily open
            ts_open = date.replace(hour=9, minute=30)
            records.append({
                'timestamp': ts_open,
                'open':  row.open,
                'high':  row.open,
                'low':   row.open,
                'close': row.open,
                'volume': 0
            })
            # 16:00 bar using the daily close
            ts_close = date.replace(hour=16, minute=0)
            records.append({
                'timestamp': ts_close,
                'open':  row.close,
                'high':  row.close,
                'low':   row.close,
                'close': row.close,
                'volume': getattr(row, 'volume', np.nan)
            })

        # 4) Assemble into a DataFrame with a DatetimeIndex
        df_intraday = (
            pd.DataFrame.from_records(records)
              .set_index('timestamp')
              .sort_index()
        )

        return df_intraday

```

# portwine/loaders/eodhd.py

```py
import os
import pandas as pd
from portwine.loaders.base import MarketDataLoader


class EODHDMarketDataLoader(MarketDataLoader):
    """
    Loads historical market data for a list of tickers from CSV files.
    Each CSV must be located in data_path and named as TICKER.US.csv for each ticker.
    The CSV is assumed to have at least these columns:
        date, open, high, low, close, adjusted_close, volume
    The loaded data will be stored in a dictionary keyed by ticker symbol.

    Once loaded, data is cached in memory. Subsequent calls for the same ticker
    will not re-read from disk.
    """

    def __init__(self, data_path, exchange_code='US'):
        """
        Parameters
        ----------
        data_path : str
            The directory path where CSV files are located.
        """
        self.data_path = data_path
        self.exchange_code = exchange_code
        super().__init__()

    def load_ticker(self, ticker):
        file_path = os.path.join(self.data_path, f"{ticker}.{self.exchange_code}.csv")
        if not os.path.isfile(file_path):
            print(f"Warning: CSV file not found for {ticker}: {file_path}")
            return None

        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        # Calculate adjusted prices
        adj_ratio = df['adjusted_close'] / df['close']

        df['open'] = df['open'] * adj_ratio
        df['high'] = df['high'] * adj_ratio
        df['low'] = df['low'] * adj_ratio
        df['close'] = df['adjusted_close']

        # Optional: reorder columns if needed
        df = df[[
            'open', 'high', 'low', 'close', 'volume',
        ]]
        df.sort_index(inplace=True)

        return df

```

# portwine/loaders/fred.py

```py
import os
import pandas as pd
from fredapi import Fred
from portwine.loaders.base import MarketDataLoader


class FREDMarketDataLoader(MarketDataLoader):
    """
    Loads data from the FRED (Federal Reserve Economic Data) system.

    This loader functions as a standard MarketDataLoader but with added capabilities:
    1. It can load existing parquet files from a specified directory
    2. If a file is not found and save_missing=True, it will attempt to download
       the data from FRED using the provided API key
    3. Downloaded data is saved as parquet for future use

    This is particularly useful for accessing economic indicators like interest rates,
    GDP, inflation metrics, and other macroeconomic data needed for advanced strategies.

    Parameters
    ----------
    data_path : str
        Directory where parquet files are stored
    api_key : str, optional
        FRED API key for downloading missing data
    save_missing : bool, default=False
        Whether to download and save missing data from FRED
    transform_to_daily : bool, default=True
        Convert non-daily data to daily frequency using forward-fill
    """

    # Source identifier for AlternativeMarketDataLoader
    SOURCE_IDENTIFIER = 'FRED'

    def __init__(self, data_path, api_key=None, save_missing=False, transform_to_daily=True):
        """
        Initialize the FRED market data loader.
        """
        super().__init__()
        self.data_path = data_path
        self.api_key = api_key
        self.save_missing = save_missing
        self.transform_to_daily = transform_to_daily

        self._fred_client = None

        # Create the data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    @property
    def fred_client(self):
        """
        Lazy initialization of the FRED client.
        """
        if self._fred_client is None and self.api_key:
            self._fred_client = Fred(api_key=self.api_key)
        return self._fred_client

    def load_ticker(self, ticker):
        """
        Load data for a specific ticker from parquet file or download from FRED.

        Parameters
        ----------
        ticker : str
            FRED series identifier (e.g., 'FEDFUNDS', 'DTB3', 'CPIAUCSL')

        Returns
        -------
        pd.DataFrame
            DataFrame with daily date index and appropriate columns for the portwine framework
        """
        file_path = os.path.join(self.data_path, f"{ticker}.parquet")

        # Check if file exists and load it
        if os.path.isfile(file_path):
            try:
                df = pd.read_parquet(file_path)
                return self._format_dataframe(df, ticker)
            except Exception as e:
                print(f"Error loading data for {ticker}: {str(e)}")

        # If file doesn't exist and save_missing is enabled, download from FRED
        if self.save_missing and self.fred_client:
            try:
                print(f"Downloading data for {ticker} from FRED...")
                # Get data from FRED
                series = self.fred_client.get_series(ticker)

                if series is not None and not series.empty:
                    # Convert to DataFrame with correct column name
                    df = pd.DataFrame(series, columns=['close'])

                    # Save to parquet for future use
                    df.to_parquet(file_path)

                    return self._format_dataframe(df, ticker)
                else:
                    print(f"No data found on FRED for ticker: {ticker}")
            except Exception as e:
                print(f"Error downloading data for {ticker} from FRED: {str(e)}")

        return None

    def _format_dataframe(self, df, ticker):
        """
        Format the dataframe to match the expected structure for portwine.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to format
        ticker : str
            Ticker symbol

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame with appropriate columns
        """
        # Make sure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Make sure the index name is 'date'
        df.index.name = 'date'

        # Convert index to date (not datetime)
        if hasattr(df.index, 'normalize'):
            df.index = df.index.normalize().to_period('D').to_timestamp()

        # If we have only a Series, convert to DataFrame with 'close' column
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df, columns=['close'])

        # Handle the case where the first column might be the values
        if 'close' not in df.columns and len(df.columns) >= 1:
            # Rename the first column to 'close'
            df = df.rename(columns={df.columns[0]: 'close'})

        # If we have frequency that's not daily and transform_to_daily is True
        if self.transform_to_daily:
            # Reindex to daily frequency with forward fill
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            df = df.reindex(date_range, method='ffill')

        # Ensure we have all required columns for portwine
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 0

        return df

    def get_fred_info(self, ticker):
        """
        Get information about a FRED series.

        Parameters
        ----------
        ticker : str
            FRED series identifier

        Returns
        -------
        pd.Series
            Series containing information about the FRED series
        """
        if self.fred_client:
            try:
                return self.fred_client.get_series_info(ticker)
            except Exception as e:
                print(f"Error getting info for {ticker}: {str(e)}")
        return None

    def search_fred(self, text, limit=10):
        """
        Search for FRED series by text.

        Parameters
        ----------
        text : str
            Text to search for
        limit : int, default=10
            Maximum number of results to return

        Returns
        -------
        pd.DataFrame
            DataFrame with search results
        """
        if self.fred_client:
            try:
                return self.fred_client.search(text, limit=limit)
            except Exception as e:
                print(f"Error searching FRED for '{text}': {str(e)}")
        return None

```

# portwine/loaders/noisy.py

```py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple


class NoisyMarketDataLoader:
    """
    A market data loader that injects noise scaled by rolling volatility.

    This implementation adds noise to price returns, with the magnitude of noise
    proportional to the local volatility (measured as rolling standard deviation
    of returns). This ensures the noise adapts to different market regimes.

    Parameters
    ----------
    base_loader : object
        A base loader with load_ticker(ticker) and fetch_data(tickers) methods
    noise_multiplier : float, optional
        Base multiplier for the noise magnitude (default: 1.0)
    volatility_window : int, optional
        Window size in days for rolling volatility calculation (default: 21)
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
            self,
            base_loader: Any,
            noise_multiplier: float = 1.0,
            volatility_window: int = 21,
            seed: Optional[int] = None
    ):
        self.base_loader = base_loader
        self.noise_multiplier = noise_multiplier
        self.volatility_window = volatility_window
        self._original_data: Dict[str, pd.DataFrame] = {}  # Cache for original data

        if seed is not None:
            np.random.seed(seed)

    def get_original_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get original data for a ticker from cache or load it.

        Parameters
        ----------
        ticker : str
            Ticker symbol

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with OHLCV data or None if not available
        """
        if ticker in self._original_data:
            return self._original_data[ticker]

        df = self.base_loader.load_ticker(ticker)
        if df is not None:
            self._original_data[ticker] = df.sort_index()
        return df

    def calculate_rolling_volatility(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate rolling standard deviation of returns.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with OHLCV data

        Returns
        -------
        numpy.ndarray
            Array of rolling volatility values
        """
        # Calculate returns
        returns = df['close'].pct_change().values

        # Initialize volatility array
        n = len(returns)
        volatility = np.ones(n)  # Default to 1 for scaling

        # Calculate overall volatility for initial values
        overall_std = np.std(returns[1:])  # Skip the first NaN value
        if not np.isfinite(overall_std) or overall_std <= 0:
            overall_std = 0.01  # Fallback if volatility is zero or invalid

        # Fill initial window with overall volatility to avoid NaNs
        volatility[:self.volatility_window] = overall_std

        # Calculate rolling standard deviation
        for i in range(self.volatility_window, n):
            window_returns = returns[i - self.volatility_window + 1:i + 1]
            # Remove any NaN values
            window_returns = window_returns[~np.isnan(window_returns)]
            if len(window_returns) > 0:
                vol = np.std(window_returns)
                # Ensure we have positive volatility
                volatility[i] = vol if np.isfinite(vol) and vol > 0 else overall_std
            else:
                volatility[i] = overall_std

        return volatility

    def generate_noise(self, size: int, volatility_scaling: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate volatility-scaled noise for OHLC data.

        Parameters
        ----------
        size : int
            Number of days to generate noise for
        volatility_scaling : numpy.ndarray
            Array of volatility scaling factors

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing noise for open, high, low, and close
        """

        # Generate base zero-mean noise
        def generate_zero_mean_noise(size):
            raw_noise = np.random.normal(0, 1, size=size)
            if size > 1:
                # Ensure perfect zero mean to avoid drift
                raw_noise = raw_noise - np.mean(raw_noise)
            return raw_noise

        # Generate noise for each price component and scale by volatility
        noise_open = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier
        noise_high = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier
        noise_low = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier
        noise_close = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier

        return noise_open, noise_high, noise_low, noise_close

    def inject_noise(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-scaled noise to OHLC data while preserving high/low consistency.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        df : pandas.DataFrame
            DataFrame with OHLCV data

        Returns
        -------
        pandas.DataFrame
            Copy of input DataFrame with adaptive noise added to OHLC values
        """
        if df is None or len(df) < 2:
            return df.copy() if df is not None else None

        df_result = df.copy()

        # Compute volatility scaling factors
        volatility_scaling = self.calculate_rolling_volatility(df)

        # Extract original values for vectorized operations
        orig_open = df['open'].values
        orig_high = df['high'].values
        orig_low = df['low'].values
        orig_close = df['close'].values

        # Create arrays for new values
        new_open = np.empty_like(orig_open)
        new_high = np.empty_like(orig_high)
        new_low = np.empty_like(orig_low)
        new_close = np.empty_like(orig_close)

        # First day remains unchanged
        new_open[0] = orig_open[0]
        new_high[0] = orig_high[0]
        new_low[0] = orig_low[0]
        new_close[0] = orig_close[0]

        # Pre-generate all noise at once with volatility scaling
        n_days = len(df) - 1
        noise_open, noise_high, noise_low, noise_close = self.generate_noise(n_days, volatility_scaling[1:])

        # Process each day
        for i in range(1, len(df)):
            prev_orig_close = orig_close[i - 1]
            prev_new_close = new_close[i - 1]

            # Calculate returns relative to previous close
            r_open = (orig_open[i] / prev_orig_close) - 1
            r_high = (orig_high[i] / prev_orig_close) - 1
            r_low = (orig_low[i] / prev_orig_close) - 1
            r_close = (orig_close[i] / prev_orig_close) - 1

            # Add noise to returns and calculate new prices
            tentative_open = prev_new_close * (1 + r_open + noise_open[i - 1])
            tentative_high = prev_new_close * (1 + r_high + noise_high[i - 1])
            tentative_low = prev_new_close * (1 + r_low + noise_low[i - 1])
            tentative_close = prev_new_close * (1 + r_close + noise_close[i - 1])

            # Ensure high is the maximum and low is the minimum
            new_open[i] = tentative_open
            new_close[i] = tentative_close
            new_high[i] = max(tentative_open, tentative_high, tentative_low, tentative_close)
            new_low[i] = min(tentative_open, tentative_high, tentative_low, tentative_close)

        # Update the result DataFrame
        df_result['open'] = new_open
        df_result['high'] = new_high
        df_result['low'] = new_low
        df_result['close'] = new_close

        return df_result

    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load ticker data and inject volatility-scaled noise.

        Parameters
        ----------
        ticker : str
            Ticker symbol

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with noisy OHLCV data or None if not available
        """
        df = self.get_original_ticker_data(ticker)
        if df is None:
            return None
        return self.inject_noise(ticker, df)

    def fetch_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers and inject volatility-scaled noise.

        Parameters
        ----------
        tickers : list of str
            List of ticker symbols

        Returns
        -------
        dict
            Dictionary mapping ticker symbols to DataFrames with noisy data
        """
        result = {}
        for ticker in tickers:
            df_noisy = self.load_ticker(ticker)
            if df_noisy is not None:
                result[ticker] = df_noisy
        return result

```

# portwine/loaders/polygon.py

```py
import os
import pandas as pd
from portwine.loaders.base import MarketDataLoader


class PolygonMarketDataLoader(MarketDataLoader):
    """
    Assumes storage as parquet files.
    """

    def __init__(self, data_path):
        """
        Parameters
        ----------
        data_path : str
            The directory path where CSV files are located.
        """
        self.data_path = data_path
        super().__init__()

    def load_ticker(self, ticker):
        file_path = os.path.join(self.data_path, f"{ticker}.parquet")
        if not os.path.isfile(file_path):
            print(f"Warning: Parquet file not found for {ticker}: {file_path}")
            return None

        df = pd.read_parquet(file_path)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df.set_index('date', inplace=True)
        df.drop(columns='timestamp', inplace=True)

        return df

```

# portwine/optimizer.py

```py
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

class Optimizer:
    """
    Base/abstract optimizer class to allow different optimization approaches.
    """

    def __init__(self, backtester):
        """
        Parameters
        ----------
        backtester : Backtester
            A pre-initialized Backtester object with run_backtest(...).
        """
        self.backtester = backtester

    def run_optimization(self, strategy_class, param_grid, **kwargs):
        """
        Abstract method: run an optimization approach (grid search, train/test, etc.)
        Must be overridden.
        """
        raise NotImplementedError("Subclasses must implement run_optimization.")

# ---------------------------------------------------------------------

class TrainTestSplitOptimizer(Optimizer):
    """
    An optimizer that implements Approach A (two separate backtests):
      1) For each param combo, run a backtest on [train_start, train_end].
         Evaluate performance (score_fn).
      2) Choose best combo by training score.
      3) Optionally run a second backtest on [test_start, test_end] for
         the best combo, storing final out-of-sample performance.
    """

    def __init__(self, backtester):
        super().__init__(backtester)

    def run_optimization(self,
                         strategy_class,
                         param_grid,
                         split_frac=0.7,
                         score_fn=None,
                         benchmark=None):
        """
        Runs a grid search over 'param_grid' for the given 'strategy_class'.

        This approach does two separate backtests:
          - one on the train portion
          - one on the test portion for the best param set

        Parameters
        ----------
        strategy_class : type
            A strategy class (e.g. SomeStrategy) that can be constructed via
            the keys in param_grid.
        param_grid : dict
            param name -> list of possible values
        split_frac : float
            fraction of the data for training (0<split_frac<1)
        score_fn : callable
            function that takes { 'strategy_returns': Series, ... } -> float
            If None, defaults to annualized Sharpe on the daily returns.
        benchmark : str or None
            optional benchmark passed to run_backtest.

        Returns
        -------
        dict with:
          'best_params' : dict of chosen param set
          'best_score'  : float
          'results_df'  : DataFrame of all combos with train_score (and maybe test_score)
          'best_test_performance' : float or None
        """
        # Default scoring => annualized Sharpe
        if score_fn is None:
            def default_sharpe(res):
                # res = {'strategy_returns': Series of daily returns, ...}
                dr = res.get('strategy_returns', pd.Series(dtype=float))
                if len(dr) < 2:
                    return np.nan
                ann = 252.0
                mu = dr.mean()*ann
                sigma = dr.std()*np.sqrt(ann)
                return mu/sigma if sigma>1e-9 else 0.0
            score_fn = default_sharpe

        # The first step is to fetch the union of all relevant data to determine the date range.
        # We can do that by a "fake" strategy with "all combos" or just pick the first param set's tickers.
        # But it's safer to gather the earliest/largest date range among all combos. We'll do a simplified approach:
        all_dates = []
        # We'll keep a dictionary mapping from each combo -> (mindate, maxdate).
        # 1) If param_grid has a 'tickers' entry that is a list of lists, convert them to tuples:
        if "tickers" in param_grid:
            converted = []
            for item in param_grid["tickers"]:
                if isinstance(item, list):
                    converted.append(tuple(item))  # cast from list -> tuple
                else:
                    converted.append(item)
            param_grid["tickers"] = converted

        combos_date_range = {}

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(product(*param_values))

        # We do a quick pass just to gather date range
        for combo in all_combos:
            combo_params = dict(zip(param_names, combo))
            # Make a strategy
            strat = strategy_class(**combo_params)
            if not hasattr(strat, 'tickers'):
                combos_date_range[combo] = (None, None)
                continue
            # fetch data
            data_dict = self.backtester.market_data_loader.fetch_data(strat.tickers)
            if not data_dict:
                combos_date_range[combo] = (None, None)
                continue
            # get min and max date across all relevant tickers
            min_d, max_d = None, None
            for tkr, df in data_dict.items():
                if df.empty:
                    continue
                dmin = df.index.min()
                dmax = df.index.max()
                if min_d is None or dmin>min_d:
                    min_d = dmin
                if max_d is None or dmax<max_d:
                    max_d = dmax
                # actually we want min_d = max of earliest
                # and max_d = min of latest
                # so let's invert:
                # if min_d is None => set it to dmin
                # else => min_d = max(min_d, dmin)
                # Similarly for max_d => min(max_d, dmax)
            combos_date_range[combo] = (min_d, max_d)

        # We'll pick the "largest common feasible range" among combos.
        # This is a design choice. Alternatively, we can do a combo-specific approach.
        # For simplicity, let's do a single union range for all combos.
        global_start = None
        global_end = None
        for c, (md, xd) in combos_date_range.items():
            if md is not None and xd is not None:
                if global_start is None or md>global_start:
                    global_start = md
                if global_end is None or xd<global_end:
                    global_end = xd
        if global_start is None or global_end is None or global_start>=global_end:
            print("No valid date range found for any combo.")
            return None

        # Now we have a global [global_start, global_end].
        # We'll pick a date range from that.
        # We'll use the naive approach: we convert them to sorted list of dates, pick split index, etc.
        # It's simpler to run two separate backtests with "start_date=..., end_date=..." for train, then for test.

        # Build a list of all daily timestamps from global_start..global_end
        # We can fetch from the underlying data in the backtester, or do a direct approach.
        # For simplicity, let's just do an approximate approach:
        # We'll gather the entire union in the same manner we do in the backtester:
        # But let's do it quickly:
        # We'll define a function get_all_dates_for_range:
        # Actually let's skip it, and do a simpler approach: we'll pass them to the backtester with end_date in train
        # and start_date in test.

        # We'll do the naive approach:
        # We want the full set of daily trading dates from global_start..global_end
        # Then we pick split. Then train is [global_start..some boundary], test is [boundary+1..global_end].
        # We'll do it with the backtester, but let's first gather a big combined df.

        # (In practice you might do a step to confirm we have a consistent set of daily trading dates. We'll do minimal approach.)
        date_range = pd.date_range(global_start, global_end, freq='B')  # business days
        n = len(date_range)
        split_idx = int(n*split_frac)
        if split_idx<1:
            print("Split fraction leaves no training days.")
            return None
        if split_idx>=n:
            print("Split fraction leaves no testing days.")
            return None
        train_end_date = date_range[split_idx-1]
        test_start_date = date_range[split_idx]
        # Summaries
        # We'll store combos results
        results_list = []

        for combo in tqdm(all_combos):
            combo_params = dict(zip(param_names, combo))
            # First => run training backtest
            strat_train = strategy_class(**combo_params)
            # train backtest
            train_res = self.backtester.run_backtest(
                strategy=strat_train,
                shift_signals=True,
                benchmark=benchmark,
                start_date=global_start,
                end_date=train_end_date
            )
            if not train_res or 'strategy_returns' not in train_res:
                results_list.append({**combo_params, "train_score": np.nan, "test_score": np.nan})
                continue
            train_dr = train_res['strategy_returns']
            if train_dr is None or len(train_dr)<2:
                results_list.append({**combo_params, "train_score": np.nan, "test_score": np.nan})
                continue

            # compute train score
            train_score = score_fn({"strategy_returns": train_dr})

            # We'll not do test backtest for each combo => to save time, do test only for best param after the loop
            # or if you want, you can do it here as well. We'll do it here so we can see how big the difference is for each combo.
            # But that can be expensive for big param grids. We'll do it anyway for completeness.

            strat_test = strategy_class(**combo_params)
            test_res = self.backtester.run_backtest(
                strategy=strat_test,
                shift_signals=True,
                benchmark=benchmark,
                start_date=test_start_date,
                end_date=global_end
            )
            if not test_res or 'strategy_returns' not in test_res:
                results_list.append({**combo_params, "train_score": train_score, "test_score": np.nan})
                continue
            test_dr = test_res['strategy_returns']
            if test_dr is None or len(test_dr)<2:
                results_list.append({**combo_params, "train_score": train_score, "test_score": np.nan})
                continue

            test_score = score_fn({"strategy_returns": test_dr})

            # store
            combo_result = {**combo_params,
                            "train_score": train_score,
                            "test_score": test_score}
            results_list.append(combo_result)

        df_results = pd.DataFrame(results_list)
        if df_results.empty:
            print("No results produced.")
            return None

        # pick best by train_score
        df_results.sort_values('train_score', ascending=False, inplace=True)
        best_row = df_results.iloc[0].to_dict()
        best_train_score = best_row['train_score']
        best_test_score = best_row['test_score']
        best_params = {k: v for k, v in best_row.items() if k not in ['train_score','test_score']}

        print("Best params:", best_params, f"train_score={best_train_score:.4f}, test_score={best_test_score:.4f}")

        return {
            "best_params": best_params,
            "best_score": best_train_score,
            "results_df": df_results,
            "best_test_score": best_test_score
        }

```

# portwine/scheduler.py

```py
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from typing import Iterator, Optional


def daily_schedule(
    after_open_minutes: Optional[int] = None,
    before_close_minutes: Optional[int] = None,
    calendar_name: str = 'NYSE',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval_seconds: Optional[int] = None,
    inclusive: bool = False
) -> Iterator[int]:
    """
    Generate UNIX‐ms timestamps for daily market open/close events with offsets.

    If after_open_minutes is None and before_close_minutes is not None: on‐close schedule.
    If after_open_minutes is not None and before_close_minutes is None: on‐open schedule.
    If both are not None: on‐open then on‐close each day.
    If both are None: raises ValueError.

    Args:
        after_open_minutes: Minutes after market open to schedule, or None.
        before_close_minutes: Minutes before market close to schedule, or None.
        calendar_name: Market calendar name (e.g. 'NYSE').
        start_date: ISO date string for start (e.g. '2023-01-01'); defaults to today.
        end_date: ISO date string for end; defaults to start_date.
        interval_seconds: Interval in seconds between points, or None for no interval.
        inclusive: Whether to include the end point if it's not already included.

    Yields:
        UNIX timestamp in milliseconds for each scheduled event.
    """
    if after_open_minutes is None and before_close_minutes is None:
        raise ValueError(
            "Must specify at least one of after_open_minutes or before_close_minutes"
        )

    calendar = mcal.get_calendar(calendar_name)
    # Default to today if no dates provided
    if start_date is None:
        start_date = datetime.now().date().isoformat()
    if end_date is None:
        end_date = start_date

    schedule_df = calendar.schedule(start_date=start_date, end_date=end_date)
    # Iterate each trading day and generate schedule
    for _, row in schedule_df.iterrows():
        market_open = row['market_open']
        market_close = row['market_close']
        # on-close only
        if after_open_minutes is None:
            # cannot specify interval for close-only schedule
            if interval_seconds is not None:
                raise ValueError(
                    "Cannot specify interval_seconds on a close-only schedule"
                )
            ts_close = market_close - timedelta(minutes=before_close_minutes)
            yield int(ts_close.timestamp() * 1000)
            continue
        # on-open only
        if before_close_minutes is None:
            ts_open = market_open + timedelta(minutes=after_open_minutes)
            if interval_seconds is None:
                yield int(ts_open.timestamp() * 1000)
            else:
                # generate from ts_open to market_close by interval_seconds
                delta = timedelta(seconds=interval_seconds)
                t = ts_open
                while t <= market_close:
                    yield int(t.timestamp() * 1000)
                    t += delta
            continue
        # both open and close with optional interval
        # compute start and end datetimes
        start_dt = market_open + timedelta(minutes=after_open_minutes)
        end_dt = market_close - timedelta(minutes=before_close_minutes)
        # no interval: just yield start and end when interval_seconds is None
        if interval_seconds is None:
            yield int(start_dt.timestamp() * 1000)
            yield int(end_dt.timestamp() * 1000)
        else:
            # generate points from start to end every interval_seconds
            delta = timedelta(seconds=interval_seconds)
            t = start_dt
            last_ts = None
            while t <= end_dt:
                yield int(t.timestamp() * 1000)
                last_ts = t
                t += delta
            # if inclusive and end_dt was not hit exactly, include it
            if inclusive and last_ts is not None and last_ts < end_dt:
                yield int(end_dt.timestamp() * 1000) 
```

# portwine/strategies/__init__.py

```py
from portwine.strategies.base import StrategyBase
from portwine.strategies.simple_moving_average import SimpleMovingAverageStrategy

```

# portwine/strategies/base.py

```py
class StrategyBase:
    """
    Base class for a trading strategy. Subclass this to implement a custom strategy.

    A 'step' method is called each day with that day's data. The method should return
    a dictionary of signals/weights for each ticker on that day.

    A strategy may also declare a separate set of 'alternative_data_tickers'
    it depends on (e.g., indices, macro data, etc.). The backtester can then fetch
    that data for the strategy's use.
    """

    def __init__(self, tickers):
        """
        Parameters
        ----------
        tickers : list
            List of primary ticker symbols that the strategy will manage.
        """
        self.tickers = tickers

    def step(self, current_date, daily_data):
        """
        Called each day with that day's data for each ticker.

        Parameters
        ----------
        current_date : pd.Timestamp
        daily_data : dict
            daily_data[ticker] = {
                'open': ..., 'high': ..., 'low': ...,
                'close': ..., 'volume': ...
            }
            or None if no data for that ticker on this date.

            daily_data[ticker] can also return any arbitrary dictionary or value as well.

            This is good for macroeconomic indices, alternative data, etc.

        Returns
        -------
        signals : dict
            { ticker -> float weight }, where the weights are the fraction
            of capital allocated to each ticker (long/short).
        """
        # Default: equally weight among all *primary* tickers that have data
        valid_tickers = [t for t in self.tickers if daily_data.get(t) is not None]
        n = len(valid_tickers)
        weight = 1.0 / n if n > 0 else 0.0
        signals = {tkr: weight for tkr in valid_tickers}
        return signals

    def rebalance_portfolio(self, current_positions, current_date):
        """
        Optional rebalancing logic can be overridden here.
        By default, returns current_positions unmodified.
        """
        return current_positions
```

# portwine/strategies/simple_moving_average.py

```py
"""
Simple Moving Average Strategy Implementation

This module provides a simple moving average crossover strategy that:
1. Calculates short and long moving averages for each ticker
2. Generates buy signals when short MA crosses above long MA
3. Generates sell signals when short MA crosses below long MA
"""

import logging
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

from portwine.strategies.base import StrategyBase

# Configure logging
logger = logging.getLogger(__name__)


class SimpleMovingAverageStrategy(StrategyBase):
    """
    Simple Moving Average Crossover Strategy
    
    This strategy:
    1. Calculates short and long moving averages for each ticker
    2. Generates buy signals when short MA crosses above long MA
    3. Generates sell signals when short MA crosses below long MA
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to trade
    short_window : int, default 20
        Short moving average window in days
    long_window : int, default 50
        Long moving average window in days
    position_size : float, default 0.1
        Position size as a fraction of portfolio (e.g., 0.1 = 10%)
    """
    
    def __init__(
        self, 
        tickers: List[str], 
        short_window: int = 20, 
        long_window: int = 50,
        position_size: float = 0.1,
        **kwargs
    ):
        """Initialize the strategy with parameters."""
        super().__init__(tickers)
        
        # Store parameters
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        
        # Validate parameters
        if short_window >= long_window:
            logger.warning("Short window should be smaller than long window")
        
        # Initialize price history
        self.price_history = {ticker: [] for ticker in tickers}
        self.dates = []
        
        # Current signals (allocations)
        self.current_signals = {ticker: 0.0 for ticker in tickers}
        
        logger.info(f"Initialized SimpleMovingAverageStrategy with {len(tickers)} tickers")
        logger.info(f"Parameters: short_window={short_window}, long_window={long_window}, position_size={position_size}")
    
    def calculate_moving_averages(self, prices: List[float]) -> Dict[str, Optional[float]]:
        """
        Calculate short and long moving averages from price history.
        
        Parameters
        ----------
        prices : List[float]
            List of historical prices
            
        Returns
        -------
        Dict[str, Optional[float]]
            Dictionary with short_ma and long_ma values, or None if not enough data
        """
        if len(prices) < self.long_window:
            return {"short_ma": None, "long_ma": None}
        
        # Calculate moving averages
        short_ma = sum(prices[-self.short_window:]) / self.short_window
        long_ma = sum(prices[-self.long_window:]) / self.long_window
        
        return {"short_ma": short_ma, "long_ma": long_ma}
    
    def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Process daily data and generate trading signals.
        
        Parameters
        ----------
        current_date : pd.Timestamp
            Current trading date
        daily_data : Dict[str, Dict[str, Any]]
            Dictionary of ticker data for the current date
            
        Returns
        -------
        Dict[str, float]
            Dictionary of target weights for each ticker
        """
        # Track dates
        self.dates.append(current_date)
        
        # Update price history
        for ticker in self.tickers:
            price = None
            if ticker in daily_data and daily_data[ticker] is not None:
                price = daily_data[ticker].get('close')
            
            # Forward fill missing data
            if price is None and len(self.price_history[ticker]) > 0:
                price = self.price_history[ticker][-1]
            
            self.price_history[ticker].append(price)
        
        # Calculate signals for each ticker
        signals = {}
        for ticker in self.tickers:
            prices = self.price_history[ticker]
            
            # Skip tickers with None values
            if None in prices:
                signals[ticker] = 0.0
                continue
            
            # Calculate moving averages
            mas = self.calculate_moving_averages(prices)
            
            # Not enough data yet
            if mas["short_ma"] is None or mas["long_ma"] is None:
                signals[ticker] = 0.0
                continue
            
            # Get previous signal
            prev_signal = self.current_signals.get(ticker, 0.0)
            
            # Generate signal based on moving average crossover
            if mas["short_ma"] > mas["long_ma"]:
                # Bullish signal - short MA above long MA
                signals[ticker] = self.position_size
            else:
                # Bearish signal - short MA below long MA
                signals[ticker] = 0.0
            
            # Log signal changes
            if signals[ticker] != prev_signal:
                direction = "BUY" if signals[ticker] > 0 else "SELL"
                logger.info(f"{current_date}: {direction} signal for {ticker} - Short MA: {mas['short_ma']:.2f}, Long MA: {mas['long_ma']:.2f}")
        
        # Update current signals
        self.current_signals = signals.copy()
        
        return signals
    
    def generate_signals(self) -> Dict[str, float]:
        """
        Generate current trading signals.
        
        This method is used by the DailyExecutor to get the current signals.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of target weights for each ticker
        """
        return self.current_signals.copy()
    
    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down SimpleMovingAverageStrategy") 
```

# pyproject.toml

```toml
[tool.poetry]
name = "portwine"
version = "0.1.1"
description = ""
authors = ["Stuart Farmer <stuart@lamden.io>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
pandas = "^2.2.3"
matplotlib = "^3.10.1"
scipy = "^1.15.2"
statsmodels = "^0.14.4"
tqdm = "^4.67.1"
cvxpy = "^1.6.4"
fredapi = "^0.5.2"
pandas-market-calendars = "^5.0.0"
requests = "^2.31.0"
pytz = "^2024.1"
django-scheduler = "^0.10.1"


[tool.poetry.group.dev.dependencies]
coverage = "^7.8.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

```

# README.md

```md
# portwine - a clean, elegant portfolio backtester
![The Triumph of Bacchus](imgs/header.jpg)
\`\`\`commandline
pip install portwine
\`\`\`
Portfolio construction, optimization, and backtesting can be a complicated web of data wrangling, signal generation, lookahead bias reduction, and parameter tuning.

But with `portwine`, strategies are clear and written in an 'online' fashion that removes most of the complexity that comes with backtesting, analyzing, and deploying your trading strategies.

---

### Simple Strategies
Strategies are only given the last day of prices to make their determinations and allocate weights. This allows them to be completely encapsulated and portable.

\`\`\`python
class SimpleMomentumStrategy(StrategyBase):
    """
    A simple momentum strategy that:
    1. Calculates N-day momentum for each ticker
    2. Invests in the top performing ticker
    3. Rebalances weekly (every Friday)
    
    This demonstrates a step-based strategy implementation in a concise, easy-to-understand way.
    """
    
    def __init__(self, tickers, lookback_days=10):
        """
        Parameters
        ----------
        tickers : list
            List of ticker symbols to consider for investment
        lookback_days : int
            Number of days to use for momentum calculation
        """
        super().__init__(tickers)
        self.lookback_days = lookback_days
        self.price_history = {ticker: [] for ticker in tickers}
        self.current_signals = {ticker: 0.0 for ticker in tickers}
        self.dates = []
    
    def is_friday(self, date):
        """Check if given date is a Friday (weekday 4)"""
        return date.weekday() == 4
    
    def calculate_momentum(self, ticker):
        """Calculate simple price momentum over lookback period"""
        prices = self.price_history[ticker]
        
        # Need at least lookback_days+1 data points
        if len(prices) <= self.lookback_days:
            return -999.0
        
        # Get starting and ending prices for momentum calculation
        start_price = prices[-self.lookback_days-1]
        end_price = prices[-1]
        
        # Check for valid prices
        if start_price is None or end_price is None or start_price <= 0:
            return -999.0
        
        # Return simple momentum (end/start - 1)
        return end_price / start_price - 1.0
    
    def step(self, current_date, daily_data):
        """
        Process daily data and determine allocations
        """
        # Track dates for rebalancing logic
        self.dates.append(current_date)
        
        # Update price history for each ticker
        for ticker in self.tickers:
            price = None
            if daily_data.get(ticker) is not None:
                price = daily_data[ticker].get('close', None)
            
            # Forward fill missing data
            if price is None and len(self.price_history[ticker]) > 0:
                price = self.price_history[ticker][-1]
                
            self.price_history[ticker].append(price)
        
        # Only rebalance on Fridays
        if self.is_friday(current_date):
            # Calculate momentum for each ticker
            momentum_scores = {}
            for ticker in self.tickers:
                momentum_scores[ticker] = self.calculate_momentum(ticker)
            
            # Find best performing ticker
            best_ticker = max(momentum_scores.items(), 
                             key=lambda x: x[1] if x[1] != -999.0 else -float('inf'))[0]
            
            # Reset all allocations to zero
            self.current_signals = {ticker: 0.0 for ticker in self.tickers}
            
            # Allocate 100% to best performer if we have valid momentum
            if momentum_scores[best_ticker] != -999.0:
                self.current_signals[best_ticker] = 1.0
        
        # Return current allocations
        return self.current_signals.copy()
\`\`\`

---

### Breezy Backtesting

Backtesting strategies is a breeze, as well. Simply tell the backtester where your data is located with a data loader manager and give it a strategy. You get results immediately.

\`\`\`python
universe = ['MTUM', 'VTV', 'VUG', 'IJR', 'MDY']
strategy = SimpleMomentumStrategy(tickers=universe, lookback_days=10)

data_loader = EODHDMarketDataLoader(data_path='../../../Developer/Data/EODHD/us_sorted/US/')
backtester = Backtester(market_data_loader=data_loader)
results = backtester.run_backtest(strategy, benchmark_ticker='SPY')
\`\`\`
---
### Streamlined Data
Managing data can be a massive pain. But as long as you have your daily flat files from EODHD or Polygon saved in a directory, the data loaders will manage the rest. You don't have to worry about anything except writing code.

---

### Effortless Analysis

After running a strategy through the backtester, put it through an array of analyzers that are simple, visual, and clear. You can easily add your own analyzers to discover anything you need to know about your portfolio's performance, risk management, volatility, etc.

Check out what comes out of the box:

##### Equity Drawdown Analysis
\`\`\`python
EquityDrawdownAnalyzer().plot(results)
\`\`\`
![Equity Drawdown](imgs/equitydrawdown.jpg)

---
##### Monte Carlo Analysis
\`\`\`python
MonteCarloAnalyzer().plot(results)
\`\`\`
![Equity Drawdown](imgs/montecarlo.jpg)

---
##### Seasonality Analysis
\`\`\`python
EquityDrawdownAnalyzer().plot(results)
\`\`\`
![Equity Drawdown](imgs/seasonality.jpg)

With more on the way!

---
*Docs and tutorials coming soon.*

```

