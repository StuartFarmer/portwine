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

    def plot(self, results, title="Monte Carlo Simulations (Log Scale)", figsize=(15, 10)):
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
        fig = plt.figure(figsize=figsize)
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
