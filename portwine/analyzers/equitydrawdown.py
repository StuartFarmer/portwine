import numpy as np
import matplotlib.pyplot as plt
from portwine.analyzers.base import Analyzer

class EquityDrawdownAnalyzer(Analyzer):
    """
    Provides common analysis functionality, including drawdown calculation,
    summary stats, and plotting.
    """

    def compute_drawdown(self, equity_series):
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        return drawdown

    def analyze_returns(self, daily_returns, ann_factor=252):
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

    def plot(self, results, benchmark_label="Benchmark", log_scale=False):
        """
        Plots the strategy equity curve (and benchmark if given) plus drawdowns.
        Also prints summary stats.

        Parameters
        ----------
        results : dict
            Results from the backtest. Contains:
            - 'strategy_returns': pd.Series
            - 'benchmark_returns': pd.Series
        benchmark_label : str
            Label for the benchmark in legends.
        log_scale : bool
            If True, plot the equity curves on a logarithmic y-axis.
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

        strategy_equity_curve = (1.0 + results['strategy_returns']).cumprod()
        benchmark_equity_curve = (1.0 + results['benchmark_returns']).cumprod()

        # Optional log scale for equity
        if log_scale:
            ax1.set_yscale('log')

        # Plot equity curves with original styling
        ax1.plot(
            strategy_equity_curve.index,
            strategy_equity_curve.values,
            label="Strategy",
            color='mediumblue',
            linewidth=1,
            alpha=0.6
        )
        ax1.plot(
            benchmark_equity_curve.index,
            benchmark_equity_curve.values,
            label=benchmark_label,
            color='black',
            linewidth=0.5,
            alpha=0.5
        )
        ax1.set_title(
            "Equity Curve (relative, starts at 1.0)" + (" [log scale]" if log_scale else "")
        )
        ax1.legend(loc='best')
        ax1.grid(True)

        # Fill between strategy and benchmark
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

        # Plot drawdowns with original styling
        strat_dd = self.compute_drawdown(strategy_equity_curve) * 100.0
        bm_dd = self.compute_drawdown(benchmark_equity_curve) * 100.0

        ax2.plot(
            strat_dd.index,
            strat_dd.values,
            label="Strategy DD (%)",
            color='mediumblue',
            linewidth=1,
            alpha=0.6
        )
        ax2.plot(
            bm_dd.index,
            bm_dd.values,
            label=f"{benchmark_label} DD (%)",
            color='black',
            linewidth=0.5,
            alpha=0.5
        )
        ax2.set_title("Drawdown (%)")
        ax2.legend(loc='best')
        ax2.grid(True)

        # Fill between drawdown lines
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

        print("\n=== Strategy vs. Benchmark (Percentage Difference) ===")
        for k in strategy_stats.keys():
            strat_val = strategy_stats.get(k)
            bench_val = benchmark_stats.get(k)
            if strat_val is None or bench_val is None:
                print(f"{k}: N/A (missing data)")
                continue
            if isinstance(strat_val, (int, float)) and isinstance(bench_val, (int, float)):
                diff = (strat_val - bench_val) / abs(bench_val) if abs(bench_val) > 1e-15 else None
                print(f"{k}: {diff * 100:.2f}%" if diff is not None else f"{k}: N/A")
            else:
                print(f"{k}: N/A (non-numeric)")
