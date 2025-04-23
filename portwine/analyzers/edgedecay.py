import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from portwine.analyzers.base import Analyzer


class EdgeDecayAnalyzer(Analyzer):
    """
    Determines whether a trading strategy's outperformance factor
    (strategy_equity / benchmark_equity) is decaying over time.

    Rolling slope is computed by OLS of ln(outperf)[t0..tN] ~ a + b*x,
    now capturing b (the slope) + p-value. 
    If slope < 0 and p < 0.05 => significant negative slope => edge decay.
    """

    def analyze(self, results, rolling_window=60, alpha=0.05):
        """
        1) Build strategy & benchmark equity curves from daily returns
        2) outperformance(t) = strategy_eq(t) / bench_eq(t)
        3) rolling_slope: slope, p_value, etc. from OLS regression of ln(outperformance)
           on x=[0..window-1], in each rolling window.

        Returns
        -------
        dict:
          {
            'strategy_equity': pd.Series,
            'benchmark_equity': pd.Series,
            'outperformance': pd.Series,
            'rolling_stats': pd.DataFrame with columns=[slope, p_value, significance],
          }
        """
        strat_ret = results.get('strategy_returns', pd.Series(dtype=float)).dropna()
        bench_ret = results.get('benchmark_returns', pd.Series(dtype=float)).dropna()
        if strat_ret.empty or bench_ret.empty:
            print("EdgeDecayAnalyzer: Strategy or benchmark daily returns are empty!")
            return {}

        # Align on common dates
        common_idx = strat_ret.index.intersection(bench_ret.index)
        strat_ret = strat_ret.loc[common_idx].sort_index()
        bench_ret = bench_ret.loc[common_idx].sort_index()
        if len(strat_ret) < rolling_window:
            print(f"Not enough data for rolling_window={rolling_window}.")
            return {}

        # Build equity curves
        strategy_eq = (1.0 + strat_ret).cumprod()
        bench_eq = (1.0 + bench_ret).cumprod()

        # Outperformance = strategy_eq / bench_eq
        bench_eq_clipped = bench_eq.replace(0.0, np.nan).ffill().dropna()
        bench_eq_aligned = bench_eq_clipped.reindex(strategy_eq.index).ffill().fillna(0)
        outperf = strategy_eq / bench_eq_aligned
        outperf = outperf.dropna()
        if len(outperf) < rolling_window:
            print(f"Not enough overlapping equity data for rolling_window={rolling_window}.")
            return {}

        # Compute rolling slope via OLS
        rolling_stats_df = self._rolling_outperf_slope_ols(outperf, rolling_window)

        return {
            'strategy_equity': strategy_eq,
            'benchmark_equity': bench_eq_aligned,
            'outperformance': outperf,
            'rolling_stats': rolling_stats_df
        }

    def plot(self, results, rolling_window=60, alpha=0.05, figsize=(12, 8)):
        """
        Plots:
          1) Outperformance factor (top subplot)
          2) Rolling slope of ln(outperf) (bottom subplot), colored by significance if p< alpha.
          3) Adds a trend line to the rolling slope chart to show overall decay trend.
        """
        data = self.analyze(results, rolling_window=rolling_window, alpha=alpha)
        if not data:
            return

        strat_eq = data['strategy_equity']
        bench_eq = data['benchmark_equity']
        outperf = data['outperformance']
        rolling_stats_df = data['rolling_stats']

        fig, (ax_top, ax_bot) = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
        fig.suptitle(
            f"Edge Decay Analysis (rolling_window={rolling_window}, alpha={alpha:.2f})",
            fontsize=13
        )

        # Top subplot: outperformance factor
        ax_top.plot(outperf.index, outperf.values,
                    label="Outperformance = Strategy / Benchmark",
                    color='blue')
        ax_top.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax_top.set_title("Strategy Outperformance Factor")
        ax_top.set_ylabel("Strat / Bench")
        ax_top.grid(True)
        ax_top.legend(loc='best')

        # Fill red/green vs 1.0
        ax_top.fill_between(
            outperf.index, outperf.values, 1.0,
            where=(outperf.values >= 1.0),
            color='green', alpha=0.1
        )
        ax_top.fill_between(
            outperf.index, outperf.values, 1.0,
            where=(outperf.values < 1.0),
            color='red', alpha=0.1
        )

        # Bottom subplot: slope of ln(outperf)
        # We'll color by significance
        slope = rolling_stats_df['slope']
        pval = rolling_stats_df['p_value']
        # We'll treat negative slope as potentially decaying
        # We'll make a line from each point to the next, but change color if p< alpha
        x_vals = slope.index
        slopes_np = slope.values
        pvals_np = pval.values

        # Plot each segment with appropriate coloring
        for i in range(len(x_vals) - 1):
            x_seg = x_vals[i:i + 2]
            y_seg = slopes_np[i:i + 2]
            p_seg = max(pvals_np[i], pvals_np[i + 1])  # if either is < alpha => we'll color it
            if p_seg < alpha:
                color_ = 'red' if y_seg.mean() < 0 else 'green'
            else:
                color_ = 'black'
            ax_bot.plot(x_seg, y_seg, color=color_, linewidth=2)

        # Add trend line through all the rolling slope data points
        if len(x_vals) > 1:
            # Convert dates to numerical format for regression
            x_numeric = np.array([(d - x_vals[0]).total_seconds() for d in x_vals])

            # Fit linear regression on all slope data points
            # Filter out NaN values
            mask = ~np.isnan(slopes_np)
            if np.sum(mask) > 1:  # Need at least 2 points for regression
                X = sm.add_constant(x_numeric[mask])
                model = sm.OLS(slopes_np[mask], X).fit()

                # Calculate trend line values
                trend_x = np.array([x_numeric[mask].min(), x_numeric[mask].max()])
                trend_y = model.params[0] + model.params[1] * trend_x

                # Convert back to datetime for plotting
                trend_dates = [x_vals[0] + pd.Timedelta(seconds=float(tx)) for tx in trend_x]

                # Plot the trend line and add label with slope information
                trend_slope = model.params[1] * 86400  # Convert to daily slope
                trend_pval = model.pvalues[1]
                significance = trend_pval < alpha
                trend_label = f"Overall Trend: {trend_slope:.2e}/day"
                if significance:
                    trend_label += f" (p={trend_pval:.3f})*"
                    trend_color = 'red' if trend_slope < 0 else 'green'
                else:
                    trend_label += f" (p={trend_pval:.3f})"
                    trend_color = 'blue'

                ax_bot.plot(trend_dates, trend_y, color=trend_color, linestyle='-',
                            linewidth=2, label=trend_label)

        ax_bot.axhline(0.0, color='gray', linestyle='--', alpha=0.5,
                       label="Slope=0 => no decay")
        ax_bot.set_title("Rolling Slope of ln(Outperformance) + p-value Sig")
        ax_bot.set_ylabel("Slope (log outperf / day)")
        ax_bot.legend(loc='best')
        ax_bot.grid(True)

        plt.tight_layout()
        plt.show()

    ###########################################################################
    # Internal function with OLS each window to get slope + p_value
    ###########################################################################
    def _rolling_outperf_slope_ols(self, outperf_series, window):
        """
        For each rolling window of length 'window', do:
          y = ln(outperf[t0..tN])
          x = 0..(N-1)
        OLS => y ~ a + b*x => slope = b, pval = p-value of slope.
        Return a DataFrame with columns: slope, p_value
        """
        y_log = np.log(outperf_series.replace([np.inf, -np.inf], np.nan).dropna())
        if len(y_log) < window:
            return pd.DataFrame()

        idx_vals = y_log.index.to_numpy()
        y_vals = y_log.values

        slope_list = []
        pval_list = []
        date_list = []

        x_base = np.arange(window)

        for i in range(window, len(y_vals) + 1):
            seg_y = y_vals[i - window:i]
            seg_dates = idx_vals[i - window:i]
            # OLS => seg_y ~ alpha + slope*x_base
            X = sm.add_constant(x_base)
            try:
                model = sm.OLS(seg_y, X).fit()
                slope_ = model.params[1]
                pval_ = model.pvalues[1]
            except:
                slope_, pval_ = np.nan, np.nan

            slope_list.append(slope_)
            pval_list.append(pval_)
            date_list.append(seg_dates[-1])  # label by the last day in window

        df = pd.DataFrame({
            'slope': slope_list,
            'p_value': pval_list
        }, index=pd.to_datetime(date_list))
        df.index.name = 'date'
        return df