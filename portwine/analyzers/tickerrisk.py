import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portwine.analyzers.base import Analyzer
from tqdm import tqdm


class TickerRiskAnalyzer(Analyzer):
    """
    This analyzer:
      1) Computes per-ticker risk metrics (Sharpe, Sortino, MaxDD, AnnualVol) for each
         ticker in 'tickers_returns' from the full backtest results.
      2) For each ticker in the original strategy, re-runs the backtest with that ticker excluded,
         comparing the new portfolio stats (Sharpe, total return, maxDD, etc.) to the full portfolio.
      3) Produces a plot with:
         - A single Equity Curve subplot (top row) showing the “Full” portfolio plus each “Exclude Ticker X”
           portfolio in the same figure, with consistent colors.
         - Bar charts for:
            * Per-ticker Sharpe
            * Per-ticker Sortino
            * Sharpe difference (Exclude - Full)
            * MaxDD difference (Exclude - Full)
         Each ticker uses the same color across the equity curve and bar charts,
         so it’s easier to follow.

    Usage:
      analyzer = TickerRiskAnalyzer(backtester, strategy)
      # run full backtest => results
      analyzer.plot(results)
    """

    def __init__(self, backtester, original_strategy, ann_factor=252):
        """
        Parameters
        ----------
        backtester : object
            A backtester instance that can run_backtest(strategy=...).
        original_strategy : StrategyBase
            The original strategy used in the full run. We'll clone it, excluding each ticker.
        ann_factor : int
            Annualization factor (252 for daily, 12 for monthly, etc.)
        """
        self.backtester = backtester
        self.original_strategy = original_strategy
        self.ann_factor = ann_factor

        # Pre-assign colors to each ticker in the original strategy for consistency.
        # We'll use a tab10 palette by default (10 distinct colors).
        # If user has more than 10 tickers, it cycles through again.
        self.tickers = list(self.original_strategy.tickers)
        cmap = plt.colormaps.get_cmap("tab10")  # Matplotlib >=3.7 approach
        self.color_map = {}
        for i, tkr in enumerate(self.tickers):
            self.color_map[tkr] = cmap(i % 10)

        # We'll use black for the "Full Strategy" curve.
        self.full_strategy_color = "black"

    def analyze(self, results, verbose=False):
        """
        1) Per-ticker metrics from results['tickers_returns'].
        2) Re-run the strategy for each ticker excluded => gather stats & equity curve.
        3) Return a dict with:
           - per_ticker_metrics (DataFrame)
           - portfolio_comparison (DataFrame or None)
           - full_equity_curve (Series)
           - excluded_equity_curves ({ticker: Series})
        """
        # --- Per-Ticker Metrics ---
        tickers_returns = results.get('tickers_returns')
        if tickers_returns is None or tickers_returns.empty:
            print("Error: 'tickers_returns' missing or empty in results.")
            return None

        # Build table of Sharpe, Sortino, MaxDD, AnnualVol for each ticker
        ticker_stats_list = []

        for tkr in tickers_returns.columns:
            dr = tickers_returns[tkr].dropna()
            if len(dr) < 2:
                continue
            stats = self._compute_per_ticker_stats(dr)
            stats['Ticker'] = tkr
            ticker_stats_list.append(stats)

        if ticker_stats_list:
            per_ticker_metrics = pd.DataFrame(ticker_stats_list).set_index('Ticker')
        else:
            per_ticker_metrics = pd.DataFrame(
                columns=['Sharpe', 'Sortino', 'MaxDD', 'AnnualVol']
            )

        # --- Full portfolio stats & equity curve ---
        full_strat_returns = results.get('strategy_returns', pd.Series(dtype=float))
        full_equity_curve = None
        full_stats = None
        if not full_strat_returns.empty:
            full_equity_curve = (1 + full_strat_returns).cumprod()
            full_stats = self._compute_portfolio_stats(full_strat_returns)

        # --- Excluding Ticker Comparisons ---
        portfolio_comparison = None
        excluded_equity_curves = {}
        if full_stats is not None:
            comp_rows = []

            if verbose:
                ticker_iter = tqdm(self.tickers, desc="Analyzing...")
            else:
                ticker_iter = self.tickers

            # For each ticker in the original strategy, remove it, re-run
            for tkr in ticker_iter:
                excl_result = self._run_excluding_ticker(tkr)
                if excl_result is None:
                    continue
                # Gather new equity and stats
                daily_excl_returns = excl_result.get('strategy_returns', pd.Series(dtype=float))
                if daily_excl_returns.empty:
                    continue

                eq_excl = (1 + daily_excl_returns).cumprod()
                excluded_equity_curves[tkr] = eq_excl  # store for plot

                excl_stats = self._compute_portfolio_stats(daily_excl_returns)
                row = {
                    'TickerExcluded': tkr,
                    'Full_TotalRet': full_stats['TotalRet'],
                    'Full_AnnualVol': full_stats['AnnualVol'],
                    'Full_Sharpe': full_stats['Sharpe'],
                    'Full_MaxDD': full_stats['MaxDrawdown'],
                    'Excl_TotalRet': excl_stats['TotalRet'],
                    'Excl_AnnualVol': excl_stats['AnnualVol'],
                    'Excl_Sharpe': excl_stats['Sharpe'],
                    'Excl_MaxDD': excl_stats['MaxDrawdown'],
                    'Sharpe_Diff': excl_stats['Sharpe'] - full_stats['Sharpe'],
                    'MaxDD_Diff': excl_stats['MaxDrawdown'] - full_stats['MaxDrawdown']
                }
                comp_rows.append(row)

            if comp_rows:
                portfolio_comparison = pd.DataFrame(comp_rows).set_index('TickerExcluded')

        return {
            'per_ticker_metrics': per_ticker_metrics,
            'portfolio_comparison': portfolio_comparison,
            'full_equity_curve': full_equity_curve,
            'excluded_equity_curves': excluded_equity_curves
        }

    def plot(self, results, verbose=False):
        """
        1) Calls analyze(results).
        2) Produces a figure with 3 rows, 2 columns => total 5 subplots:

           Row 0 (cols=0..1 merged): Equity curves
             - "Full" in black
             - "Exclude Ticker X" in color_map[X]

           Row 1, col=0: Per-ticker Sharpe
           Row 1, col=1: Per-ticker Sortino

           Row 2, col=0: Sharpe_Diff (Exclude - Full)
           Row 2, col=1: MaxDD_Diff  (Exclude - Full)
        3) The color used for each ticker’s bars/lines is consistent across the subplots.
        4) Grids are added to each axis, for improved readability.
        """
        analysis_dict = self.analyze(results, verbose=verbose)
        if not analysis_dict:
            print("No analysis results to plot.")
            return

        per_ticker = analysis_dict['per_ticker_metrics']
        port_comp = analysis_dict['portfolio_comparison']
        full_eq = analysis_dict['full_equity_curve']
        excl_eq_map = analysis_dict['excluded_equity_curves']

        # if everything is empty, no data
        if (per_ticker.empty and not port_comp) and (full_eq is None or full_eq.empty):
            print("No metrics or equity data to plot.")
            return

        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(nrows=3, ncols=2, hspace=0.4, wspace=0.3)

        # === Row 0 => merged columns => equity
        ax_eq = fig.add_subplot(gs[0, :])
        if full_eq is not None and not full_eq.empty:
            ax_eq.plot(full_eq.index, full_eq.values, label="Full Strategy",
                       color=self.full_strategy_color, linewidth=2)
        if excl_eq_map:
            for tkr, eq_ser in excl_eq_map.items():
                clr = self.color_map.get(tkr, "gray")
                ax_eq.plot(eq_ser.index, eq_ser.values,
                           label=f"Excl {tkr}",
                           color=clr, alpha=0.85)
        ax_eq.set_title("Equity Curves: Full vs. Excluding Ticker")
        ax_eq.legend(loc='best')
        ax_eq.tick_params(axis='x', rotation=45)
        ax_eq.grid(True, alpha=0.3)

        # === Row 1, col=0 => Per-ticker Sharpe
        ax_sharpe = fig.add_subplot(gs[1, 0])
        if not per_ticker.empty:
            df_sharpe = per_ticker.sort_values('Sharpe', ascending=False)
            xvals = df_sharpe.index
            height = df_sharpe['Sharpe']
            # color each bar by the ticker's assigned color
            bar_colors = [self.color_map.get(t, "gray") for t in xvals]
            ax_sharpe.bar(xvals, height, color=bar_colors)
            ax_sharpe.set_title("Per-Ticker Sharpe")
            ax_sharpe.tick_params(axis='x', rotation=45)
            ax_sharpe.grid(True, alpha=0.3)
        else:
            ax_sharpe.text(0.5, 0.5, "No per-ticker data",
                           ha='center', va='center')
            ax_sharpe.grid(True, alpha=0.3)

        # === Row 1, col=1 => Per-ticker Sortino
        ax_sortino = fig.add_subplot(gs[1, 1])
        if not per_ticker.empty:
            df_sortino = per_ticker.sort_values('Sortino', ascending=False)
            xvals = df_sortino.index
            height = df_sortino['Sortino']
            bar_colors = [self.color_map.get(t, "gray") for t in xvals]
            ax_sortino.bar(xvals, height, color=bar_colors)
            ax_sortino.set_title("Per-Ticker Sortino")
            ax_sortino.tick_params(axis='x', rotation=45)
            ax_sortino.grid(True, alpha=0.3)
        else:
            ax_sortino.text(0.5, 0.5, "No per-ticker data",
                            ha='center', va='center')
            ax_sortino.grid(True, alpha=0.3)

        # === Row 2, col=0 => Sharpe_Diff
        ax_diff_sharpe = fig.add_subplot(gs[2, 0])
        if port_comp is not None and not port_comp.empty:
            df_sh = port_comp.sort_values('Sharpe_Diff', ascending=False)
            xvals = df_sh.index
            height = df_sh['Sharpe_Diff']
            bar_colors = [self.color_map.get(t, "gray") for t in xvals]
            ax_diff_sharpe.bar(xvals, height, color=bar_colors)
            ax_diff_sharpe.axhline(y=0, color='k', linewidth=1)
            ax_diff_sharpe.set_title("Change in Sharpe (Exclude - Full)")
            ax_diff_sharpe.tick_params(axis='x', rotation=45)
            ax_diff_sharpe.grid(True, alpha=0.3)
        else:
            ax_diff_sharpe.text(0.5, 0.5, "No exclude-ticker data",
                                ha='center', va='center')
            ax_diff_sharpe.axhline(y=0, color='k', linewidth=1)
            ax_diff_sharpe.grid(True, alpha=0.3)

        # === Row 2, col=1 => MaxDD_Diff
        ax_diff_maxdd = fig.add_subplot(gs[2, 1])
        if port_comp is not None and not port_comp.empty:
            df_md = port_comp.sort_values('MaxDD_Diff', ascending=False)
            xvals = df_md.index
            height = df_md['MaxDD_Diff']
            bar_colors = [self.color_map.get(t, "gray") for t in xvals]
            ax_diff_maxdd.bar(xvals, height, color=bar_colors)
            ax_diff_maxdd.axhline(y=0, color='k', linewidth=1)
            ax_diff_maxdd.set_title("Change in MaxDD (Exclude - Full)")
            ax_diff_maxdd.tick_params(axis='x', rotation=45)
            ax_diff_maxdd.grid(True, alpha=0.3)
        else:
            ax_diff_maxdd.text(0.5, 0.5, "No exclude-ticker data",
                               ha='center', va='center')
            ax_diff_maxdd.axhline(y=0, color='k', linewidth=1)
            ax_diff_maxdd.grid(True, alpha=0.3)

        plt.show()

    # -------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------
    def _compute_per_ticker_stats(self, daily_returns):
        """
        For a single ticker's daily returns: Sharpe, Sortino, MaxDD, AnnualVol
        """
        if daily_returns is None or daily_returns.empty:
            return {'Sharpe': np.nan, 'Sortino': np.nan, 'MaxDD': np.nan, 'AnnualVol': np.nan}
        dr = daily_returns.dropna()
        if len(dr) < 2:
            return {'Sharpe': np.nan, 'Sortino': np.nan, 'MaxDD': np.nan, 'AnnualVol': np.nan}

        ann_factor = self.ann_factor
        avg_ret = dr.mean()
        std_ret = dr.std()
        ann_ret = avg_ret * ann_factor
        ann_vol = std_ret * np.sqrt(ann_factor)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else np.nan

        neg = dr[dr < 0]
        neg_std = neg.std()
        if neg_std is None or np.isnan(neg_std) or neg_std < 1e-9:
            sortino = np.nan
        else:
            sortino = ann_ret / (neg_std * np.sqrt(ann_factor))

        eq = (1 + dr).cumprod()
        maxdd = self._compute_drawdown(eq).min()

        return {
            'Sharpe': sharpe,
            'Sortino': sortino,
            'MaxDD': maxdd,
            'AnnualVol': ann_vol
        }

    def _compute_portfolio_stats(self, daily_returns):
        """
        For the portfolio's daily returns: total ret, annual vol, Sharpe, maxDD
        """
        if daily_returns is None or daily_returns.empty:
            return None
        dr = daily_returns.dropna()
        if len(dr) < 2:
            return None

        ann_factor = self.ann_factor
        tot_ret = (1 + dr).prod() - 1.0
        ann_vol = dr.std() * np.sqrt(ann_factor)
        ann_ret = dr.mean() * ann_factor
        sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else np.nan

        eq = (1 + dr).cumprod()
        maxdd = self._compute_drawdown(eq).min()

        return {
            'TotalRet': tot_ret,
            'AnnualVol': ann_vol,
            'Sharpe': sharpe,
            'MaxDrawdown': maxdd
        }

    def _compute_drawdown(self, equity_series):
        roll_max = equity_series.cummax()
        dd = (equity_series - roll_max) / roll_max
        return dd

    def _run_excluding_ticker(self, tkr):
        """
        1) Clone the strategy minus 'tkr'.
        2) run_backtest => gather the results.
        """
        new_strat = self._clone_strategy_excluding(tkr)
        if not new_strat.tickers:
            return None
        return self.backtester.run_backtest(strategy=new_strat)

    def _clone_strategy_excluding(self, tkr_excl):
        """
        Produce a new Strategy object with 'tkr_excl' removed from original_strategy.tickers
        using a simple approach. Adjust as needed for your environment.
        """
        old_list = list(self.original_strategy.tickers)
        if tkr_excl in old_list:
            old_list.remove(tkr_excl)

        if len(old_list) < 1:
            return None  # no point in re-running with zero tickers

        # create new strategy instance
        StrategyClass = self.original_strategy.__class__
        new_strategy = StrategyClass(old_list)
        return new_strategy
