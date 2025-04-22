import matplotlib.pyplot as plt
from portwine.analyzers import Analyzer

class PerformanceAttributionAnalyzer(Analyzer):
    """
    This analyzer shows how each ticker contributes to the portfolio's performance,
    given:
      - signals_df (the daily portfolio weights for each ticker),
      - tickers_returns (each ticker's daily returns).
    """

    def __init__(self):
        pass

    def analyze(self, results):
        """
        Given a results dict with:
          {
            'signals_df':      DataFrame of daily weights per ticker,
            'tickers_returns': DataFrame of daily returns per ticker,
            'strategy_returns': Series of daily strategy returns (optional),
            'benchmark_returns': Series of daily benchmark returns (optional)
          }

        We compute:
          - daily_contrib:  DataFrame of daily return contributions per ticker
          - cumulative_contrib: DataFrame of the cumulative sum of these contributions
          - final_contrib:  final sum (scalar) of each ticker's contribution to total PnL

        Returns an attribution dict:
          {
            'daily_contrib':        DataFrame,
            'cumulative_contrib':   DataFrame,
            'final_contrib':        Series
          }
        """
        signals_df = results.get('signals_df')
        tickers_returns = results.get('tickers_returns')

        if signals_df is None or tickers_returns is None:
            print("Error: 'signals_df' or 'tickers_returns' missing in results.")
            return None

        # Align indexes & columns to ensure multiplication is valid
        signals_df, tickers_returns = signals_df.align(tickers_returns, join='inner', axis=1)
        signals_df, tickers_returns = signals_df.align(tickers_returns, join='inner', axis=0)

        daily_contrib = signals_df * tickers_returns
        daily_contrib = daily_contrib.fillna(0.0)
        cumulative_contrib = daily_contrib.cumsum()
        final_contrib = daily_contrib.sum(axis=0)

        return {
            'daily_contrib':      daily_contrib,
            'cumulative_contrib': cumulative_contrib,
            'final_contrib':      final_contrib
        }

    def plot(self, results):
        """
        Plots both:
          1) Cumulative contribution per ticker over time
          2) Final total contribution per ticker
        as a single figure with two subplots.
        """
        attribution = self.analyze(results)
        if attribution is None:
            print("No attribution data to plot.")
            return

        cumulative_contrib = attribution['cumulative_contrib']
        final_contrib = attribution['final_contrib']

        if cumulative_contrib.empty or final_contrib.empty:
            print("Attribution data is empty. Nothing to plot.")
            return

        tickers = cumulative_contrib.columns.tolist()

        # Create one figure with two subplots stacked vertically
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

        # 1) Line chart: Cumulative contribution over time
        for tkr in tickers:
            ax1.plot(cumulative_contrib.index, cumulative_contrib[tkr], label=tkr)
        ax1.set_title("Cumulative Contribution per Ticker")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Contribution (fraction of initial capital)")
        ax1.legend(loc='best')
        ax1.grid(True)

        # 2) Bar chart: Final total contribution
        final_contrib.plot(kind='bar', ax=ax2, alpha=0.7)
        ax2.set_title("Final Total Contribution by Ticker")
        ax2.set_xlabel("Ticker")
        ax2.set_ylabel("Total Contribution")
        ax2.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()
