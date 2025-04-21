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
        # (In case they don't exactly match)
        signals_df, tickers_returns = signals_df.align(tickers_returns, join='inner', axis=1)
        # Also align on dates
        signals_df, tickers_returns = signals_df.align(tickers_returns, join='inner', axis=0)

        # daily_contrib[ticker] = signals_df[ticker] * tickers_returns[ticker]
        # This is the fraction of the portfolio in that ticker times its return,
        # i.e. the "contribution" of that ticker's daily return to the overall portfolio.
        daily_contrib = signals_df * tickers_returns

        # Replace any NaN with 0, if needed
        daily_contrib = daily_contrib.fillna(0.0)

        # The cumulative contribution of each ticker over time
        # Summation of each ticker's daily contributions
        cumulative_contrib = daily_contrib.cumsum()

        # final_contrib is how much each ticker contributed to total PnL over the entire period
        # i.e. sum of daily contributions
        final_contrib = daily_contrib.sum(axis=0)  # one value per ticker

        attribution_dict = {
            'daily_contrib': daily_contrib,
            'cumulative_contrib': cumulative_contrib,
            'final_contrib': final_contrib
        }
        return attribution_dict

    def plot(self, results):
        """
        Plots:
          1) A line chart of cumulative contribution per ticker vs. time.
          2) A bar chart of the final total contribution per ticker.

        Parameters
        ----------
        attribution_dict : dict
          {
            'daily_contrib': DataFrame of daily contributions,
            'cumulative_contrib': DataFrame of cumulative contributions,
            'final_contrib': Series of total contributions
          }
        """
        attribution_dict = self.analyze(results)

        if attribution_dict is None:
            print("No attribution data to plot.")
            return

        cumulative_contrib = attribution_dict['cumulative_contrib']
        final_contrib = attribution_dict['final_contrib']

        if cumulative_contrib.empty or final_contrib.empty:
            print("Attribution data is empty. Nothing to plot.")
            return

        tickers = cumulative_contrib.columns.tolist()

        # === 1) Line chart: Cumulative contribution over time by ticker ===
        fig, ax = plt.subplots(figsize=(10, 6))
        for tkr in tickers:
            ax.plot(cumulative_contrib.index, cumulative_contrib[tkr], label=tkr)
        ax.set_title("Cumulative Contribution per Ticker")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Contribution (fraction of initial capital)")
        ax.legend(loc='best')
        ax.grid(True)
        plt.show()

        # === 2) Bar chart: Final total contribution per ticker ===
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        final_contrib.plot(kind='bar', ax=ax2, color='blue', alpha=0.7)
        ax2.set_title("Final Total Contribution by Ticker")
        ax2.set_ylabel("Total Contribution")
        ax2.set_xlabel("Ticker")
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
