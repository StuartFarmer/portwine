import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portwine.analyzers import Analyzer

class RiskContributionAnalyzer(Analyzer):
    """
    Analyzes each ticker's contribution to overall portfolio risk (variance)
    under a rolling covariance approach. Useful for observing whether
    the portfolio approximates a 'risk parity' distribution or if some
    tickers dominate risk.

    Requirements:
      - signals_df: DataFrame of daily portfolio weights, [dates x tickers]
      - tickers_returns: DataFrame of daily returns for each ticker, [dates x tickers]
      - We compute a rolling covariance matrix using the previous 'cov_window' days
        for each date. Then for that date:
           portfolio_variance = w^T * Sigma * w
           ARC_i = w_i * (Sigma * w)_i      (absolute risk contribution)
           PRC_i = ARC_i / sum_j( ARC_j )   (percentage risk contribution)
      - Summation of ARC_i across all tickers = portfolio variance
      - Summation of PRC_i across all tickers = 1.0 (100%)
    """

    def __init__(self, cov_window=60):
        """
        Parameters
        ----------
        cov_window : int
            Number of past days used to estimate the rolling covariance matrix.
        """
        self.cov_window = cov_window

    def analyze(self, results):
        """
        Compute rolling risk contributions (absolute and percentage) for each ticker.

        Parameters
        ----------
        results : dict
            {
              'signals_df': DataFrame of daily portfolio weights,
              'tickers_returns': DataFrame of daily ticker returns,
              ...
            }

        Returns
        -------
        risk_dict : dict
            {
              'abs_risk_contrib': DataFrame of shape [dates x tickers],
                  where entry (t, i) is ARC_i(t),
              'pct_risk_contrib': DataFrame of shape [dates x tickers],
                  where entry (t, i) is PRC_i(t),
              'portfolio_vol': Series of shape [dates,],
                  daily portfolio volatility (annualized if desired),
              'portfolio_variance': Series of shape [dates,],
                  daily portfolio variance,
            }
        """
        signals_df = results.get('signals_df')
        tickers_returns = results.get('tickers_returns')
        if signals_df is None or tickers_returns is None:
            print("Error: signals_df or tickers_returns missing.")
            return None

        # Align columns & rows
        # We want signals_df and tickers_returns to have the same tickers & date index
        signals_df, tickers_returns = signals_df.align(
            tickers_returns, join='inner', axis=1
        )
        signals_df, tickers_returns = signals_df.align(
            tickers_returns, join='inner', axis=0
        )

        # We'll create DataFrames to store absolute & percentage risk contributions
        abs_risk_contrib = pd.DataFrame(index=signals_df.index, columns=signals_df.columns, dtype=float)
        pct_risk_contrib = pd.DataFrame(index=signals_df.index, columns=signals_df.columns, dtype=float)
        portfolio_variance = pd.Series(index=signals_df.index, dtype=float)
        portfolio_vol = pd.Series(index=signals_df.index, dtype=float)

        tickers = signals_df.columns.tolist()

        # We'll iterate through the dates, but skip the first 'cov_window' because we
        # need enough lookback to compute the covariance matrix.
        for i in range(self.cov_window, len(signals_df)):
            current_date = signals_df.index[i]
            # We'll look back cov_window days (excluding current_date).
            start_idx = i - self.cov_window
            window_slice = tickers_returns.iloc[start_idx:i]

            # Estimate covariance matrix from the last 'cov_window' days of returns
            Sigma = window_slice.cov()  # shape [N x N] for N tickers

            # Get the weight vector at current_date
            w = signals_df.loc[current_date].values  # shape [N,]

            # If there are any NaNs in weights, fill with 0
            w = np.nan_to_num(w)

            # portfolio variance = w^T * Sigma * w
            port_var = float(np.dot(w, Sigma.dot(w)))
            portfolio_variance.loc[current_date] = port_var

            # absolute risk contribution ARC_i = w_i * (Sigma*w)_i
            # We'll do it in vectorized form:
            Sigma_w = Sigma.dot(w)  # shape [N,]
            arc = w * Sigma_w  # shape [N,]

            abs_risk_contrib.loc[current_date] = arc

            # percentage risk contribution prc_i = arc_i / sum_j(arc_j)
            sum_arc = arc.sum()
            if abs(sum_arc) < 1e-12:
                # If portfolio variance is ~0, set PRC to 0
                pct_risk_contrib.loc[current_date] = 0.0
            else:
                pct_risk_contrib.loc[current_date] = arc / sum_arc

            # If you'd like daily *annualized* volatility, do something like:
            # daily vol = sqrt(port_var), annual vol = sqrt(port_var * 252)
            # For simplicity, let's keep it daily:
            daily_vol = np.sqrt(port_var)
            portfolio_vol.loc[current_date] = daily_vol

        risk_dict = {
            'abs_risk_contrib': abs_risk_contrib,
            'pct_risk_contrib': pct_risk_contrib,
            'portfolio_variance': portfolio_variance,
            'portfolio_vol': portfolio_vol
        }
        return risk_dict

    def plot(self,
             results,
             plot_type='pct',
             snapshot='last_valid',
             rolling_window=30,
             title="Risk Contribution Over Time"):
        """
        Plot either absolute or percentage risk contributions over time
        as a stacked area chart, plus a bar chart "snapshot" at a chosen date or
        averaged over a chosen window.

        Parameters
        ----------
        risk_dict : dict
            Output of self.analyze(), with keys:
                'abs_risk_contrib': DataFrame [dates x tickers]
                'pct_risk_contrib': DataFrame [dates x tickers]
                'portfolio_variance': Series [dates,]
                'portfolio_vol': Series [dates,]
        plot_type : str
            Either 'pct' (percentage) or 'abs' (absolute) risk contribution
        snapshot : str or pd.Timestamp or 'last_valid' or None
            Controls which date(s) to show in the final bar chart:
              - 'last_valid' (default): use the last date with non-NaN data
              - a date string like "2021-12-31" or a pd.Timestamp
              - None => skip the snapshot bar chart
        average_window : int or None
            If provided (e.g. 30), we compute an average of the last N days
            leading up to `snapshot` rather than a single date.
        title : str
            Chart title for the stacked area chart.
        """

        risk_dict = self.analyze(results)

        if risk_dict is None:
            print("No risk data to plot.")
            return

        if plot_type.lower() == 'pct':
            df_risk_contrib = risk_dict['pct_risk_contrib']
            y_label = "Percentage Risk Contribution"
        else:
            df_risk_contrib = risk_dict['abs_risk_contrib']
            y_label = "Absolute Risk Contribution"

        if df_risk_contrib is None or df_risk_contrib.empty:
            print("Risk contribution data is empty.")
            return

        # 1) Stacked area chart for entire timeseries
        fig, ax = plt.subplots(figsize=(10, 6))
        df_risk_contrib.plot.area(ax=ax, linewidth=0, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Date")
        ax.grid(True)
        # plt.tight_layout()
        # plt.show()

        # 2) If snapshot is None, skip the bar chart
        if snapshot is None:
            return

        # We'll define a helper function to get a valid end_date if needed
        def get_valid_end_date(df, snap):
            """
            If 'snap' == 'last_valid', return the last non-NaN date in df.
            If snap is a string or timestamp, parse and find the closest date <= snap.
            Otherwise, return None if invalid.
            """
            if snap == 'last_valid':
                valid_df = df.dropna(how='all')
                if valid_df.empty:
                    return None
                return valid_df.index[-1]  # a Timestamp
            else:
                # attempt parse
                snap_date = pd.to_datetime(snap)
                # find the closest date in df_risk_contrib <= snap_date
                valid_dates = df.index[df.index <= snap_date]
                if len(valid_dates) == 0:
                    return None
                return valid_dates[-1]

        if rolling_window and rolling_window > 1:
            # We want an average over the last N days up to 'snapshot'.
            end_date = get_valid_end_date(df_risk_contrib, snapshot)
            if end_date is None:
                print(f"No valid dates found for snapshot='{snapshot}'. Skipping bar chart.")
                return

            all_dates = df_risk_contrib.index
            end_idx = all_dates.get_loc(end_date)
            start_idx = max(0, end_idx - (rolling_window - 1))
            window_dates = all_dates[start_idx:end_idx + 1]

            final_vec = df_risk_contrib.loc[window_dates].mean(axis=0)
            bar_title = f"Average {y_label} over last {rolling_window} days (ending {end_date.date()})"

        else:
            # Single date snapshot
            end_date = get_valid_end_date(df_risk_contrib, snapshot)
            if end_date is None:
                print(f"No valid dates found for snapshot='{snapshot}'. Skipping bar chart.")
                return

            final_vec = df_risk_contrib.loc[end_date]
            bar_title = f"{y_label} at {end_date.date()}"

        # We simply use the rolling mean. This will produce some NaNs for the first
        # rolling_window - 1 points. We can drop them or let them remain.
        df_rolling = df_risk_contrib.rolling(rolling_window).mean()

        # If you prefer to skip the leading NaNs, you can do df_rolling = df_rolling.dropna(...)
        # For demonstration, let's keep them.

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        df_rolling.plot.area(ax=ax2, linewidth=0, alpha=0.8)
        ax2.set_title(f"Rolling {rolling_window}-day Mean of {y_label}")
        ax2.set_ylabel(y_label)
        ax2.set_xlabel("Date")
        ax2.grid(True)

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        final_vec.plot(kind='bar', ax=ax3, color='blue', alpha=0.7)
        ax3.set_title(bar_title)
        ax3.set_ylabel(y_label)
        ax3.set_xlabel("Tickers")
        ax3.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()
