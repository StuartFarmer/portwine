import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portwine.analyzers import Analyzer

"""
    Gives a clearer picture if tickers are offsetting downside and 'helpfully' uncorrelated.
"""

class DownsideCorrelationAnalyzer(Analyzer):
    """
    For each ticker T in 'tickers_returns':
      1) Identify T < 0 days.
      2) Restrict the DataFrame to just those days.
      3) For each asset (including T):
         - MeanWhenTneg => average return on T<0 days
         - CorrWithTneg => correlation with T's returns on T<0 days
      4) In the plot, each row is a different focal ticker, with 2 subplots:
         - Left: bar chart of MeanWhenTneg (T is one pastel color, others are a different pastel color)
         - Right: bar chart of CorrWithTneg (same palette usage)
      This yields a more visually appealing, non-jarring color scheme.
    """

    def __init__(self):
        pass

    def analyze(self, results):
        """
        1) Retrieve 'tickers_returns' from results.
        2) For each ticker T:
           - filter T < 0 days
           - compute the average return (MeanWhenTneg) and correlation (CorrWithTneg)
             for each asset on those T < 0 days
        3) stats_all: dict { T => DataFrame [nAssets x 2], index=assets,
                             columns=['MeanWhenTneg','CorrWithTneg'] }
        """
        tickers_df = results.get('tickers_returns')
        if tickers_df is None or tickers_df.empty:
            print("Error: 'tickers_returns' missing or empty in results.")
            return None

        if not isinstance(tickers_df.index, pd.DatetimeIndex):
            tickers_df.index = pd.to_datetime(tickers_df.index)

        all_tickers = tickers_df.columns.tolist()
        stats_all = {}

        for focal_ticker in all_tickers:
            focal_series = tickers_df[focal_ticker].dropna()
            if focal_series.empty:
                continue

            # Filter T < 0 days
            negative_mask = (focal_series < 0)
            if not negative_mask.any():
                # No negative days => store empty
                stats_all[focal_ticker] = pd.DataFrame()
                continue

            downside_df = tickers_df.loc[negative_mask]

            # For each asset, compute mean & correlation
            stats_list = []
            for asset in all_tickers:
                sub = downside_df[asset].dropna()
                mean_ret = sub.mean() if len(sub) > 0 else np.nan

                if asset == focal_ticker:
                    corr_val = 1.0  # with itself
                else:
                    corr_val = downside_df[asset].corr(downside_df[focal_ticker])

                stats_list.append({
                    'Asset': asset,
                    'MeanWhenTneg': mean_ret,
                    'CorrWithTneg': corr_val
                })

            focal_stats_df = pd.DataFrame(stats_list).set_index('Asset')
            stats_all[focal_ticker] = focal_stats_df

        return {
            'stats_all': stats_all,
            'tickers_df': tickers_df
        }

    def plot(self, results):
        """
        Produces a figure with:
          - One row per ticker T that had negative days
          - 2 columns =>
             left subplot: bar chart of MeanWhenTneg (focal T is one pastel color, others another)
             right subplot: bar chart of CorrWithTneg
        Includes a figure-level title, with a more visually appealing color palette (Set2).
        """
        analysis_dict = self.analyze(results)
        if not analysis_dict:
            print("No analysis data. Cannot plot.")
            return

        stats_all = analysis_dict['stats_all']
        if not stats_all:
            print("No stats to plot.")
            return

        # Filter out empty data
        real_tickers = [t for t, df in stats_all.items() if not df.empty]
        n_tickers = len(real_tickers)
        if n_tickers == 0:
            print("No tickers had negative days or no data. Nothing to plot.")
            return

        fig_height = 3.0 * n_tickers
        fig, axes = plt.subplots(nrows=n_tickers, ncols=2, figsize=(10, fig_height))

        # Add figure-level title
        fig.suptitle("Downside Correlation Analysis (All Tickers)", fontsize=14)

        # Prepare a pastel palette from "Set2"
        # We'll pick two distinct pastel colors from the colormap:
        palette = plt.colormaps.get_cmap("Set2")
        focal_color = palette(2)   # e.g. pastel green
        other_color = palette(0)   # e.g. pastel orange

        # If only one ticker => shape(2,) => reshape to shape(1,2)
        if n_tickers == 1:
            axes = [axes]

        for row_idx, focal_ticker in enumerate(real_tickers):
            focal_df = stats_all[focal_ticker]
            if focal_df.empty:
                continue

            ax_left = axes[row_idx][0] if n_tickers>1 else axes[0]
            ax_right = axes[row_idx][1] if n_tickers>1 else axes[1]

            # ============ MeanWhenTneg Subplot ============
            df_mean = focal_df.sort_values('MeanWhenTneg', ascending=False)
            x_vals1 = df_mean.index
            y_vals1 = df_mean['MeanWhenTneg'].values

            # We'll color the focal ticker bar in 'focal_color', others in 'other_color'.
            bar_colors_mean = []
            for asset in x_vals1:
                if asset == focal_ticker:
                    bar_colors_mean.append(focal_color)
                else:
                    bar_colors_mean.append(other_color)

            ax_left.bar(x_vals1, y_vals1, color=bar_colors_mean)
            ax_left.axhline(y=0, color='k', linewidth=1)
            ax_left.set_title(f"{focal_ticker} < 0: Mean Return by Asset")
            ax_left.tick_params(axis='x', rotation=45)
            ax_left.grid(True, alpha=0.3)

            # ============ CorrWithTneg Subplot ============
            df_corr = focal_df.sort_values('CorrWithTneg', ascending=False)
            x_vals2 = df_corr.index
            y_vals2 = df_corr['CorrWithTneg'].values

            bar_colors_corr = []
            for asset in x_vals2:
                if asset == focal_ticker:
                    bar_colors_corr.append(focal_color)
                else:
                    bar_colors_corr.append(other_color)

            ax_right.bar(x_vals2, y_vals2, color=bar_colors_corr)
            ax_right.axhline(y=0, color='k', linewidth=1)
            ax_right.set_title(f"{focal_ticker} < 0: Corr by Asset")
            ax_right.tick_params(axis='x', rotation=45)
            ax_right.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
