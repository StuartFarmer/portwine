import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portwine.analyzers import Analyzer

class DrawdownFlattenAnalyzer(Analyzer):
    """
    An analyzer that post-processes a strategy's daily returns with a one-day shift
    drawdown check. If the prior day's (yesterday's) drawdown is worse than 'max_dd',
    we flatten (scale to zero) today's returns (or partial flatten, if scale_factor>0).

    When you call 'plot(results)', it:
      1) Runs 'analyze(results)' internally to get the protected returns.
      2) Creates a single figure with three rows:
         - (Row 0) Original vs. Flattened equity curves,
         - (Row 1) The original daily drawdown curve (threshold & breach lines),
         - (Row 2) A matplotlib table comparing summary stats for Original vs. Flattened,
                   plus a “Change” column that shows the *percentage difference* for most metrics,
                   but for MaxDrawdown specifically we invert the logic so that “lower is better.”
                   We color the "Change" cell light green if it is "good" or red if it is "bad."
                   We also bold the header row and the first column.
      3) Displays the figure (no console prints).
    """

    def __init__(self, max_dd=0.2, scale_factor=0.0):
        """
        Parameters
        ----------
        max_dd : float
            Fraction of drawdown that triggers flattening (e.g. 0.2 => 20%).
        scale_factor : float
            Fraction of daily returns to keep once threshold is triggered
            (commonly 0.0 => fully flatten, or 0.5 => half exposure, etc.).
        """
        self.max_dd = max_dd
        self.scale_factor = scale_factor

    def analyze(self, results):
        """
        Applies the day-lag drawdown check to daily returns:
          - We look at yesterday's drawdown in the original equity. If it's <= -max_dd,
            we flatten (or scale) today's returns by 'scale_factor'.
          - Once the drawdown recovers above -max_dd, we revert to full exposure the next day.

        Returns a dict or None:
          {
            'original_returns' : pd.Series,
            'flattened_returns': pd.Series,
            'original_equity'  : pd.Series,
            'flattened_equity' : pd.Series,
            'drawdown_series'  : pd.Series (original daily drawdown),
            'breached_dates'   : list of Timestamps
          }
        """
        if 'strategy_returns' not in results:
            return None

        original_returns = results['strategy_returns'].copy().dropna()
        if len(original_returns) < 2:
            return None

        # 1) Original equity & daily drawdown
        original_equity = (1.0 + original_returns).cumprod()
        rolling_peak = original_equity.cummax()
        drawdown_series = (original_equity - rolling_peak) / rolling_peak  # negative or zero

        # 2) Build flattened returns (one-day shift)
        dates = original_returns.index
        flattened_returns = []
        in_drawdown_mode_yesterday = False
        dd_breach_dates = []

        # Day 0 => no prior day => normal
        flattened_returns.append(original_returns.iloc[0])

        # For day i=1..N-1 => check day i-1's drawdown
        for i in range(1, len(dates)):
            today_date = dates[i]
            today_ret = original_returns.iloc[i]
            y_dd = drawdown_series.iloc[i - 1]

            if in_drawdown_mode_yesterday:
                flattened_returns.append(self.scale_factor * today_ret)
            else:
                flattened_returns.append(today_ret)

            # Now decide if we'll be in dd_mode tomorrow
            t_dd = drawdown_series.iloc[i]
            if in_drawdown_mode_yesterday:
                # if recovered above threshold, normal tomorrow
                if t_dd > -self.max_dd:
                    in_drawdown_mode_yesterday = False
            else:
                # was not in dd_mode, check if we trigger threshold now
                if t_dd <= -self.max_dd:
                    in_drawdown_mode_yesterday = True
                    dd_breach_dates.append(today_date)

        flattened_returns = pd.Series(flattened_returns, index=dates)
        flattened_equity = (1.0 + flattened_returns).cumprod()

        return {
            'original_returns':  original_returns,
            'flattened_returns': flattened_returns,
            'original_equity':   original_equity,
            'flattened_equity':  flattened_equity,
            'drawdown_series':   drawdown_series,
            'breached_dates':    dd_breach_dates
        }

    def _compute_stats(self, daily_returns, ann_factor=252):
        """
        Compute summary statistics for a daily returns series:
          - TotalReturn
          - CAGR
          - AnnualVol
          - Sharpe (CAGR-based)
          - MaxDrawdown
        Returns a dict with these keys. If insufficient data, returns NaNs.
        """
        dr = daily_returns.dropna()
        if len(dr) < 2:
            return dict.fromkeys(["TotalReturn","CAGR","AnnualVol","Sharpe","MaxDrawdown"], np.nan)

        # 1) TotalReturn
        total_ret = (1.0 + dr).prod() - 1.0

        # 2) MaxDrawdown
        eq_curve = (1.0 + dr).cumprod()
        roll_max = eq_curve.cummax()
        dd_series = (eq_curve - roll_max) / roll_max
        max_dd = dd_series.min()  # negative or zero

        # 3) CAGR
        n_days = len(dr)
        years = n_days / ann_factor if ann_factor else np.nan
        if years > 0:
            cagr = (1.0 + total_ret)**(1.0 / years) - 1.0
        else:
            cagr = np.nan

        # 4) AnnualVol
        ann_vol = dr.std() * np.sqrt(ann_factor)

        # 5) Sharpe (CAGR-based)
        if ann_vol > 1e-9 and not np.isnan(cagr):
            sharpe = cagr / ann_vol
        else:
            sharpe = np.nan

        return {
            'TotalReturn':  total_ret,
            'CAGR':         cagr,
            'AnnualVol':    ann_vol,
            'Sharpe':       sharpe,
            'MaxDrawdown':  max_dd
        }

    def plot(self, results, ann_factor=252):
        """
        Builds a figure with 3 rows:
          Row 0 => Original vs. Flattened equity,
          Row 1 => Original daily drawdown + threshold & breach lines,
          Row 2 => Table comparing stats for Original vs. Flattened,
                   plus a 'Change' column showing percentage difference for most metrics:
                     diff = (flattened / original -1)*100
                   BUT for MaxDrawdown, we invert logic because "lower is better," so:
                     diff = (orig / flattened -1)*100
                   Then color the cell green if the 'Change' is >0 (improvement), red if <0 (worse).
        """
        ddp_results = self.analyze(results)
        if not ddp_results:
            return  # nothing to plot

        # (A) Unpack data
        orig_eq = ddp_results['original_equity']
        flat_eq = ddp_results['flattened_equity']
        dd_ser  = ddp_results['drawdown_series']
        breach_dates = ddp_results['breached_dates']

        # (B) Create figure with 3 rows
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1.5, 1])

        # Row 0 => Original vs. Flattened equity
        ax_equity = fig.add_subplot(gs[0, 0])
        ax_equity.plot(orig_eq.index, orig_eq.values, label="Original Equity")
        ax_equity.plot(flat_eq.index, flat_eq.values, label="Flattened Equity", alpha=0.8)
        ax_equity.set_title(
            f"Drawdown Flatten (One-Day Lag)\nmax_dd={self.max_dd:.0%}, scale_factor={self.scale_factor}"
        )
        ax_equity.set_xlabel("Date")
        ax_equity.set_ylabel("Cumulative Return")
        ax_equity.legend(loc="best")
        ax_equity.grid(True)

        # Row 1 => daily drawdown
        ax_dd = fig.add_subplot(gs[1, 0])
        ax_dd.plot(dd_ser.index, dd_ser.values, label="Original Drawdown", color='blue')
        ax_dd.axhline(-self.max_dd, color='red', linestyle='--', alpha=0.7,
                      label=f"Threshold {self.max_dd:.0%}")
        for d in breach_dates:
            ax_dd.axvline(d, color='orange', linestyle='--', alpha=0.5)
        ax_dd.set_title("Daily Drawdown (Original) & Breach Points")
        ax_dd.set_xlabel("Date")
        ax_dd.set_ylabel("Drawdown (fraction)")
        ax_dd.legend(loc="best")
        ax_dd.grid(True)

        # Row 2 => Stats table
        ax_table = fig.add_subplot(gs[2, 0])
        ax_table.axis('off')

        # (C) Compute stats for Original vs. Flattened
        stats_orig = self._compute_stats(ddp_results['original_returns'], ann_factor=ann_factor)
        stats_flat = self._compute_stats(ddp_results['flattened_returns'], ann_factor=ann_factor)

        metrics = ["TotalReturn", "CAGR", "AnnualVol", "Sharpe", "MaxDrawdown"]
        table_header = ["Metric", "Original", "Flattened", "Change"]
        table_data = [table_header]

        # We'll store numeric differences in a dict so we can color the cells
        differences = {}

        def pct_fmt(x):
            return f"{x*100:.2f}%" if pd.notnull(x) else "NaN"

        def ratio_fmt(x):
            return f"{x:.2f}" if pd.notnull(x) else "NaN"

        def pct_diff_for_metric(m, val_orig, val_flat):
            """
            For most metrics, we want: diff = (val_flat / val_orig -1)*100
            But for MaxDrawdown, we invert logic: diff = (val_orig / val_flat -1)*100
            because a lower (more negative) drawdown is actually worse.
            (We interpret negative as less negative => improvement.)
            If either val_orig or val_flat is near zero, or NaN, we return NaN.
            """
            if pd.isnull(val_orig) or pd.isnull(val_flat):
                return np.nan

            # If either original or flattened is extremely small in absolute value,
            # we can't do a safe ratio => NaN
            if abs(val_orig) < 1e-12 or abs(val_flat) < 1e-12:
                return np.nan

            if m == "MaxDrawdown":
                # ratio = (orig / flat -1)*100
                # because "lower absolute drawdown" => improvement
                # note: both are negative or zero, so we must handle that carefully
                # e.g. orig = -0.05, flat=-0.06 => ratio = (-0.05 / -0.06 -1)*100 => +16.7 => red
                # Actually "improvement" means we want the flattened to be less negative, i.e. -0.04 is better than -0.05
                # We define "improvement" as having a more negative number is actually worse, so if flattened is -0.04 vs orig -0.05 => flatten is better => ratio>0 => green
                # => ratio = (orig / flat -1)*100
                return ((val_orig / val_flat) - 1.0)*100.0
            else:
                # ratio = (val_flat / val_orig -1)*100
                return ((val_flat / val_orig) -1.0)*100.0

        # Now build the rows
        row_index = 1  # b/c row 0 is the header
        for m in metrics:
            val_o = stats_orig[m]
            val_f = stats_flat[m]

            # Format Original & Flattened
            if m in ["TotalReturn","CAGR","AnnualVol","MaxDrawdown"]:
                s_orig = pct_fmt(val_o)
                s_flat = pct_fmt(val_f)
            else:  # Sharpe
                s_orig = ratio_fmt(val_o)
                s_flat = ratio_fmt(val_f)

            # Compute difference ratio
            diff_val = pct_diff_for_metric(m, val_o, val_f)
            if pd.notnull(diff_val):
                diff_str = f"{diff_val:.2f}%"
            else:
                diff_str = "NaN"

            table_data.append([m, s_orig, s_flat, diff_str])
            differences[row_index] = (m, diff_val)
            row_index += 1

        # (D) Create the table
        the_table = ax_table.table(
            cellText=table_data,
            loc='center',
            cellLoc='center'
        )
        the_table.set_fontsize(10)
        the_table.scale(1.2, 1.2)

        # (E) Bold the header row (row=0) & the first column (col=0)
        n_rows = len(table_data)
        n_cols = len(table_data[0])

        # Bold the header row
        for col in range(n_cols):
            the_table[(0, col)].set_text_props(weight='bold')

        # Bold the first column
        for row in range(n_rows):
            the_table[(row, 0)].set_text_props(weight='bold')

        # (F) Color the "Change" column (col=3) based on 'differences'
        for row_idx in range(1, n_rows):  # data rows
            metric_name, diff_val = differences.get(row_idx, ("", np.nan))
            cell = the_table[(row_idx, 3)]
            if pd.isnull(diff_val):
                continue

            # If diff_val>0 => improvement => green, else red
            # For MaxDrawdown, we inverted logic above, so a "positive" ratio => means better
            if diff_val > 0:
                cell.set_facecolor("#d6f5d6")  # light green
            elif diff_val < 0:
                cell.set_facecolor("#f7d6d6")  # light red
            # if exactly 0 => no color

        fig.tight_layout()
        plt.show()
