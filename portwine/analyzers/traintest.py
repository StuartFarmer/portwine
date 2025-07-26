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
        Now supports 'split' as either a float (fraction) or a date string ('YYYY-MM-DD').
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

        # Support split as float (fraction) or date string
        split_idx = None
        split_date = None
        if isinstance(split, float):
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
        elif isinstance(split, str):
            try:
                split_date = pd.to_datetime(split)
            except Exception as e:
                print(f"Invalid split date string: {split}. Error: {e}")
                return
            # Find the last date in all_dates that is <= split_date
            valid_dates = all_dates[all_dates <= split_date]
            if len(valid_dates) == 0:
                print(f"Split date {split} is before the start of the data.")
                return
            split_idx = len(valid_dates)
            if split_idx < 1:
                print("Train set is empty. Increase 'split'.")
                return
            if split_idx >= n:
                print("Test set is empty. Decrease 'split'.")
                return
            train_dates = all_dates[:split_idx]
            test_dates = all_dates[split_idx:]
            split_date = train_dates[-1]
        else:
            print("'split' must be a float (fraction) or a date string (YYYY-MM-DD).")
            return

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
