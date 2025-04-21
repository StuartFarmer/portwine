import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portwine.analyzers import Analyzer

class PositionSizingAnalyzer(Analyzer):
    """
    A single class that:
      - Sweeps position sizes from position_size_start up to max_position_size in steps,
      - Computes performance metrics (CAGR, Sharpe, Sortino, etc.) at each step,
      - Selects the best position size under the chosen objective,
      - Produces a single combined plot including:
         1) Top: Equity curve (best levered) vs. benchmark
         2) Middle row: position size vs. objective, position size vs. max drawdown
         3) Bottom row: summary table comparing unlevered vs. best levered stats
    """

    def __init__(self):
        """No parameters stored at init.  All are passed to analyze(...) or plot(...)."""
        pass

    def _compute_stats(self, daily_returns, ann_factor=252):
        """
        Compute performance metrics for a given daily returns Series:
          - TotalReturn, CAGR, AnnualVol, Sharpe, MaxDrawdown, Sortino, Calmar, UlcerIndex, UPI.
        """
        dr = daily_returns.dropna()
        if len(dr) < 2:
            return {
                'TotalReturn':  np.nan,
                'CAGR':         np.nan,
                'AnnualVol':    np.nan,
                'Sharpe':       np.nan,
                'MaxDrawdown':  np.nan,
                'Sortino':      np.nan,
                'Calmar':       np.nan,
                'UlcerIndex':   np.nan,
                'UPI':          np.nan
            }

        # 1) Total Return
        total_ret = (1.0 + dr).prod() - 1.0

        # 2) CAGR
        n_days = len(dr)
        years = n_days / ann_factor
        if years > 0 and (1.0 + total_ret) > 1e-12:
            try:
                cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0
            except:
                cagr = np.nan
        else:
            cagr = np.nan

        # 3) AnnualVol
        ann_vol = dr.std() * np.sqrt(ann_factor)

        # 4) Sharpe (CAGR / annual_vol)
        if ann_vol > 1e-9 and not np.isnan(cagr):
            sharpe = cagr / ann_vol
        else:
            sharpe = np.nan

        # 5) Equity curve for drawdown calculations
        equity_curve = (1.0 + dr).cumprod()
        rolling_max = equity_curve.cummax()
        drawdown_series = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown_series.min()  # negative or zero

        # 6) Sortino
        negative_returns = dr[dr < 0]
        if len(negative_returns) > 0:
            downside_vol = negative_returns.std() * np.sqrt(ann_factor)
            if downside_vol > 1e-9 and not np.isnan(cagr):
                sortino = cagr / downside_vol
            else:
                sortino = np.nan
        else:
            sortino = np.nan

        # 7) Calmar (CAGR / abs(MaxDrawdown))
        calmar = np.nan
        if not np.isnan(cagr):
            dd_abs = abs(max_drawdown)
            if dd_abs > 1e-9:
                calmar = cagr / dd_abs

        # 8) Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown_series**2))

        # 9) UPI (CAGR / UlcerIndex)
        upi = np.nan
        if ulcer_index > 1e-9 and not np.isnan(cagr):
            upi = cagr / ulcer_index

        return {
            'TotalReturn':  total_ret,
            'CAGR':         cagr,
            'AnnualVol':    ann_vol,
            'Sharpe':       sharpe,
            'MaxDrawdown':  max_drawdown,
            'Sortino':      sortino,
            'Calmar':       calmar,
            'UlcerIndex':   ulcer_index,
            'UPI':          upi
        }

    def analyze(self,
                results,
                position_size_start=0.1,
                position_size_step=0.1,
                max_position_size=10.0,
                objective='sharpe',
                stop_drawdown_threshold=1.0,
                ann_factor=252):
        """
        Sweep position size from position_size_start to max_position_size in increments,
        compute stats, and pick the best under the chosen objective.

        Returns
        -------
        {
           'best_size': float,
           'best_stats': dict,
           'all_stats': pd.DataFrame  # indexed by position_size
        }
        """
        if 'strategy_returns' not in results:
            print("No 'strategy_returns' found in results.")
            return None

        strategy_returns = results['strategy_returns'].dropna()
        if strategy_returns.empty:
            print("Strategy returns is empty.")
            return None

        records = []
        current_size = position_size_start
        objective = objective.lower().strip()

        while current_size <= max_position_size + 1e-9:
            scaled_ret = strategy_returns * current_size
            stats = self._compute_stats(scaled_ret, ann_factor)
            row = {
                'position_size': current_size,
                **stats
            }
            records.append(row)

            # If max drawdown <= -stop_drawdown_threshold => stop
            if stats['MaxDrawdown'] <= -stop_drawdown_threshold:
                break

            current_size += position_size_step

        df_stats = pd.DataFrame(records).set_index('position_size')
        if df_stats.empty:
            print("No valid position sizes tested.")
            return None

        # Identify best row
        objective_col_map = {
            'sharpe':       'Sharpe',
            'sortino':      'Sortino',
            'cagr':         'CAGR',
            'max_drawdown': 'MaxDrawdown',
            'calmar':       'Calmar',
            'upi':          'UPI'
        }
        col_name = objective_col_map.get(objective, 'Sharpe')

        if objective == 'max_drawdown':
            # We want the "least negative" => idxmax
            best_idx = df_stats[col_name].idxmax()
        else:
            best_idx = df_stats[col_name].idxmax()

        best_size = float(best_idx)
        best_stats = df_stats.loc[best_size].to_dict()

        return {
            'best_size':  best_size,
            'best_stats': best_stats,
            'all_stats':  df_stats
        }

    def plot(self,
             results,
             position_size_start=0.1,
             position_size_step=0.1,
             max_position_size=10.0,
             objective='sharpe',
             stop_drawdown_threshold=1.0,
             ann_factor=252):
        """
        Runs analyze(...) with the user-provided parameters, then
        produces a single figure with:
         - (Top) Best equity vs. benchmark
         - (Middle-Left) position_size vs. objective
         - (Middle-Right) position_size vs. max drawdown
         - (Bottom) summary table comparing unlevered vs best levered stats
        """

        # 1) Analyze
        analysis_results = self.analyze(
            results=results,
            position_size_start=position_size_start,
            position_size_step=position_size_step,
            max_position_size=max_position_size,
            objective=objective,
            stop_drawdown_threshold=stop_drawdown_threshold,
            ann_factor=ann_factor
        )
        if not analysis_results:
            print("No analysis results; cannot plot.")
            return

        if 'strategy_returns' not in results:
            print("No 'strategy_returns' in results; cannot plot.")
            return

        df_stats = analysis_results.get('all_stats')
        if df_stats is None or df_stats.empty:
            print("No stats DataFrame; nothing to plot.")
            return

        # 2) Prepare data for top equity plot
        best_size = analysis_results['best_size']
        strategy_returns = results['strategy_returns'].dropna()
        scaled_returns = strategy_returns * best_size
        equity_curve = (1.0 + scaled_returns).cumprod()

        # Possibly benchmark
        benchmark = results.get('benchmark_returns')
        bm_ec = None
        if benchmark is not None and not benchmark.empty:
            bm_ec = (1.0 + benchmark.fillna(0.0)).cumprod()

        # 3) Layout with 4 rows: top for equity, 2 for mid subplots, 1 for table
        # We'll reduce the bottom row's ratio so the table doesn't get huge space
        fig = plt.figure(figsize=(12, 10))
        grid_ratios = [2,1,1,0.8]  # less space for table row => 0.8
        gs = fig.add_gridspec(nrows=4, ncols=2, height_ratios=grid_ratios)

        # (A) Top row => equity
        ax_equity = fig.add_subplot(gs[0, :])
        ax_equity.plot(equity_curve.index, equity_curve.values,
                       label=f"Strategy x{best_size:.2f}")
        if bm_ec is not None:
            ax_equity.plot(bm_ec.index, bm_ec.values, label="Benchmark", alpha=0.7)

        ax_equity.set_title("Position Sizing Analysis: Best Strategy vs. Benchmark")
        ax_equity.set_xlabel("Date")
        ax_equity.set_ylabel("Cumulative Return")
        ax_equity.legend(loc="best")
        ax_equity.grid(True)

        # (B) Middle row => objective, dd
        ax_obj = fig.add_subplot(gs[1, 0])
        ax_dd = fig.add_subplot(gs[1, 1])

        # objective col
        objective_col_map = {
            'sharpe':       'Sharpe',
            'sortino':      'Sortino',
            'cagr':         'CAGR',
            'max_drawdown': 'MaxDrawdown',
            'calmar':       'Calmar',
            'upi':          'UPI'
        }
        col_name = objective_col_map.get(objective, 'Sharpe')

        ax_obj.plot(df_stats.index, df_stats[col_name], marker='o', linestyle='-')
        ax_obj.set_title(f"Position Size vs. {col_name}")
        ax_obj.set_xlabel("Position Size")
        ax_obj.set_ylabel(col_name)
        ax_obj.grid(True)

        ax_dd.plot(df_stats.index, df_stats['MaxDrawdown'], marker='o', linestyle='-', color='red')
        ax_dd.set_title("Position Size vs. Max Drawdown")
        ax_dd.set_xlabel("Position Size")
        ax_dd.set_ylabel("Max Drawdown")
        ax_dd.grid(True)

        # (C) We'll do an additional subplot row for any expansions, or skip it.
        # Let's put the table in row=3 (the 4th row).
        ax_table = fig.add_subplot(gs[2:, :])  # spans row=2..3 => 2 rows
        ax_table.axis('off')

        # 4) Build the table comparing unlevered vs. best levered
        unlevered_stats = self._compute_stats(strategy_returns, ann_factor)
        levered_stats = self._compute_stats(scaled_returns, ann_factor)

        show_metrics = [
            ("TotalReturn",   "Total Return",  True),
            ("CAGR",          "CAGR",          True),
            ("Sharpe",        "Sharpe",        False),
            ("Sortino",       "Sortino",       False),
            ("MaxDrawdown",   "Max Drawdown",  True),
            ("Calmar",        "Calmar",        False),
            ("UlcerIndex",    "Ulcer Index",   False),
            ("UPI",           "UPI",           False)
        ]
        table_data = []
        col_labels = ["Metric", "Unlevered (1×)", f"Levered (x{best_size:.2f})", "Diff %"]

        for (key, label, is_percent) in show_metrics:
            unlev_val = unlevered_stats[key]
            lev_val   = levered_stats[key]

            if is_percent:
                unlev_str = f"{unlev_val:.2%}" if pd.notnull(unlev_val) else "NaN"
                lev_str   = f"{lev_val:.2%}"   if pd.notnull(lev_val)   else "NaN"
                # Diff ratio
                if pd.notnull(unlev_val) and abs(unlev_val) > 1e-12:
                    diff_ratio = (lev_val - unlev_val) / abs(unlev_val)
                else:
                    diff_ratio = np.nan
                diff_str = f"{diff_ratio:.2%}" if pd.notnull(diff_ratio) else "NaN"
            else:
                unlev_str = f"{unlev_val:.3f}" if pd.notnull(unlev_val) else "NaN"
                lev_str   = f"{lev_val:.3f}"   if pd.notnull(lev_val)   else "NaN"
                if pd.notnull(unlev_val) and abs(unlev_val) > 1e-12:
                    diff_ratio = (lev_val - unlev_val) / abs(unlev_val)
                else:
                    diff_ratio = np.nan
                diff_str = f"{diff_ratio:.2%}" if pd.notnull(diff_ratio) else "NaN"

            table_data.append([label, unlev_str, lev_str, diff_str])

        tbl = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            # loc='center'
            loc='upper center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.2)

        # color the last column if parseable
        n_data_rows = len(table_data)
        diff_col_idx = len(col_labels) - 1
        for row_i in range(n_data_rows):
            cell = tbl[(row_i+1, diff_col_idx)]  # data rows start at 1
            raw_str = table_data[row_i][3].strip('%')
            try:
                val = float(raw_str)/100
                if val > 0:
                    cell.set_facecolor('#d8f3dc')  # light green
                elif val < 0:
                    cell.set_facecolor('#ffccd5')  # light red
            except:
                pass

        # Tweak layout: we've already used GridSpec with row ratio,
        # but let's further reduce vertical gaps:
        fig.tight_layout()
        # or a small manual subplots_adjust
        plt.subplots_adjust(hspace=0.3)

        plt.show()
