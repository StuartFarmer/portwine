import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colormaps
from portwine.analyzers.base import Analyzer

DEFAULT_SHOCK_PERIODS = {
    "Dot-Com Bubble Crash": ("2000-03-24", "2002-10-09"),
    "9/11 Aftermath": ("2001-09-10", "2001-09-21"),
    "Global Financial Crisis": ("2007-10-09", "2009-03-09"),
    "European Debt Crisis": ("2011-05-02", "2011-10-04"),
    "2015-2016 Correction": ("2015-08-10", "2016-02-11"),
    "Volmageddon": ("2018-01-26", "2018-02-09"),
    "COVID Crash": ("2020-02-20", "2020-03-23"),
    "2022 Rate Hike Selloff": ("2022-01-03", "2022-06-16")
}


class ShockAnalyzer(Analyzer):
    """
    Dynamically lays out the figure so we don't have large empty spaces
    if only a few stress periods remain.
    """

    def __init__(self, stress_periods=DEFAULT_SHOCK_PERIODS):
        """
        Parameters
        ----------
        stress_periods : dict or None
            {
              "Dot-Com Bubble": ("2000-03-24", "2002-10-09"),
              ...
            }
            If None/empty, no stress shading or subplots/table.
        """
        self.stress_periods = stress_periods if stress_periods else {}

    def analyze(self, results):
        strat_rets = results.get('strategy_returns')
        if strat_rets is None or strat_rets.empty:
            print("Error: 'strategy_returns' missing or empty.")
            return None

        # Strategy
        strat_eq = (1.0 + strat_rets).cumprod()
        strat_dd = self._compute_drawdown(strat_eq)
        strat_mdd = strat_dd.min()
        strategy_data = {
            'equity_curve': strat_eq,
            'drawdown_series': strat_dd,
            'max_drawdown': strat_mdd
        }

        # Benchmark
        bm_rets = results.get('benchmark_returns')
        benchmark_data = None
        if bm_rets is not None and not bm_rets.empty:
            if isinstance(bm_rets, pd.DataFrame) and bm_rets.shape[1] == 1:
                bm_rets = bm_rets.iloc[:, 0]
            bm_eq = (1.0 + bm_rets).cumprod()
            bm_dd = self._compute_drawdown(bm_eq)
            bm_mdd = bm_dd.min()

            benchmark_data = {
                'equity_curve': bm_eq,
                'drawdown_series': bm_dd,
                'max_drawdown': bm_mdd
            }

        if not self.stress_periods:
            return {
                'strategy': strategy_data,
                'benchmark': benchmark_data,
                'period_slices': {},
                'stress_df': None
            }

        # Build slices + summary stats
        period_slices = {}
        stress_records = []

        s_idx = strat_eq.index
        b_idx = benchmark_data['equity_curve'].index if benchmark_data else None

        for pname, (start_str, end_str) in self.stress_periods.items():
            start_dt = pd.to_datetime(start_str)
            end_dt = pd.to_datetime(end_str)

            # MODIFIED: Find overlapping data rather than requiring full coverage
            actual_start = max(s_idx.min(), start_dt)
            actual_end = min(s_idx.max(), end_dt)

            # Check if we have at least some data in this period
            if actual_end < actual_start or actual_start > end_dt or actual_end < start_dt:
                # No overlap with the data we have
                continue

            # Get the data slice for this (possibly partial) crisis period
            s_sub = strat_eq.loc[(strat_eq.index >= actual_start) & (strat_eq.index <= actual_end)]
            if len(s_sub) < 2:
                continue

            # Add flag to indicate if this is a partial period
            is_partial = (actual_start > start_dt) or (actual_end < end_dt)
            partial_label = " (Partial)" if is_partial else ""

            sub_dd = self._compute_drawdown(s_sub)
            sub_ret = (s_sub.iloc[-1] / s_sub.iloc[0]) - 1.0
            sub_mdd = sub_dd.min()

            # Possibly do benchmark
            bm_ret, bm_mdd_val = np.nan, np.nan
            b_sub = None
            if benchmark_data:
                bm_start = max(b_idx.min(), start_dt)
                bm_end = min(b_idx.max(), end_dt)

                if bm_end >= bm_start:  # If there's some overlap
                    b_eq_full = benchmark_data['equity_curve']
                    b_sub = b_eq_full.loc[(b_eq_full.index >= bm_start) & (b_eq_full.index <= bm_end)]
                    if len(b_sub) > 1:
                        bm_dd_sub = self._compute_drawdown(b_sub)
                        bm_mdd_val = bm_dd_sub.min()
                        bm_ret = (b_sub.iloc[-1] / b_sub.iloc[0]) - 1.0

            display_name = pname + partial_label
            period_slices[display_name] = {'s_sub': s_sub, 'b_sub': b_sub}
            stress_records.append({
                'Period': display_name,
                'Start': actual_start,
                'End': actual_end,
                'Strategy_TotalRet': sub_ret,
                'Strategy_MaxDD': sub_mdd,
                'Benchmark_TotalRet': bm_ret,
                'Benchmark_MaxDD': bm_mdd_val,
                'IsPartial': is_partial,
                'OriginalStart': start_dt,
                'OriginalEnd': end_dt
            })

        if not stress_records:
            stress_df = None
        else:
            stress_df = pd.DataFrame(stress_records).set_index('Period')

        return {
            'strategy': strategy_data,
            'benchmark': benchmark_data,
            'period_slices': period_slices,
            'stress_df': stress_df
        }

    def plot(self, results):
        adict = self.analyze(results)
        if adict is None:
            print("No analysis data to plot.")
            return

        strat = adict['strategy']
        bench = adict['benchmark']
        period_slices = adict['period_slices']
        stress_df = adict['stress_df']

        # Data
        strat_eq = strat['equity_curve']
        strat_dd = strat['drawdown_series'] * 100.0
        has_bm = (bench is not None)
        bm_eq = bench['equity_curve'] * 1 if has_bm else None
        bm_dd = (bench['drawdown_series'] * 100.0) if has_bm else None

        # Which events remain
        events_list = list(period_slices.items())  # [(name, {s_sub, b_sub}), ...]
        # Limit to 8 if you want
        displayed_events = min(8, len(events_list))
        events_list = events_list[:displayed_events]

        # If none, we just do eq+dd
        # # subplots_rows = math.ceil(displayed_events / 2)
        subplots_rows = int(math.ceil(displayed_events / 2.0))
        # total rows = 2 (eq+dd) + subplots_rows + 1 (table)
        # If no events at all, skip the subplots entirely but still do 1 row for the table
        # so the table can show "no stress events"? We'll do a check if stress_df is None
        # or empty

        # If no events, subplots_rows=0 => total_rows=2+0+1=3
        # But if stress_df is also None, we might skip the table row?
        have_table = (stress_df is not None and not stress_df.empty)

        total_rows = 2 + subplots_rows
        if have_table:
            total_rows += 1

        # Create figure
        fig = plt.figure(figsize=(14, 4 * (total_rows)), constrained_layout=True)
        gs = GridSpec(nrows=total_rows, ncols=2, figure=fig)
        cmap = colormaps.get_cmap("tab10")

        # Row 0: equity
        ax_eq = fig.add_subplot(gs[0, :])
        ax_eq.plot(strat_eq.index, strat_eq.values, label="Strategy")
        if has_bm:
            ax_eq.plot(bm_eq.index, bm_eq.values, label="Benchmark", alpha=0.7)
        ax_eq.set_title("Economic Stress Equity Curve")
        ax_eq.legend(loc='best')
        ax_eq.tick_params(axis='x', rotation=45)

        # shading only for those events we actually kept
        if stress_df is not None:
            for i, pname in enumerate(stress_df.index):
                row_data = stress_df.loc[pname]

                # MODIFIED: Instead of stacking, use conditional approach
                color = cmap(i % 10)

                if 'OriginalStart' in row_data and 'OriginalEnd' in row_data and pd.notnull(row_data['OriginalStart']):
                    start_dt = row_data['OriginalStart']
                    end_dt = row_data['OriginalEnd']
                    actual_start = row_data['Start']
                    actual_end = row_data['End']

                    # For partial periods, don't stack spans
                    if row_data.get('IsPartial', False):
                        # Add light shading only for parts we don't have data for
                        if start_dt < actual_start:
                            ax_eq.axvspan(start_dt, actual_start, color=color, alpha=0.05)
                        if actual_end < end_dt:
                            ax_eq.axvspan(actual_end, end_dt, color=color, alpha=0.05)

                        # Add normal shading only for the part we have data for
                        ax_eq.axvspan(actual_start, actual_end, color=color, alpha=0.1)
                    else:
                        # For complete periods, just add normal shading
                        ax_eq.axvspan(start_dt, end_dt, color=color, alpha=0.1)
                else:
                    # Fallback to actual dates if original not available
                    start_dt = row_data['Start']
                    end_dt = row_data['End']
                    ax_eq.axvspan(start_dt, end_dt, color=color, alpha=0.1)

            # Add legend entries for each unique event
            handles, labels = ax_eq.get_legend_handles_labels()
            event_handles = []
            event_labels = []

            for i, pname in enumerate(stress_df.index):
                color = cmap(i % 10)
                # Remove " (Partial)" from legend
                clean_name = pname.replace(" (Partial)", "")

                # Use consistent alpha for legend that matches what's shown for complete periods
                event_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.1))
                event_labels.append(clean_name)

            all_handles = handles + event_handles
            all_labels = labels + event_labels
            ax_eq.legend(all_handles, all_labels, loc='best')

        # Row 1: dd
        ax_dd = fig.add_subplot(gs[1, :])
        ax_dd.plot(strat_dd.index, strat_dd.values, label="Strategy DD (%)")
        if has_bm:
            ax_dd.plot(bm_dd.index, bm_dd.values, label="Benchmark DD (%)", alpha=0.7)
        ax_dd.set_title("Drawdown (%)")
        ax_dd.legend(loc='best')
        ax_dd.tick_params(axis='x', rotation=45)

        if stress_df is not None:
            for i, pname in enumerate(stress_df.index):
                row_data = stress_df.loc[pname]

                # MODIFIED: Instead of stacking, use conditional approach
                color = cmap(i % 10)

                if 'OriginalStart' in row_data and 'OriginalEnd' in row_data and pd.notnull(row_data['OriginalStart']):
                    start_dt = row_data['OriginalStart']
                    end_dt = row_data['OriginalEnd']
                    actual_start = row_data['Start']
                    actual_end = row_data['End']

                    # For partial periods, don't stack spans
                    if row_data.get('IsPartial', False):
                        # Add light shading only for parts we don't have data for
                        if start_dt < actual_start:
                            ax_dd.axvspan(start_dt, actual_start, color=color, alpha=0.05)
                        if actual_end < end_dt:
                            ax_dd.axvspan(actual_end, end_dt, color=color, alpha=0.05)

                        # Add normal shading only for the part we have data for
                        ax_dd.axvspan(actual_start, actual_end, color=color, alpha=0.1)
                    else:
                        # For complete periods, just add normal shading
                        ax_dd.axvspan(start_dt, end_dt, color=color, alpha=0.1)
                else:
                    # Fallback to actual dates if original not available
                    start_dt = row_data['Start']
                    end_dt = row_data['End']
                    ax_dd.axvspan(start_dt, end_dt, color=color, alpha=0.1)

        # Subplots for each event
        for i, (pname, subdict) in enumerate(events_list):
            row_idx = 2 + (i // 2)
            col_idx = i % 2
            ax_stress = fig.add_subplot(gs[row_idx, col_idx])

            s_sub = subdict['s_sub']
            b_sub = subdict['b_sub']
            if len(s_sub) < 2:
                ax_stress.text(0.5, 0.5, f"No data: {pname}",
                               ha='center', va='center', transform=ax_stress.transAxes)
                ax_stress.set_title(pname)
                ax_stress.tick_params(axis='x', rotation=45)
                continue

            base_val = s_sub.iloc[0]
            s_norm = s_sub / base_val
            ax_stress.plot(s_norm.index, s_norm.values, label="Strategy")
            if has_bm and b_sub is not None and len(b_sub) >= 2:
                b0 = b_sub.iloc[0]
                b_norm = b_sub / b0
                ax_stress.plot(b_norm.index, b_norm.values, label="Benchmark", alpha=0.7)

            start_date = s_sub.index[0].strftime("%Y-%m-%d")
            end_date = s_sub.index[-1].strftime("%Y-%m-%d")
            ax_stress.set_title(f"{pname}: ({start_date} to {end_date})")
            ax_stress.legend(loc='best')
            ax_stress.grid(True)
            ax_stress.tick_params(axis='x', rotation=45)

        # If we have a table, put it in the last row
        if have_table:
            # table row = 2 + subplots_rows
            table_row = 2 + subplots_rows
            ax_table = fig.add_subplot(gs[table_row, :])
            ax_table.axis('off')

            headers = ["Period", "Duration", "Strategy Return", "Strategy Max DD"]
            if has_bm:
                headers.extend(["Benchmark Return", "Benchmark Max DD", "Relative Perf"])

            table_data = []
            for idx, row_data in stress_df.iterrows():
                start_dt = row_data['Start']
                end_dt = row_data['End']
                duration_days = (end_dt - start_dt).days if pd.notnull(start_dt) and pd.notnull(end_dt) else 0
                s_ret = row_data['Strategy_TotalRet']
                s_mdd = row_data['Strategy_MaxDD']

                line = [
                    idx,
                    f"{duration_days} days",
                    f"{s_ret:.2%}",
                    f"{s_mdd:.2%}",
                ]

                if has_bm:
                    bm_ret = row_data['Benchmark_TotalRet']
                    bm_dd_val = row_data['Benchmark_MaxDD']
                    outperf = (s_ret - bm_ret)
                    line.append(f"{bm_ret:.2%}")
                    line.append(f"{bm_dd_val:.2%}")
                    line.append(f"{outperf:.2%}")

                table_data.append(line)

            tbl = ax_table.table(
                cellText=table_data,
                colLabels=headers,
                loc='center',
                cellLoc='center'
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.2)

            # color the "Relative Perf"
            if has_bm:
                col_idx = len(headers) - 1
                for i, row_vals in enumerate(table_data):
                    cell = tbl[(i + 1, col_idx)]
                    raw_str = row_vals[col_idx].strip('%')
                    try:
                        val = float(raw_str) / 100
                        if val > 0:
                            cell.set_facecolor('#d8f3dc')
                        elif val < 0:
                            cell.set_facecolor('#ffccd5')
                    except:
                        pass

        plt.show()

    # -------------------------
    # Helper
    # -------------------------
    def _compute_drawdown(self, equity_series):
        roll_max = equity_series.cummax()
        return (equity_series - roll_max) / roll_max