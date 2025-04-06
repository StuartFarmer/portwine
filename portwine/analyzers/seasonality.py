import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
from scipy import stats
from statsmodels.stats.multitest import multipletests
from portwine.analyzers.base import Analyzer

class SeasonalityAnalyzer(Analyzer):
    """
    A Seasonality Analyzer that:
      1) Adds time-based features (day_of_week, month, quarter, year, etc.)
         and special "turn-of" indicators for month and quarter (±3 days).
      2) Uses a generic seasonality analysis method to handle day_of_week,
         month, quarter, year, or any other time grouping.
      3) Plots each grouping in a temporal (natural) order on the x-axis
         (e.g., Monday->Sunday, Jan->Dec, Q1->Q4, ascending years, etc.).
    """

    def analyze(self, results, significance_level=0.05, benchmark_comparison=True):
        """
        Performs seasonality analysis on strategy & optional benchmark returns.

        Parameters
        ----------
        results : dict
            Backtester results dict with keys:
                'strategy_returns': pd.Series
                'benchmark_returns': pd.Series (optional)
        significance_level : float
            Alpha level for statistical tests (default: 0.05).
        benchmark_comparison : bool
            Whether to compare strategy against benchmark.

        Returns
        -------
        analysis_results : dict
            {
                'day_of_week': { 'stats': DataFrame, ... },
                'month_of_year': { ... },
                'quarter': { ... },
                'year': { ... },
                'turn_of_month': { ... },
                'turn_of_quarter': { ... }
            }
        """
        strategy_returns = results.get('strategy_returns', pd.Series(dtype=float))
        benchmark_returns = (results.get('benchmark_returns', pd.Series(dtype=float))
                             if benchmark_comparison else None)

        # Ensure DatetimeIndex
        if not isinstance(strategy_returns.index, pd.DatetimeIndex):
            strategy_returns.index = pd.to_datetime(strategy_returns.index)
        if benchmark_returns is not None and not isinstance(benchmark_returns.index, pd.DatetimeIndex):
            benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

        # Build DataFrames with features
        strat_df = self._add_time_features(strategy_returns)
        bench_df = self._add_time_features(benchmark_returns) if benchmark_returns is not None else None

        # We define a dictionary mapping each "analysis label" to the column & optional display name
        time_periods = {
            'day_of_week': {'group_col': 'day_of_week', 'display_col': 'day_name'},
            'month_of_year': {'group_col': 'month', 'display_col': 'month_name'},
            'quarter': {'group_col': 'quarter', 'display_col': None},
            'year': {'group_col': 'year', 'display_col': None},
        }

        analysis_results = {}

        # Generic analysis for day_of_week, month_of_year, quarter, year
        for label, info in time_periods.items():
            analysis_results[label] = self._analyze_seasonality(
                df=strat_df,
                column=info['group_col'],
                display_name_column=info['display_col'],
                benchmark_df=bench_df,
                alpha=significance_level
            )

        # Turn-of-month and turn-of-quarter
        analysis_results['turn_of_month'] = self._analyze_turn(
            strat_df, bench_df, prefix='tom', alpha=significance_level
        )
        analysis_results['turn_of_quarter'] = self._analyze_turn(
            strat_df, bench_df, prefix='toq', alpha=significance_level
        )

        return analysis_results

    def plot(self, results, analysis_results=None, figsize=(15, 18)):
        """
        Plot each seasonal grouping in subplots, with a temporal or natural ordering on the x-axis.

        Parameters
        ----------
        results : dict
            Backtester results dict (see analyze).
        analysis_results : dict, optional
            Results from analyze() method (will be computed if None).
        figsize : tuple
            Size of the entire figure.
        """
        if analysis_results is None:
            analysis_results = self.analyze(results)

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)

        # For each row/col, we define a label & chart title
        subplot_map = [
            ('day_of_week', (0, 0), "Day of Week"),
            ('month_of_year', (0, 1), "Month of Year"),
            ('quarter', (1, 0), "Quarter"),
            ('year', (1, 1), "Year"),
            ('turn_of_month', (2, 0), "Turn of Month"),
            ('turn_of_quarter', (2, 1), "Turn of Quarter")
        ]

        for key, (r, c), title in subplot_map:
            ax = axes[r][c]
            self._plot_seasonality(analysis_results.get(key), title=title, ax=ax)

        plt.tight_layout()
        plt.show()

    def generate_report(self, results, analysis_results=None, alpha=0.05):
        """
        Generate a text report summarizing the analysis.

        Parameters
        ----------
        results : dict
            Backtester results dict (see analyze).
        analysis_results : dict, optional
            Results from analyze() method (will be computed if None).
        alpha : float
            Significance level (default: 0.05).

        Returns
        -------
        str
            Multi-line text report.
        """
        if analysis_results is None:
            analysis_results = self.analyze(results, significance_level=alpha)

        lines = []
        lines.append("=== SEASONALITY ANALYSIS REPORT ===\n")

        strat_ret = results.get('strategy_returns', pd.Series(dtype=float))
        if not strat_ret.empty:
            lines.append(f"Overall Mean Return: {strat_ret.mean():.4%}")
            lines.append(f"Overall Positive Days: {(strat_ret > 0).mean():.2%}")
            lines.append(f"Total Days Analyzed: {len(strat_ret)}")
        lines.append("")

        label_map = {
            'day_of_week': "DAY OF WEEK",
            'month_of_year': "MONTH OF YEAR",
            'quarter': "QUARTER",
            'year': "YEAR",
            'turn_of_month': "TURN OF MONTH",
            'turn_of_quarter': "TURN OF QUARTER"
        }

        for key, title in label_map.items():
            lines.append(f"=== {title} ANALYSIS ===")
            result_dict = analysis_results.get(key)
            lines.extend(self._format_seasonality_report(result_dict, label_name=title))
            lines.append("")

        return "\n".join(lines)

    ###########################################################################
    # Internals
    ###########################################################################
    def _add_time_features(self, returns_series):
        """
        Adds standard time-based features to a returns Series (day, month, quarter, etc.)
        Also adds flags for turn-of-month and turn-of-quarter from T-3 to T+3.
        """
        if returns_series is None or returns_series.empty:
            return None

        df = pd.DataFrame({'returns': returns_series})

        # Basic time features
        df['day_of_week'] = df.index.dayofweek  # 0=Monday
        df['day_name'] = df.index.day_name()
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['month_name'] = df.index.month_name()
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Reorder categories for day_of_week and month_name
        # so plots can follow chronological order if we choose to map them
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_name'] = pd.Categorical(df['day_name'], categories=day_order, ordered=True)

        month_order = [calendar.month_name[i] for i in range(1, 13)]
        df['month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)

        # Mark turn-of-month and turn-of-quarter
        self._mark_turn(df, prefix='tom', period_months=range(1, 13))  # All months
        self._mark_turn(df, prefix='toq', period_months=[1, 4, 7, 10])  # Quarter start months
        return df

    def _mark_turn(self, df, prefix, period_months, day_range=range(-3, 4)):
        """
        Mark turn-of-month or turn-of-quarter:
          prefix='tom' or 'toq'
          period_months: which months define a "turn" (1..12 or [1,4,7,10])
          day_range: e.g. -3..+3
        """
        # Identify the 'first day' of each relevant month
        is_start = df.index.month.isin(period_months) & (df.index.day == 1)

        for offset in day_range:
            col_name = f"{prefix}_{offset}"
            df[col_name] = False
            for idx in df.index[is_start]:
                shifted_idx = idx + pd.Timedelta(days=offset)
                if shifted_idx in df.index:
                    df.loc[shifted_idx, col_name] = True

    def _analyze_turn(self, strategy_df, benchmark_df, prefix='tom', alpha=0.05):
        """
        Analyze returns around turn-of-month (tom) or turn-of-quarter (toq) ±3 days
        by grouping them into a single factor column (e.g. 'tom_day').
        """
        if strategy_df is None or strategy_df.empty:
            return None

        offset_labels = {
            f'{prefix}_-3': 'T-3',
            f'{prefix}_-2': 'T-2',
            f'{prefix}_-1': 'T-1',
            f'{prefix}_0': 'T+0',
            f'{prefix}_1': 'T+1',
            f'{prefix}_2': 'T+2',
            f'{prefix}_3': 'T+3'
        }

        strat_frames = []
        bench_frames = []

        for col, label in offset_labels.items():
            if col in strategy_df.columns:
                subset_strat = strategy_df.loc[strategy_df[col], :].copy()
                if not subset_strat.empty:
                    subset_strat[f'{prefix}_day'] = label
                    strat_frames.append(subset_strat)

            if benchmark_df is not None and not benchmark_df.empty and col in benchmark_df.columns:
                subset_bench = benchmark_df.loc[benchmark_df[col], :].copy()
                if not subset_bench.empty:
                    subset_bench[f'{prefix}_day'] = label
                    bench_frames.append(subset_bench)

        if not strat_frames:
            return None

        df_strat_all = pd.concat(strat_frames)
        df_bench_all = pd.concat(bench_frames) if bench_frames else None

        return self._analyze_seasonality(
            df_strat_all,
            column=f'{prefix}_day',
            benchmark_df=df_bench_all,
            alpha=alpha
        )

    def _analyze_seasonality(self,
                             df,
                             column,
                             value_column='returns',
                             benchmark_df=None,
                             alpha=0.05,
                             display_name_column=None):
        """
        Generic seasonality analysis for a given column (e.g., 'day_of_week', 'month', etc.).
        Groups by that column, computes means, medians, stats, t-tests, etc.
        Optionally compares strategy vs. benchmark if benchmark_df is provided.
        """
        if df is None or df.empty:
            return None

        grouped = df.groupby(column)[value_column]
        stats_df = pd.DataFrame({
            'mean': grouped.mean(),
            'median': grouped.median(),
            'std': grouped.std(),
            'count': grouped.count(),
            'positive_pct': grouped.apply(lambda x: (x > 0).mean()),
            'cumulative': grouped.apply(lambda x: (1 + x).prod() - 1),
        })

        # Overall stats
        overall_mean = df[value_column].mean()
        overall_std = df[value_column].std()
        overall_n = len(df)

        # T-tests: group vs. non-group
        t_stats, p_values = [], []
        for group_val in stats_df.index:
            group_data = df.loc[df[column] == group_val, value_column]
            non_group_data = df.loc[df[column] != group_val, value_column]
            t_stat, p_val = stats.ttest_ind(group_data, non_group_data, equal_var=False)
            t_stats.append(t_stat)
            p_values.append(p_val)

        stats_df['t_stat'] = t_stats
        stats_df['p_value'] = p_values
        stats_df['significant'] = stats_df['p_value'] < alpha
        # Effect size
        stats_df['effect_size'] = (stats_df['mean'] - overall_mean) / (overall_std if overall_std != 0 else 1e-9)

        # Multiple testing correction
        _, corr_pvals, _, _ = multipletests(stats_df['p_value'].values, alpha=alpha, method='fdr_bh')
        stats_df['corrected_p_value'] = corr_pvals
        stats_df['significant_corrected'] = stats_df['corrected_p_value'] < alpha

        # If benchmark is provided
        if benchmark_df is not None and not benchmark_df.empty:
            bm_grouped = benchmark_df.groupby(column)[value_column]
            bm_stats = pd.DataFrame({
                'benchmark_mean': bm_grouped.mean(),
                'benchmark_median': bm_grouped.median(),
                'benchmark_cumulative': bm_grouped.apply(lambda x: (1 + x).prod() - 1)
            })
            stats_df = stats_df.merge(bm_stats, left_index=True, right_index=True, how='left')

            # Strategy vs. Benchmark t-test within each group
            strat_vs_bm_t, strat_vs_bm_p = [], []
            for group_val in stats_df.index:
                strat_grp = df.loc[df[column] == group_val, value_column]
                bm_grp = benchmark_df.loc[benchmark_df[column] == group_val, value_column]
                if not strat_grp.empty and not bm_grp.empty:
                    t_stat, p_val = stats.ttest_ind(strat_grp, bm_grp, equal_var=False)
                else:
                    t_stat, p_val = np.nan, np.nan
                strat_vs_bm_t.append(t_stat)
                strat_vs_bm_p.append(p_val)

            stats_df['strat_vs_bm_t'] = strat_vs_bm_t
            stats_df['strat_vs_bm_p'] = strat_vs_bm_p
            stats_df['strat_vs_bm_significant'] = stats_df['strat_vs_bm_p'] < alpha

            # Correct these p-values
            finite_mask = stats_df['strat_vs_bm_p'].notnull()
            if finite_mask.sum() > 0:
                _, corr_bm_pvals, _, _ = multipletests(
                    stats_df.loc[finite_mask, 'strat_vs_bm_p'].values,
                    alpha=alpha, method='fdr_bh'
                )
                stats_df['strat_vs_bm_corrected_p'] = np.nan
                stats_df.loc[finite_mask, 'strat_vs_bm_corrected_p'] = corr_bm_pvals
                stats_df['strat_vs_bm_significant_corrected'] = (
                        stats_df['strat_vs_bm_corrected_p'] < alpha
                )

        # Remap to display names if requested
        if display_name_column and display_name_column in df.columns:
            # Build a map from numeric group_val -> display_name (like day_of_week -> Monday, etc.)
            unique_map = df[[column, display_name_column]].drop_duplicates()
            mapper = dict(zip(unique_map[column], unique_map[display_name_column]))
            # Re-map index to display
            new_index = []
            for val in stats_df.index:
                new_index.append(mapper[val] if val in mapper else val)
            stats_df.index = new_index

        # No longer automatically sorting by 'mean' descending.
        # Instead, we keep the natural/temporal order in the final plot step.

        return {
            'stats': stats_df,
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'overall_n': overall_n
        }

    def _plot_seasonality(self, seasonality_result, title, ax):
        """
        Plot a bar chart of the mean returns for each group in their temporal/natural order.
        If a benchmark is present, show it alongside strategy bars.
        """
        if seasonality_result is None:
            ax.text(0.5, 0.5, "No data available",
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
            return

        df_stats = seasonality_result['stats'].copy()
        if df_stats.empty:
            ax.text(0.5, 0.5, "No data available",
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
            return

        # Reorder the index in a temporal manner depending on 'title'
        df_stats = self._reorder_index_temporally(df_stats, title)

        x_labels = df_stats.index
        x_pos = np.arange(len(x_labels))
        mean_vals = df_stats['mean'].values
        has_benchmark = ('benchmark_mean' in df_stats.columns)

        width = 0.35 if has_benchmark else 0.5

        # Plot strategy bars
        bars_strat = ax.bar(
            x_pos - (width / 2 if has_benchmark else 0),
            mean_vals,
            width=width,
            label='Strategy'
        )

        # Color code significant bars
        if 'significant_corrected' in df_stats.columns:
            for i, sig in enumerate(df_stats['significant_corrected']):
                if sig:
                    bars_strat[i].set_color('green')
                    bars_strat[i].set_alpha(0.7)

        # Plot benchmark bars if available
        if has_benchmark:
            bench_vals = df_stats['benchmark_mean'].values
            bars_bench = ax.bar(
                x_pos + width / 2,
                bench_vals,
                width=width,
                label='Benchmark',
                color='orange',
                alpha=0.7
            )

        # Plot an overall mean line
        overall_mean = seasonality_result['overall_mean']
        ax.axhline(y=overall_mean, color='r', linestyle='--', alpha=0.5,
                   label=f'Overall Mean: {overall_mean:.4%}')

        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_ylabel('Mean Return')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend()

    def _reorder_index_temporally(self, df_stats, title):
        """
        Reorder df_stats.index in a natural or temporal way based on the chart title:
          - Day of Week: Monday->Sunday
          - Month of Year: use abbreviated month names (Jan->Dec)
          - Quarter: map numeric to Q1..Q4 or reorder Q1..Q4
          - Year: use the last two digits (e.g., 2021 -> 21)
          - Turn of Month or Quarter: T-3..T+3
        Only drop rows if their 'mean' is truly NaN, preserving partial data.
        """
        if df_stats is None or df_stats.empty:
            return df_stats

        idx_list = list(df_stats.index)

        # 1) Day of Week
        if "Day of Week" in title:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sorted_order = [d for d in day_order if d in idx_list]
            df_stats = df_stats.reindex(sorted_order)
            df_stats = df_stats.dropna(subset=["mean"])
            return df_stats

        # 2) Month of Year
        if "Month of Year" in title:
            # The DataFrame index currently has full month names like "January", "February", ...
            # We reorder them, then rename them to abbreviated forms.
            full_to_abbrev = {
                "January": "Jan", "February": "Feb", "March": "Mar",
                "April": "Apr", "May": "May", "June": "Jun",
                "July": "Jul", "August": "Aug", "September": "Sep",
                "October": "Oct", "November": "Nov", "December": "Dec"
            }

            month_order = list(full_to_abbrev.keys())  # [January, February, ..., December]
            # Reindex in chronological order
            sorted_order = [m for m in month_order if m in idx_list]
            df_stats = df_stats.reindex(sorted_order)
            df_stats = df_stats.dropna(subset=["mean"])

            # Now rename to abbreviations
            new_index = []
            for m in df_stats.index:
                if m in full_to_abbrev:
                    new_index.append(full_to_abbrev[m])
                else:
                    new_index.append(m)  # fallback if something unrecognized
            df_stats.index = new_index

            return df_stats

        # 3) Quarter
        if "Quarter" in title:
            # We might see numeric 1..4 or strings "Q1","Q2", etc.
            def is_qstring(x):
                return isinstance(x, str) and len(x) == 2 and x.upper().startswith("Q")

            string_items = [x for x in idx_list if isinstance(x, str)]
            all_qstrings = all(is_qstring(s) for s in string_items) and len(string_items) == len(idx_list)

            if all_qstrings:
                quarter_order = ["Q1", "Q2", "Q3", "Q4"]
                sorted_order = [q for q in quarter_order if q in idx_list]
                df_stats = df_stats.reindex(sorted_order)
                df_stats = df_stats.dropna(subset=["mean"])
                return df_stats

            # Otherwise parse numerics
            numeric_map = {}
            for item in idx_list:
                try:
                    numeric_map[item] = int(item)  # e.g. 1..4
                except:
                    numeric_map[item] = None

            valid_items = [k for k, v in numeric_map.items() if v in [1, 2, 3, 4]]
            valid_items.sort(key=lambda x: numeric_map[x])

            if valid_items:
                df_stats = df_stats.loc[valid_items]
                new_index = []
                for old in valid_items:
                    q_num = numeric_map[old]
                    new_index.append(f"Q{q_num}")
                df_stats.index = new_index
                df_stats = df_stats.dropna(subset=["mean"])
                return df_stats

            # Fallback: just drop rows that are NaN in mean
            df_stats = df_stats.dropna(subset=["mean"])
            return df_stats

        # 4) Year
        if "Year" in title:
            # We parse each index item as int => then sort ascending => rename to last two digits
            numeric_map = {}
            for item in idx_list:
                try:
                    numeric_map[item] = int(item)
                except:
                    numeric_map[item] = None

            valid_items = [k for k, v in numeric_map.items() if v is not None]
            # Sort ascending by parsed year
            valid_items.sort(key=lambda x: numeric_map[x])

            if valid_items:
                df_stats = df_stats.loc[valid_items]
                # rename to last two digits
                new_index = []
                for old in valid_items:
                    y = numeric_map[old]
                    # last two digits
                    y2 = y % 100
                    new_index.append(f"{y2:02d}")  # e.g. "21", "22"
                df_stats.index = new_index

            df_stats = df_stats.dropna(subset=["mean"])
            return df_stats

        # 5) Turn of Month / Quarter
        if "Turn of Month" in title or "Turn of Quarter" in title:
            turn_order = ["T-3", "T-2", "T-1", "T+0", "T+1", "T+2", "T+3"]
            sorted_order = [d for d in turn_order if d in idx_list]
            df_stats = df_stats.reindex(sorted_order)
            df_stats = df_stats.dropna(subset=["mean"])
            return df_stats

        # Otherwise, return as is
        df_stats = df_stats.dropna(subset=["mean"])
        return df_stats

    def _format_seasonality_report(self, seasonality_result, label_name):
        """
        Return a list of lines describing the seasonality results for a given period.
        """
        if (seasonality_result is None
                or 'stats' not in seasonality_result
                or seasonality_result['stats'].empty):
            return ["No data available.\n"]

        df_stats = seasonality_result['stats']

        lines = []
        lines.append(f"--- {label_name} ---")

        # We won't reorder the DataFrame here,
        # because the user specifically wanted temporal ordering in the plot,
        # not necessarily for the table. But we can do it here if desired:
        # For textual best/worst we might want them by "mean" descending:
        sorted_df = df_stats.sort_values('mean', ascending=False)

        # Best & worst
        best_idx = sorted_df.index[0]
        worst_idx = sorted_df.index[-1]
        best_val = sorted_df.iloc[0]['mean']
        worst_val = sorted_df.iloc[-1]['mean']

        lines.append(f"Best {label_name}: {best_idx} ({best_val:.4%})")
        lines.append(f"Worst {label_name}: {worst_idx} ({worst_val:.4%})\n")

        # Table header
        header = (f"{label_name:<15s} | {'Mean':>7s} | {'Median':>7s} | "
                  f"{'Win %':>5s} | {'Count':>5s} | {'Signif?'} ")
        lines.append(header)
        lines.append("-" * len(header))

        for idx, row in sorted_df.iterrows():
            is_sig = '*' if row.get('significant_corrected', False) else ''
            lines.append(
                f"{str(idx):<15s} | "
                f"{row['mean']:>7.2%} | "
                f"{row['median']:>7.2%} | "
                f"{row['positive_pct']:>5.2%} | "
                f"{int(row['count']):>5d} | {is_sig}"
            )

        lines.append("")
        lines.append("* indicates statistical significance (after FDR correction)")
        return lines
