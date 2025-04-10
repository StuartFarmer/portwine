import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

class Optimizer:
    """
    Base/abstract optimizer class to allow different optimization approaches.
    """

    def __init__(self, backtester):
        """
        Parameters
        ----------
        backtester : Backtester
            A pre-initialized Backtester object with run_backtest(...).
        """
        self.backtester = backtester

    def run_optimization(self, strategy_class, param_grid, **kwargs):
        """
        Abstract method: run an optimization approach (grid search, train/test, etc.)
        Must be overridden.
        """
        raise NotImplementedError("Subclasses must implement run_optimization.")

# ---------------------------------------------------------------------

class TrainTestSplitOptimizer(Optimizer):
    """
    An optimizer that implements Approach A (two separate backtests):
      1) For each param combo, run a backtest on [train_start, train_end].
         Evaluate performance (score_fn).
      2) Choose best combo by training score.
      3) Optionally run a second backtest on [test_start, test_end] for
         the best combo, storing final out-of-sample performance.
    """

    def __init__(self, backtester):
        super().__init__(backtester)

    def run_optimization(self,
                         strategy_class,
                         param_grid,
                         split_frac=0.7,
                         score_fn=None,
                         benchmark=None):
        """
        Runs a grid search over 'param_grid' for the given 'strategy_class'.

        This approach does two separate backtests:
          - one on the train portion
          - one on the test portion for the best param set

        Parameters
        ----------
        strategy_class : type
            A strategy class (e.g. SomeStrategy) that can be constructed via
            the keys in param_grid.
        param_grid : dict
            param name -> list of possible values
        split_frac : float
            fraction of the data for training (0<split_frac<1)
        score_fn : callable
            function that takes { 'strategy_returns': Series, ... } -> float
            If None, defaults to annualized Sharpe on the daily returns.
        benchmark : str or None
            optional benchmark passed to run_backtest.

        Returns
        -------
        dict with:
          'best_params' : dict of chosen param set
          'best_score'  : float
          'results_df'  : DataFrame of all combos with train_score (and maybe test_score)
          'best_test_performance' : float or None
        """
        # Default scoring => annualized Sharpe
        if score_fn is None:
            def default_sharpe(res):
                # res = {'strategy_returns': Series of daily returns, ...}
                dr = res.get('strategy_returns', pd.Series(dtype=float))
                if len(dr) < 2:
                    return np.nan
                ann = 252.0
                mu = dr.mean()*ann
                sigma = dr.std()*np.sqrt(ann)
                return mu/sigma if sigma>1e-9 else 0.0
            score_fn = default_sharpe

        # The first step is to fetch the union of all relevant data to determine the date range.
        # We can do that by a "fake" strategy with "all combos" or just pick the first param set's tickers.
        # But it's safer to gather the earliest/largest date range among all combos. We'll do a simplified approach:
        all_dates = []
        # We'll keep a dictionary mapping from each combo -> (mindate, maxdate).
        # 1) If param_grid has a 'tickers' entry that is a list of lists, convert them to tuples:
        if "tickers" in param_grid:
            converted = []
            for item in param_grid["tickers"]:
                if isinstance(item, list):
                    converted.append(tuple(item))  # cast from list -> tuple
                else:
                    converted.append(item)
            param_grid["tickers"] = converted

        combos_date_range = {}

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(product(*param_values))

        # We do a quick pass just to gather date range
        for combo in all_combos:
            combo_params = dict(zip(param_names, combo))
            # Make a strategy
            strat = strategy_class(**combo_params)
            if not hasattr(strat, 'tickers'):
                combos_date_range[combo] = (None, None)
                continue
            # fetch data
            data_dict = self.backtester.market_data_loader.fetch_data(strat.tickers)
            if not data_dict:
                combos_date_range[combo] = (None, None)
                continue
            # get min and max date across all relevant tickers
            min_d, max_d = None, None
            for tkr, df in data_dict.items():
                if df.empty:
                    continue
                dmin = df.index.min()
                dmax = df.index.max()
                if min_d is None or dmin>min_d:
                    min_d = dmin
                if max_d is None or dmax<max_d:
                    max_d = dmax
                # actually we want min_d = max of earliest
                # and max_d = min of latest
                # so let's invert:
                # if min_d is None => set it to dmin
                # else => min_d = max(min_d, dmin)
                # Similarly for max_d => min(max_d, dmax)
            combos_date_range[combo] = (min_d, max_d)

        # We'll pick the "largest common feasible range" among combos.
        # This is a design choice. Alternatively, we can do a combo-specific approach.
        # For simplicity, let's do a single union range for all combos.
        global_start = None
        global_end = None
        for c, (md, xd) in combos_date_range.items():
            if md is not None and xd is not None:
                if global_start is None or md>global_start:
                    global_start = md
                if global_end is None or xd<global_end:
                    global_end = xd
        if global_start is None or global_end is None or global_start>=global_end:
            print("No valid date range found for any combo.")
            return None

        # Now we have a global [global_start, global_end].
        # We'll pick a date range from that.
        # We'll use the naive approach: we convert them to sorted list of dates, pick split index, etc.
        # It's simpler to run two separate backtests with "start_date=..., end_date=..." for train, then for test.

        # Build a list of all daily timestamps from global_start..global_end
        # We can fetch from the underlying data in the backtester, or do a direct approach.
        # For simplicity, let's just do an approximate approach:
        # We'll gather the entire union in the same manner we do in the backtester:
        # But let's do it quickly:
        # We'll define a function get_all_dates_for_range:
        # Actually let's skip it, and do a simpler approach: we'll pass them to the backtester with end_date in train
        # and start_date in test.

        # We'll do the naive approach:
        # We want the full set of daily trading dates from global_start..global_end
        # Then we pick split. Then train is [global_start..some boundary], test is [boundary+1..global_end].
        # We'll do it with the backtester, but let's first gather a big combined df.

        # (In practice you might do a step to confirm we have a consistent set of daily trading dates. We'll do minimal approach.)
        date_range = pd.date_range(global_start, global_end, freq='B')  # business days
        n = len(date_range)
        split_idx = int(n*split_frac)
        if split_idx<1:
            print("Split fraction leaves no training days.")
            return None
        if split_idx>=n:
            print("Split fraction leaves no testing days.")
            return None
        train_end_date = date_range[split_idx-1]
        test_start_date = date_range[split_idx]
        # Summaries
        # We'll store combos results
        results_list = []

        for combo in tqdm(all_combos):
            combo_params = dict(zip(param_names, combo))
            # First => run training backtest
            strat_train = strategy_class(**combo_params)
            # train backtest
            train_res = self.backtester.run_backtest(
                strategy=strat_train,
                shift_signals=True,
                benchmark=benchmark,
                start_date=global_start,
                end_date=train_end_date
            )
            if not train_res or 'strategy_returns' not in train_res:
                results_list.append({**combo_params, "train_score": np.nan, "test_score": np.nan})
                continue
            train_dr = train_res['strategy_returns']
            if train_dr is None or len(train_dr)<2:
                results_list.append({**combo_params, "train_score": np.nan, "test_score": np.nan})
                continue

            # compute train score
            train_score = score_fn({"strategy_returns": train_dr})

            # We'll not do test backtest for each combo => to save time, do test only for best param after the loop
            # or if you want, you can do it here as well. We'll do it here so we can see how big the difference is for each combo.
            # But that can be expensive for big param grids. We'll do it anyway for completeness.

            strat_test = strategy_class(**combo_params)
            test_res = self.backtester.run_backtest(
                strategy=strat_test,
                shift_signals=True,
                benchmark=benchmark,
                start_date=test_start_date,
                end_date=global_end
            )
            if not test_res or 'strategy_returns' not in test_res:
                results_list.append({**combo_params, "train_score": train_score, "test_score": np.nan})
                continue
            test_dr = test_res['strategy_returns']
            if test_dr is None or len(test_dr)<2:
                results_list.append({**combo_params, "train_score": train_score, "test_score": np.nan})
                continue

            test_score = score_fn({"strategy_returns": test_dr})

            # store
            combo_result = {**combo_params,
                            "train_score": train_score,
                            "test_score": test_score}
            results_list.append(combo_result)

        df_results = pd.DataFrame(results_list)
        if df_results.empty:
            print("No results produced.")
            return None

        # pick best by train_score
        df_results.sort_values('train_score', ascending=False, inplace=True)
        best_row = df_results.iloc[0].to_dict()
        best_train_score = best_row['train_score']
        best_test_score = best_row['test_score']
        best_params = {k: v for k, v in best_row.items() if k not in ['train_score','test_score']}

        print("Best params:", best_params, f"train_score={best_train_score:.4f}, test_score={best_test_score:.4f}")

        return {
            "best_params": best_params,
            "best_score": best_train_score,
            "results_df": df_results,
            "best_test_score": best_test_score
        }
