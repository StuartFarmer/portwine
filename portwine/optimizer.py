import numpy as np
import pandas as pd
from itertools import product


class Optimizer:
    """
    A more abstract optimizer that:
      1) Takes a pre-initialized Backtester object (which already has a MarketDataLoader).
      2) In run_optimization(), we:
         - Accept a strategy_class, a param_grid dict for hyperparameters,
         - Accept split_frac, score_fn, and benchmark arguments,
         - Split historical data by date (train/test) for each strategy's tickers,
         - Run a single backtest over the entire date range,
         - Evaluate performance only on the test slice using a user-supplied (or default) score function.
      3) Returns the best combination plus a DataFrame of all combos & scores.
    """

    def __init__(self, backtester):
        """
        Parameters
        ----------
        backtester : Backtester
            A pre-initialized Backtester object that includes a MarketDataLoader
            and a run_backtest(...) method.
        """
        self.backtester = backtester

    def run_optimization(self,
                         strategy_class,
                         param_grid,
                         split_frac=0.7,
                         score_fn=None,
                         benchmark=None):
        """
        Runs a grid search over 'param_grid' for the given 'strategy_class'.

        Parameters
        ----------
        strategy_class : type
            A strategy class (e.g. AnticorStrategy) with a constructor that accepts
            the parameters in 'param_grid'.
        param_grid : dict
            A mapping from parameter name -> list of possible values, e.g.:
                {
                    "tickers": [["AAPL", "MSFT"], ["AMZN", "TSLA"]],
                    "window_size": [3, 5, 10]
                }
            We take the Cartesian product of these lists to get all parameter combos.
        split_frac : float, default 0.7
            Fraction of the date range used for 'training'; remainder is 'testing'.
        score_fn : callable, optional
            A function that takes a dict with 'strategy_returns' (and optionally 'benchmark_returns')
            **for the test portion** and returns a single float (higher is better).
            If None, defaults to an annualized Sharpe ratio.
        benchmark : str or None
            Optional benchmark ticker to request from the backtester, so we can have
            'benchmark_returns' in the final results dict if desired.

        Returns
        -------
        dict with keys:
            'best_params': dict of the best parameter combo,
            'best_score': float (the top score),
            'results_df': DataFrame with all combos + 'score' column, sorted descending by 'score'.
        """

        # If no scoring function is given, use an annualized Sharpe
        if score_fn is None:
            def default_sharpe(results):
                test_returns = results['strategy_returns']
                ann_factor = 252.0
                mu = test_returns.mean() * ann_factor
                sigma = test_returns.std() * np.sqrt(ann_factor)
                return mu / sigma if sigma > 1e-9 else 0.0
            score_fn = default_sharpe

        # Helper function: given a price DataFrame (index=dates), split into train/test sets
        def split_dates(df, frac):
            all_dates = df.index.unique().sort_values()
            n = len(all_dates)
            split_idx = int(n * frac)
            train_dates = all_dates[:split_idx]
            test_dates = all_dates[split_idx:]
            return set(train_dates), set(test_dates)

        results_list = []

        # Cartesian product of param_grid
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(product(*param_values))  # e.g. if 2 keys, we get cross product

        for combo in tqdm(all_combos):
            combo_params = dict(zip(param_names, combo))

            # Instantiate the strategy
            strategy = strategy_class(**combo_params)

            # The strategy should define .tickers
            if not hasattr(strategy, "tickers"):
                # No tickers => no data => can't proceed
                results_list.append({**combo_params, "score": np.nan})
                continue

            # Fetch data for the strategy's tickers
            data_dict = self.backtester.market_data_loader.fetch_data(strategy.tickers)
            if not data_dict:
                # No data
                results_list.append({**combo_params, "score": np.nan})
                continue

            # Combine into a single DataFrame of close prices for date splitting
            combined_closes = []
            for tk, df_data in data_dict.items():
                if 'close' not in df_data.columns:
                    continue
                tmp = df_data[['close']].copy()
                tmp.columns = [tk]
                combined_closes.append(tmp)

            if not combined_closes:
                results_list.append({**combo_params, "score": np.nan})
                continue

            full_df = pd.concat(combined_closes, axis=1).sort_index()
            full_df.ffill(inplace=True)
            full_df.dropna(how='all', inplace=True)

            if full_df.empty:
                results_list.append({**combo_params, "score": np.nan})
                continue

            # Split into train/test date sets
            train_dates, test_dates = split_dates(full_df, split_frac)

            # Run backtest over entire date range
            backtest_res = self.backtester.run_backtest(
                strategy=strategy,
                shift_signals=True,
                benchmark=benchmark
            )

            # If something failed or no returns are produced
            if not backtest_res or 'strategy_returns' not in backtest_res:
                results_list.append({**combo_params, "score": np.nan})
                continue

            strat_ret = backtest_res['strategy_returns']
            if strat_ret is None or len(strat_ret) < 2:
                results_list.append({**combo_params, "score": np.nan})
                continue

            # Slice test portion
            strat_ret_train = strat_ret[strat_ret.index.map(lambda d: d in train_dates)]
            if len(strat_ret_train) < 2:
                # Not enough test data to compute meaningful metrics
                results_list.append({**combo_params, "score": np.nan})
                continue

            # Prepare dict for scoring
            train_dict = {"strategy_returns": strat_ret_train}
            if 'benchmark_returns' in backtest_res and backtest_res['benchmark_returns'] is not None:
                bench_ret = backtest_res['benchmark_returns']
                bench_ret_train = bench_ret[bench_ret.index.map(lambda d: d in train_dates)]
                train_dict["benchmark_returns"] = bench_ret_train

            # Compute score
            score = score_fn(train_dict)

            # Save
            combo_result = {**combo_params, "score": score}
            results_list.append(combo_result)

        # Build a DataFrame of all combos & scores
        df_results = pd.DataFrame(results_list)
        if df_results.empty:
            print("No results produced.")
            return None

        df_results.sort_values('score', ascending=False, inplace=True)
        best_row = df_results.iloc[0].to_dict()
        best_score = best_row['score']
        best_params = {k: v for k, v in best_row.items() if k != 'score'}

        print("Best params:", best_params, "score=", best_score)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "results_df": df_results
        }
