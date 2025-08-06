import cvxpy as cp
import numpy as np
from typing import Callable, Dict, List
import pandas as pd
from tqdm import tqdm
from portwine.loaders.base import MarketDataLoader

class InvalidBenchmarkError(Exception):
    """Raised when the requested benchmark is neither a standard name nor a valid ticker."""
    pass



def benchmark_equal_weight(ret_df: pd.DataFrame, *_, **__) -> pd.Series:
    return ret_df.mean(axis=1)


def benchmark_markowitz(
    ret_df: pd.DataFrame,
    lookback: int = 60,
    shift_signals: bool = True,
    verbose: bool = False,
) -> pd.Series:
    tickers = ret_df.columns
    n = len(tickers)
    iterator = tqdm(ret_df.index, desc="Markowitz") if verbose else ret_df.index
    w_rows: List[np.ndarray] = []
    for ts in iterator:
        win = ret_df.loc[:ts].tail(lookback)
        if len(win) < 2:
            w = np.ones(n) / n
        else:
            cov = win.cov().values
            w_var = cp.Variable(n, nonneg=True)
            prob = cp.Problem(cp.Minimize(cp.quad_form(w_var, cov)), [cp.sum(w_var) == 1])
            try:
                prob.solve()
                w = w_var.value if w_var.value is not None else np.ones(n) / n
            except Exception:
                w = np.ones(n) / n
        w_rows.append(w)
    w_df = pd.DataFrame(w_rows, index=ret_df.index, columns=tickers)
    if shift_signals:
        w_df = w_df.shift(1).ffill().fillna(1.0 / n)
    return (w_df * ret_df).sum(axis=1)


STANDARD_BENCHMARKS: Dict[str, Callable] = {
    "equal_weight": benchmark_equal_weight,
    "markowitz":    benchmark_markowitz,
}


class BenchmarkTypes:
    STANDARD_BENCHMARK = 0
    TICKER             = 1
    CUSTOM_METHOD      = 2
    INVALID            = 3


# Returns the type of benchmark
# 0: Standard benchmark (exists in STANDARD_BENCHMARKS)
# 1: Ticker (exists in market_data_loader)
# 2: Custom method (callable, or a string of a function in the global namespace)
# 3: Invalid (neither a string nor a callable)
def get_benchmark_type(benchmark: str | Callable, market_data_loader: MarketDataLoader=None) -> int:
    benchmark_is_str = isinstance(benchmark, str)

    # Check if the benchmark is a standard benchmark
    if benchmark in STANDARD_BENCHMARKS:
        return BenchmarkTypes.STANDARD_BENCHMARK
    

    # Check if the benchmark is a in the market_data_loader
    if market_data_loader is not None and benchmark_is_str:
        if market_data_loader.fetch_data([benchmark]).get(benchmark) is not None:
            return BenchmarkTypes.TICKER
    

    # Check if the benchmark is a function in the global namespace
    if callable(benchmark):
        return BenchmarkTypes.CUSTOM_METHOD
    
    # Check if the benchmark is a string and a function in the global namespace
    if benchmark_is_str:
        obj = globals().get(benchmark)
        if callable(obj):
            return BenchmarkTypes.CUSTOM_METHOD
        
    return BenchmarkTypes.INVALID
