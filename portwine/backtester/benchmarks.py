import cvxpy as cp
import numpy as np
from typing import Callable, Dict, List
import pandas as pd
from tqdm import tqdm
from portwine.data.providers.loader_adapters import MarketDataLoader

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
            # Ensure covariance is symmetric/Hermitian for quad_form
            cov = (cov + cov.T) / 2.0
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
# 1: Ticker (exists in market_data_loader or data_interface)
# 2: Custom method (callable, or a string of a function in the global namespace)
# 3: Invalid (neither a string nor a callable)
def get_benchmark_type(benchmark: str | Callable, market_data_loader: MarketDataLoader=None, data_interface=None) -> int:
    benchmark_is_str = isinstance(benchmark, str)

    # Check if the benchmark is a standard benchmark
    if benchmark in STANDARD_BENCHMARKS:
        return BenchmarkTypes.STANDARD_BENCHMARK
    

    # Check if the benchmark is a ticker in the market_data_loader or data_interface
    if benchmark_is_str:
        # Alternative data tickers (with colons) should not be valid benchmarks
        if ':' in benchmark:
            return BenchmarkTypes.INVALID
        
        # Try market_data_loader first (for backward compatibility). Support DataStore-like loaders too.
        if market_data_loader is not None:
            try:
                # Preferred old-style loader API
                if hasattr(market_data_loader, "fetch_data"):
                    fetched = market_data_loader.fetch_data([benchmark])
                    if fetched.get(benchmark) is not None:
                        return BenchmarkTypes.TICKER
                # DataStore-style API: exists/identifiers/get
                elif hasattr(market_data_loader, "exists"):
                    if market_data_loader.exists(benchmark):
                        return BenchmarkTypes.TICKER
                elif hasattr(market_data_loader, "identifiers"):
                    if benchmark in set(market_data_loader.identifiers() or []):
                        return BenchmarkTypes.TICKER
                elif hasattr(market_data_loader, "get"):
                    # Probe a few dates
                    for probe in [
                        pd.Timestamp('2020-01-01'),
                        pd.Timestamp('2020-06-01'),
                        pd.Timestamp('2021-01-01'),
                    ]:
                        if market_data_loader.get(benchmark, probe) is not None:
                            return BenchmarkTypes.TICKER
            except Exception:
                # Fall through to interface probe
                pass
        
        # Try data_interface if available
        if data_interface is not None:
            try:
                # Set a dummy timestamp to test if the ticker exists
                original_timestamp = data_interface.current_timestamp
                
                # Try multiple timestamps to find one that works
                test_timestamps = [
                    pd.Timestamp('2020-01-01'),
                    pd.Timestamp('2020-01-15'),
                    pd.Timestamp('2020-06-01'),
                    pd.Timestamp('2021-01-01')
                ]
                
                ticker_found = False
                for test_ts in test_timestamps:
                    try:
                        data_interface.set_current_timestamp(test_ts)
                        data_interface[benchmark]  # This will raise KeyError if ticker doesn't exist
                        ticker_found = True
                        break
                    except (KeyError, ValueError):
                        continue
                
                data_interface.set_current_timestamp(original_timestamp)
                
                if ticker_found:
                    return BenchmarkTypes.TICKER
                else:
                    return BenchmarkTypes.INVALID
                    
            except (KeyError, ValueError):
                return BenchmarkTypes.INVALID
    

    # Check if the benchmark is a function in the global namespace
    if callable(benchmark):
        return BenchmarkTypes.CUSTOM_METHOD
    
    # Check if the benchmark is a string and a function in the global namespace
    if benchmark_is_str:
        obj = globals().get(benchmark)
        if callable(obj):
            return BenchmarkTypes.CUSTOM_METHOD
        
    return BenchmarkTypes.INVALID
