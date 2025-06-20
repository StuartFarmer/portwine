# Backtester API Reference

This module provides a step-driven backtester that supports intraday bars and optional exchange trading calendars.

## Exceptions

### InvalidBenchmarkError

```python
class InvalidBenchmarkError(Exception):
    """Raised when the requested benchmark is neither a standard name nor a valid ticker."""
```

**Description**: Custom exception raised when an invalid benchmark is provided to the backtester.

## Functions

### benchmark_equal_weight

```python
def benchmark_equal_weight(ret_df: pd.DataFrame, *_, **__) -> pd.Series:
    """
    Calculate equal-weighted benchmark returns.
    
    Parameters
    ----------
    ret_df : pd.DataFrame
        DataFrame containing asset returns with tickers as columns and dates as index.
    *_, **__ : 
        Additional arguments (ignored for compatibility).
        
    Returns
    -------
    pd.Series
        Equal-weighted portfolio returns.
        
    Notes
    -----
    This benchmark assigns equal weights (1/n) to all assets in the portfolio.
    """
    return ret_df.mean(axis=1)
```

### benchmark_markowitz

```python
def benchmark_markowitz(
    ret_df: pd.DataFrame,
    lookback: int = 60,
    shift_signals: bool = True,
    verbose: bool = False,
) -> pd.Series:
    """
    Calculate Markowitz mean-variance optimized benchmark returns.
    
    Parameters
    ----------
    ret_df : pd.DataFrame
        DataFrame containing asset returns with tickers as columns and dates as index.
    lookback : int, default=60
        Number of periods to use for covariance estimation.
    shift_signals : bool, default=True
        Whether to apply weights on the next day to prevent lookahead bias.
    verbose : bool, default=False
        Whether to show progress bar during optimization.
        
    Returns
    -------
    pd.Series
        Markowitz optimized portfolio returns.
        
    Notes
    -----
    Uses convex optimization to minimize portfolio variance subject to full investment constraint.
    Falls back to equal weights if optimization fails.
    """
```

## Constants

### STANDARD_BENCHMARKS

```python
STANDARD_BENCHMARKS: Dict[str, Callable] = {
    "equal_weight": benchmark_equal_weight,
    "markowitz":    benchmark_markowitz,
}
```

**Description**: Dictionary mapping standard benchmark names to their corresponding functions.

## Classes

### BenchmarkTypes

```python
class BenchmarkTypes:
    """
    Enumeration of benchmark types used by the backtester.
    
    Attributes
    ----------
    STANDARD_BENCHMARK : int
        Built-in benchmark (equal_weight, markowitz).
    TICKER : int
        Single ticker symbol benchmark.
    CUSTOM_METHOD : int
        Custom benchmark function.
    INVALID : int
        Invalid benchmark type.
    """
    STANDARD_BENCHMARK = 0
    TICKER             = 1
    CUSTOM_METHOD      = 2
    INVALID            = 3
```

### Backtester

```python
class Backtester:
    """
    A step-driven backtester that supports intraday bars and optional exchange trading calendars.
    
    The Backtester class is the core component for executing trading strategies and generating
    performance results. It handles data loading, signal generation, return calculation,
    and benchmark comparison.
    
    Parameters
    ----------
    market_data_loader : MarketDataLoader
        The primary data loader for market data.
    alternative_data_loader : optional
        Additional data loader for alternative data sources.
    calendar : str or mcal.ExchangeCalendar, optional
        Trading calendar for date filtering. Can be a string (calendar name) or
        ExchangeCalendar object.
        
    Attributes
    ----------
    market_data_loader : MarketDataLoader
        The primary market data loader instance.
    alternative_data_loader : optional
        Alternative data loader instance.
    calendar : mcal.ExchangeCalendar or None
        Trading calendar instance.
        
    Examples
    --------
    >>> from portwine import Backtester, EODHDMarketDataLoader
    >>> import pandas_market_calendars as mcal
    >>> 
    >>> # Basic backtester
    >>> data_loader = EODHDMarketDataLoader(data_path='path/to/data/')
    >>> backtester = Backtester(market_data_loader=data_loader)
    >>> 
    >>> # With trading calendar
    >>> calendar = mcal.get_calendar('NYSE')
    >>> backtester = Backtester(
    ...     market_data_loader=data_loader,
    ...     calendar=calendar
    ... )
    """
```

#### Methods

##### __init__

```python
def __init__(
    self,
    market_data_loader: MarketDataLoader,
    alternative_data_loader=None,
    calendar: Optional[Union[str, mcal.ExchangeCalendar]] = None
):
    """
    Initialize the Backtester.
    
    Parameters
    ----------
    market_data_loader : MarketDataLoader
        The primary data loader for market data.
    alternative_data_loader : optional
        Additional data loader for alternative data sources.
    calendar : str or mcal.ExchangeCalendar, optional
        Trading calendar for date filtering. Can be a string (calendar name) or
        ExchangeCalendar object.
    """
```

##### _split_tickers

```python
def _split_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Split tickers into regular and alternative data tickers.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to split.
        
    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of (regular_tickers, alternative_tickers).
        
    Notes
    -----
    Alternative data tickers are identified by the presence of ':' in the ticker symbol.
    """
```

##### get_benchmark_type

```python
def get_benchmark_type(self, benchmark) -> int:
    """
    Determine the type of benchmark provided.
    
    Parameters
    ----------
    benchmark : str or callable
        The benchmark to classify.
        
    Returns
    -------
    int
        Benchmark type from BenchmarkTypes enum.
        
    Notes
    -----
    Checks if benchmark is a standard name, valid ticker, or custom function.
    """
```

##### run_backtest

```python
def run_backtest(
    self,
    strategy,
    shift_signals: bool = True,
    benchmark: Union[str, Callable, None] = "equal_weight",
    start_date=None,
    end_date=None,
    require_all_history: bool = False,
    require_all_tickers: bool = False,
    verbose: bool = False
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Execute a backtest using the provided strategy.
    
    Parameters
    ----------
    strategy : object
        Strategy object that must implement a `step` method.
    shift_signals : bool, default=True
        Whether to apply signals on the next day (prevents lookahead bias).
    benchmark : str, callable, or None, default="equal_weight"
        Benchmark for comparison. Options:
        - String: "equal_weight", "markowitz", or ticker symbol
        - Callable: Custom benchmark function
        - None: No benchmark
    start_date : datetime or str, optional
        Start date for backtest.
    end_date : datetime or str, optional
        End date for backtest.
    require_all_history : bool, default=False
        Require all tickers to have data from the same start date.
    require_all_tickers : bool, default=False
        Require data for all requested tickers.
    verbose : bool, default=False
        Show progress bars during execution.
        
    Returns
    -------
    Dict[str, pd.DataFrame] or None
        Dictionary containing backtest results:
        - 'signals_df': Strategy allocations over time
        - 'tickers_returns': Individual asset returns
        - 'strategy_returns': Strategy performance
        - 'benchmark_returns': Benchmark performance
        
    Raises
    ------
    ValueError
        If start_date > end_date or no trading dates after filtering.
    InvalidBenchmarkError
        If benchmark is invalid.
        
    Notes
    -----
    The strategy object must have:
    - A `tickers` attribute containing the list of ticker symbols
    - A `step(timestamp, bar_data)` method that returns allocation weights
    
    Examples
    --------
    >>> # Basic backtest
    >>> results = backtester.run_backtest(
    ...     strategy=my_strategy,
    ...     benchmark='SPY',
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31'
    ... )
    >>> 
    >>> # With custom benchmark
    >>> def custom_benchmark(returns_df):
    ...     return returns_df.mean(axis=1)
    >>> 
    >>> results = backtester.run_backtest(
    ...     strategy=my_strategy,
    ...     benchmark=custom_benchmark,
    ...     verbose=True
    ... )
    """
```

##### _bar_dict

```python
@staticmethod
def _bar_dict(ts: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, dict | None]:
    """
    Create bar data dictionary for a specific timestamp.
    
    Parameters
    ----------
    ts : pd.Timestamp
        Timestamp for which to create bar data.
    data : Dict[str, pd.DataFrame]
        Dictionary mapping ticker symbols to their price data.
        
    Returns
    -------
    Dict[str, dict | None]
        Dictionary mapping ticker symbols to bar data or None if no data available.
        
    Notes
    -----
    Bar data includes open, high, low, close, and volume for each ticker.
    Returns None for tickers without data at the specified timestamp.
    """
```

## Data Structures

### Bar Data Format

The bar data passed to strategy step methods has the following structure:

```python
{
    "ticker_symbol": {
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float
    }
}
```

### Strategy Step Method

Strategies must implement a `step` method with the following signature:

```python
def step(self, timestamp: pd.Timestamp, bar_data: Dict[str, dict]) -> Dict[str, float]:
    """
    Generate allocation weights for the current timestamp.
    
    Parameters
    ----------
    timestamp : pd.Timestamp
        Current timestamp.
    bar_data : Dict[str, dict]
        Bar data for all tickers.
        
    Returns
    -------
    Dict[str, float]
        Allocation weights for each ticker (should sum to 1.0).
    """
```

### Backtest Results

The `run_backtest` method returns a dictionary with the following structure:

```python
{
    "signals_df": pd.DataFrame,        # Strategy allocations over time
    "tickers_returns": pd.DataFrame,   # Individual asset returns
    "strategy_returns": pd.Series,     # Strategy performance
    "benchmark_returns": pd.Series     # Benchmark performance
}
```

## Error Handling

### Common Exceptions

- **`InvalidBenchmarkError`**: Raised when an invalid benchmark is provided
- **`ValueError`**: Raised for invalid date ranges or missing data requirements
- **`KeyError`**: May be raised when accessing ticker data

### Data Validation

The backtester performs several validation checks:

1. **Date range validation**: Ensures start_date ≤ end_date
2. **Data availability**: Checks for missing tickers based on `require_all_tickers`
3. **History requirements**: Validates data availability based on `require_all_history`
4. **Trading calendar**: Ensures valid trading dates when calendar is provided

## Performance Considerations

- **Memory usage**: Large datasets may require significant memory
- **Processing time**: Complex strategies or long time periods increase computation time
- **Data validation**: Use `require_all_tickers=True` for strict data requirements
- **Progress tracking**: Enable `verbose=True` for long-running backtests

## Best Practices

1. **Always use `shift_signals=True`** to prevent lookahead bias
2. **Validate your data** before running backtests
3. **Use appropriate benchmarks** for meaningful comparisons
4. **Handle missing data** gracefully in your strategies
5. **Test with small datasets** before running large backtests
6. **Use trading calendars** for realistic backtesting scenarios 