# Backtester API

The `Backtester` class is the core component of portwine that executes strategies and generates performance results.

## Class Definition

```python
class Backtester:
    """
    A step‑driven back‑tester that supports intraday bars and,
    optionally, an exchange trading calendar.
    """
```

## Constructor

```python
def __init__(
    self,
    market_data_loader: MarketDataLoader,
    alternative_data_loader=None,
    calendar: Optional[Union[str, mcal.ExchangeCalendar]] = None
):
```

### Parameters

- **`market_data_loader`** (`MarketDataLoader`): The primary data loader for market data
- **`alternative_data_loader`** (optional): Additional data loader for alternative data sources
- **`calendar`** (optional): Trading calendar for date filtering. Can be a string (calendar name) or `mcal.ExchangeCalendar` object

### Example

```python
from portwine import Backtester, EODHDMarketDataLoader
import pandas_market_calendars as mcal

# Basic backtester
data_loader = EODHDMarketDataLoader(data_path='path/to/data/')
backtester = Backtester(market_data_loader=data_loader)

# With trading calendar
calendar = mcal.get_calendar('NYSE')
backtester = Backtester(
    market_data_loader=data_loader,
    calendar=calendar
)
```

## Main Method: run_backtest

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
```

### Parameters

- **`strategy`**: Your strategy object (must implement `step` method)
- **`shift_signals`** (bool, default: `True`): Whether to apply signals on the next day (prevents lookahead bias)
- **`benchmark`** (str, callable, or None, default: `"equal_weight"`): Benchmark for comparison
  - String options: `"equal_weight"`, `"markowitz"`
  - Custom callable function
  - Ticker symbol (e.g., `"SPY"`)
- **`start_date`** (optional): Start date for backtest (datetime or string)
- **`end_date`** (optional): End date for backtest (datetime or string)
- **`require_all_history`** (bool, default: `False`): Require all tickers to have data from the same start date
- **`require_all_tickers`** (bool, default: `False`): Require data for all requested tickers
- **`verbose`** (bool, default: `False`): Show progress bars

### Returns

A dictionary containing:

- **`signals_df`** (`pd.DataFrame`): Strategy allocations over time
- **`tickers_returns`** (`pd.DataFrame`): Individual asset returns
- **`strategy_returns`** (`pd.Series`): Strategy performance
- **`benchmark_returns`** (`pd.Series`): Benchmark performance

### Example

```python
# Basic backtest
results = backtester.run_backtest(
    strategy=my_strategy,
    benchmark_ticker='SPY',
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# With custom benchmark
def custom_benchmark(returns_df):
    return returns_df.mean(axis=1)

results = backtester.run_backtest(
    strategy=my_strategy,
    benchmark=custom_benchmark,
    verbose=True
)
```

## Built-in Benchmarks

### Equal Weight Benchmark

```python
def benchmark_equal_weight(ret_df: pd.DataFrame, *_, **__) -> pd.Series:
    return ret_df.mean(axis=1)
```

### Markowitz Benchmark

```python
def benchmark_markowitz(
    ret_df: pd.DataFrame,
    lookback: int = 60,
    shift_signals: bool = True,
    verbose: bool = False,
) -> pd.Series:
```

Uses mean-variance optimization to find optimal weights.

## Benchmark Types

The backtester supports three types of benchmarks:

```python
class BenchmarkTypes:
    STANDARD_BENCHMARK = 0  # Built-in benchmarks
    TICKER             = 1  # Single ticker
    CUSTOM_METHOD      = 2  # Custom function
    INVALID            = 3  # Invalid benchmark
```

## Error Handling

### InvalidBenchmarkError

Raised when the requested benchmark is neither a standard name nor a valid ticker:

```python
class InvalidBenchmarkError(Exception):
    """Raised when the requested benchmark is neither a standard name nor a valid ticker."""
    pass
```

### Common Errors

- **No trading dates**: Raised when date filtering results in no valid trading dates
- **Missing tickers**: Warning or error when data is missing for requested tickers
- **Invalid date range**: Raised when start_date > end_date

## Advanced Features

### Alternative Data Support

The backtester can integrate alternative data sources:

```python
# Alternative data loader
alt_loader = AlternativeDataLoader()

backtester = Backtester(
    market_data_loader=market_loader,
    alternative_data_loader=alt_loader
)
```

### Trading Calendar Integration

```python
import pandas_market_calendars as mcal

# Use NYSE calendar
calendar = mcal.get_calendar('NYSE')
backtester = Backtester(
    market_data_loader=data_loader,
    calendar=calendar
)
```

### Signal Shifting

By default, signals are applied on the next trading day to prevent lookahead bias:

```python
# Apply signals immediately (not recommended)
results = backtester.run_backtest(
    strategy=my_strategy,
    shift_signals=False
)
```

## Performance Considerations

- **Memory usage**: Large datasets may require significant memory
- **Processing time**: Complex strategies or long time periods increase computation time
- **Data validation**: Use `require_all_tickers=True` for strict data requirements

## Best Practices

1. **Always use `shift_signals=True`** to prevent lookahead bias
2. **Validate your data** before running backtests
3. **Use appropriate benchmarks** for meaningful comparisons
4. **Handle missing data** gracefully in your strategies
5. **Test with small datasets** before running large backtests 