# Backtesting

Backtesting is the process of testing a trading strategy on historical data to evaluate its performance. Portwine makes this process simple and intuitive.

## Basic Backtesting

### Setting Up a Backtest

```python
from portwine import Backtester, SimpleMomentumStrategy, EODHDMarketDataLoader

# 1. Define your strategy
strategy = SimpleMomentumStrategy(
    tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
    lookback_days=20
)

# 2. Set up data loader
data_loader = EODHDMarketDataLoader(data_path='path/to/your/data/')

# 3. Create backtester
backtester = Backtester(market_data_loader=data_loader)

# 4. Run backtest
results = backtester.run_backtest(
    strategy=strategy,
    benchmark_ticker='SPY',
    start_date='2020-01-01',
    end_date='2023-12-31',
    verbose=True
)
```

## Understanding Results

The backtest returns a dictionary with four key components:

```python
# Strategy allocations over time
signals_df = results['signals_df']
print(signals_df.head())
# Output:
#            AAPL  GOOGL  MSFT  AMZN
# 2020-01-02   0.0    0.0   0.0   1.0
# 2020-01-03   0.0    0.0   0.0   1.0
# 2020-01-06   0.0    0.0   0.0   1.0

# Individual asset returns
ticker_returns = results['tickers_returns']
print(ticker_returns.head())
# Output:
#            AAPL     GOOGL     MSFT     AMZN
# 2020-01-02  0.0123   0.0089   0.0156   0.0234
# 2020-01-03 -0.0056   0.0123  -0.0034   0.0189

# Strategy performance
strategy_returns = results['strategy_returns']
print(strategy_returns.head())
# Output:
# 2020-01-02    0.0234
# 2020-01-03    0.0189
# 2020-01-06    0.0156

# Benchmark performance
benchmark_returns = results['benchmark_returns']
```

## Key Parameters

### Date Range

```python
# Specific date range
results = backtester.run_backtest(
    strategy=strategy,
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# No date restrictions (uses all available data)
results = backtester.run_backtest(strategy=strategy)
```

### Benchmarks

```python
# Built-in benchmarks
results = backtester.run_backtest(
    strategy=strategy,
    benchmark="equal_weight"  # Equal weight portfolio
)

results = backtester.run_backtest(
    strategy=strategy,
    benchmark="markowitz"     # Mean-variance optimized
)

# Single ticker benchmark
results = backtester.run_backtest(
    strategy=strategy,
    benchmark="SPY"
)

# Custom benchmark function
def custom_benchmark(returns_df):
    """Custom benchmark that weights by market cap"""
    # Your custom logic here
    return returns_df.mean(axis=1)

results = backtester.run_backtest(
    strategy=strategy,
    benchmark=custom_benchmark
)
```

### Signal Timing

```python
# Default: signals applied next day (prevents lookahead bias)
results = backtester.run_backtest(
    strategy=strategy,
    shift_signals=True
)

# Signals applied same day (not recommended)
results = backtester.run_backtest(
    strategy=strategy,
    shift_signals=False # Will most likely be deprecated in the future, as this is not recommended ever.
)
```

## Data Requirements

### Requiring All Tickers

```python
# Error if any ticker is missing data
results = backtester.run_backtest(
    strategy=strategy,
    require_all_tickers=True
)

# Warning if tickers are missing (default)
results = backtester.run_backtest(
    strategy=strategy,
    require_all_tickers=False
)
```

### Requiring Full History

```python
# Only use dates where all tickers have data
results = backtester.run_backtest(
    strategy=strategy,
    require_all_history=True
)
```

## Trading Calendars

### Using Exchange Calendars

```python
import pandas_market_calendars as mcal

# NYSE calendar
calendar = mcal.get_calendar('NYSE')
backtester = Backtester(
    market_data_loader=data_loader,
    calendar=calendar
)

# NASDAQ calendar
calendar = mcal.get_calendar('NASDAQ')
backtester = Backtester(
    market_data_loader=data_loader,
    calendar=calendar
)
```

### Calendar Benefits

- **Accurate trading days**: Only uses actual trading days
- **Holiday handling**: Automatically excludes market holidays
- **Time zone support**: Handles different exchange time zones

## Alternative Data

### Adding Alternative Data Sources

```python
from portwine import AlternativeDataLoader

# Set up alternative data loader
alt_loader = AlternativeDataLoader()

# Create backtester with alternative data
backtester = Backtester(
    market_data_loader=market_loader,
    alternative_data_loader=alt_loader
)

# Strategy can now access alternative data
class AltDataStrategy(StrategyBase):
    def step(self, current_date, daily_data):
        # Access alternative data
        if 'alt:sentiment' in daily_data:
            sentiment = daily_data['alt:sentiment']
            # Use sentiment in strategy logic
```

## Performance Analysis

### Basic Performance Metrics

You can now use the results to calculate any metrics you'd like:

```python
import pandas as pd
import numpy as np

# Calculate cumulative returns
cumulative_returns = (1 + results['strategy_returns']).cumprod(axis=0)  # axis=0 for time series
benchmark_cumulative = (1 + results['benchmark_returns']).cumprod(axis=0)

# Calculate annualized return
annual_return = results['strategy_returns'].mean() * 252

# Calculate volatility
volatility = results['strategy_returns'].std() * np.sqrt(252)

# Calculate Sharpe ratio
risk_free_rate = 0.02  # 2% annual
sharpe_ratio = (annual_return - risk_free_rate) / volatility

# Calculate maximum drawdown
cumulative = (1 + results['strategy_returns']).cumprod(axis=0)  # axis=0 for time series
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()
```

### Using Built-in Analyzers

However, you'd probably prefer the prebuilt analyzers that offer raw analysis (output as Dataframes in most cases) and visual plotting:

```python
from portwine.analyzers import (
    EquityDrawdownAnalyzer,
    MonteCarloAnalyzer,
    SeasonalityAnalyzer
)

# Equity and drawdown analysis
EquityDrawdownAnalyzer().plot(results)

# Monte Carlo simulation
MonteCarloAnalyzer().plot(results)

# Seasonality analysis
SeasonalityAnalyzer().plot(results)
```

## Best Practices

### 1. Prevent Lookahead Bias

Always use `shift_signals=True` (default):

```python
# ✅ Good
results = backtester.run_backtest(strategy=strategy, shift_signals=True)

# ❌ Bad - introduces lookahead bias
results = backtester.run_backtest(strategy=strategy, shift_signals=False)
```

### 2. Use Appropriate Benchmarks

```python
# For equity strategies
results = backtester.run_backtest(strategy=strategy, benchmark="SPY")

# For multi-asset strategies
results = backtester.run_backtest(strategy=strategy, benchmark="equal_weight")

# For factor strategies
results = backtester.run_backtest(strategy=strategy, benchmark="markowitz")
```

## Next Steps

- Learn about [performance analysis](analysis.md)
- Explore [data management](data-management.md)
- Check out [advanced strategies](examples/advanced-strategies.md)
