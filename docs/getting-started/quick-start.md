# Quick Start

This guide will walk you through creating and running your first strategy with portwine.

## Your First Strategy

Let's create a simple momentum strategy that invests in the best-performing asset from the previous period.

```python
from portwine import SimpleMomentumStrategy, Backtester, EODHDMarketDataLoader

# Define your investment universe
universe = ['MTUM', 'VTV', 'VUG', 'IJR', 'MDY']

# Create a momentum strategy
strategy = SimpleMomentumStrategy(
    tickers=universe, 
    lookback_days=10
)

# Set up your data loader
data_loader = EODHDMarketDataLoader(
    data_path='path/to/your/eodhd/data/'
)

# Create the backtester
backtester = Backtester(market_data_loader=data_loader)

# Run the backtest
results = backtester.run_backtest(
    strategy=strategy,
    benchmark_ticker='SPY',
    start_date='2020-01-01',
    end_date='2023-12-31',
    verbose=True
)
```

## Understanding the Results

The backtest returns a dictionary with several key components:

```python
# Strategy signals over time
signals_df = results['signals_df']

# Individual asset returns
ticker_returns = results['tickers_returns']

# Strategy performance
strategy_returns = results['strategy_returns']

# Benchmark performance
benchmark_returns = results['benchmark_returns']
```

## Analyzing Performance

Portwine comes with built-in analyzers to help you understand your strategy's performance:

```python
from portwine.analyzers import EquityDrawdownAnalyzer, MonteCarloAnalyzer

# Equity and drawdown analysis
EquityDrawdownAnalyzer().plot(results)

# Monte Carlo simulation
MonteCarloAnalyzer().plot(results)
```

## What's Happening Under the Hood

1. **Data Loading**: The data loader fetches historical price data for your universe
2. **Strategy Execution**: Each day, your strategy receives the latest prices and decides allocations
3. **Signal Processing**: Portwine handles the mechanics of applying your signals to the market
4. **Performance Calculation**: Returns are calculated and compared against your benchmark

## Next Steps

- Learn more about [building strategies](user-guide/strategies.md)
- Explore [different analyzers](user-guide/analysis.md)
- Check out [advanced examples](examples/advanced-strategies.md) 