# Welcome to Portwine

![The Triumph of Bacchus](imgs/header.jpg)

Portwine is a clean, elegant portfolio backtester that makes strategy development and testing simple and intuitive.

## What is Portwine?

Portfolio construction, optimization, and backtesting can be a complicated web of data wrangling, signal generation, lookahead bias reduction, and parameter tuning.

But with `portwine`, strategies are clear and written in an 'online' fashion that removes most of the complexity that comes with backtesting, analyzing, and deploying your trading strategies.

## Key Features

### 🎯 Simple Strategies
Strategies are only given the last day of prices to make their determinations and allocate weights. This allows them to be completely encapsulated and portable.

### ⚡ Breezy Backtesting
Backtesting strategies is a breeze. Simply tell the backtester where your data is located with a data loader manager and give it a strategy. You get results immediately.

### 📊 Streamlined Data
Managing data can be a massive pain. But as long as you have your daily flat files from EODHD or Polygon saved in a directory, the data loaders will manage the rest.

### 📈 Effortless Analysis
After running a strategy through the backtester, put it through an array of analyzers that are simple, visual, and clear.

## Quick Start

```bash
pip install portwine
```

```python
from portwine import SimpleMomentumStrategy, Backtester, EODHDMarketDataLoader

# Define your universe
universe = ['MTUM', 'VTV', 'VUG', 'IJR', 'MDY']

# Create a strategy
strategy = SimpleMomentumStrategy(tickers=universe, lookback_days=10)

# Set up data and backtester
data_loader = EODHDMarketDataLoader(data_path='path/to/your/data/')
backtester = Backtester(market_data_loader=data_loader)

# Run backtest
results = backtester.run_backtest(strategy, benchmark_ticker='SPY')
```

## What's Next?

- **[Installation](getting-started/installation.md)** - Get portwine up and running
- **[Quick Start](getting-started/quick-start.md)** - Your first strategy in minutes
- **[User Guide](user-guide/strategies.md)** - Learn how to build strategies
- **[API Reference](api/strategies.md)** - Complete API documentation
- **[Examples](examples/basic-strategies.md)** - See portwine in action

## Contributing

We welcome contributions! See our [Contributing Guide](contributing.md) for details on how to get started. 