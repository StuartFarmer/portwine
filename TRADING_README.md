# Trading Execution with Portwine

This document explains how to configure and run trading strategies using the DailyExecutor class in the portwine package.

## Configuration Files

Several example configuration files are provided:

1. **example_config.json** - Standard configuration for daily trading with Alpaca
2. **mock_config.json** - Configuration for testing with mock execution (no real trading)
3. **intraday_config.json** - Configuration for intraday trading with more frequent executions

### Configuration Structure

Each configuration file has the following structure:

```json
{
    "strategy": {
        "class": "path.to.strategy.class",
        "tickers": ["TICKER1", "TICKER2", ...],
        "params": {
            // Strategy-specific parameters
        }
    },
    "execution": {
        "class": "path.to.execution.class",
        "params": {
            // Execution-specific parameters
        }
    },
    "data_loader": {
        "class": "path.to.data_loader.class",
        "params": {
            // Data loader-specific parameters
        }
    },
    "schedule": {
        // Scheduling parameters
    },
    "logging": {
        // Logging parameters
    }
}
```

## Strategy Options

The `SimpleMovingAverageStrategy` provided in the examples has the following parameters:

- `short_window`: The window size for the short moving average (default: 20)
- `long_window`: The window size for the long moving average (default: 50)
- `position_size`: The size of each position as a fraction of portfolio (default: 0.1)

## Execution Options

Two execution implementations are provided:

1. **AlpacaExecution** - For executing trades with the Alpaca API
   - Parameters:
     - `paper_trading`: Whether to use paper trading (default: true)
     - `api_key`: Your Alpaca API key
     - `api_secret`: Your Alpaca API secret

2. **MockExecution** - For mock/simulated trading (testing)
   - Parameters:
     - `initial_cash`: Initial cash amount (default: 100000.0)
     - `fail_symbols`: List of symbols that should fail when executing orders (for testing)

## Data Loader Options

The `AlpacaMarketDataLoader` has the following parameters:

- `api_key`: Your Alpaca API key
- `api_secret`: Your Alpaca API secret
- `timeframe`: Data timeframe (e.g., "1Day", "15Min")
- `limit`: Maximum number of data points to retrieve
- `cache_dir`: Directory for caching market data
- `paper_trading`: Whether to use paper trading API endpoints (default: true)

## Schedule Options

Several scheduling options are available:

- `run_time`: When to run the strategy each day
  - Fixed time (e.g., "15:45")
  - Market event based (e.g., "market_open", "market_close")
  - Offset from market events (e.g., "market_open+30m", "market_close-15m")
- `time_zone`: Timezone for the run time (e.g., "US/Eastern")
- `days`: Days of the week to run (e.g., ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
- `market_hours_only`: Whether to only run during market hours (default: true)
- `intraday`: Intraday scheduling option (e.g., "interval:30m" for every 30 minutes)
- `exchange`: Exchange to use for market hours (default: "NYSE")

## Running the Examples

### Standard Daily Trading

```bash
python example_daily_executor.py --config example_config.json
```

To run once without scheduling:

```bash
python example_daily_executor.py --config example_config.json --run-once
```

### Mock Execution (Testing)

```bash
python run_mock_strategy.py --config mock_config.json
```

To run once without scheduling:

```bash
python run_mock_strategy.py --config mock_config.json --run-once
```

### Intraday Trading

```bash
python example_daily_executor.py --config intraday_config.json
```

## Using the Custom Daily Executor

The `CustomDailyExecutor` class is provided to properly handle initialization of the AlpacaExecution class, ensuring that market_data_loader is correctly passed as a parameter.

```python
from custom_daily_executor import CustomDailyExecutor

executor = CustomDailyExecutor.from_config_file("example_config.json")
executor.initialize()
executor.run_once()  # Or executor.run_scheduled()
```

## API Keys

For the Alpaca execution and data loader to work properly, you need to set your Alpaca API keys in the configuration file or as environment variables:

```bash
export ALPACA_API_KEY=your_api_key
export ALPACA_API_SECRET=your_api_secret
```

Then you can use placeholders in your configuration files:

```json
"api_key": "${ALPACA_API_KEY}",
"api_secret": "${ALPACA_API_SECRET}"
```

## Troubleshooting

If you encounter issues with the AlpacaExecution class, make sure you're using the CustomDailyExecutor which correctly handles passing the market_data_loader to the execution system.

For more detailed information about specific classes, consult the module docstrings and source code. 