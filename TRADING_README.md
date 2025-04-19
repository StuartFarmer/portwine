# Trading Execution with Portwine

This document explains how to configure and run trading strategies using Portwine's execution framework.

## New Scheduling Approach

The scheduling functionality has been integrated directly into the `ExecutionBase` class, eliminating the need for the separate `DailyExecutor` class. The new approach is simpler and more flexible:

```python
from portwine.execution_complex.base import ExecutionBase
from portwine.utils.schedule_iterator import DailyMarketScheduleIterator

# Create your execution system
execution = ExecutionBase(strategy, data_loader, broker)

# Create a schedule iterator
schedule = DailyMarketScheduleIterator(
    exchange="NYSE",
    minutes_before_close=15,
    timezone="America/New_York"
)

# Run with the schedule
execution.run(schedule)
```

This approach allows for more flexibility in scheduling and better integration with the execution system. The `ExecutionBase.run()` method:

1. Takes a `ScheduleIterator` that determines when to execute trades
2. Enters an infinite loop, waiting for the next scheduled time
3. Executes the trading step at each scheduled time
4. Can be stopped with Ctrl+C or by calling `execution.stop()`

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

Two broker implementations are provided:

1. **AlpacaBroker** - For executing trades with the Alpaca API
   - Parameters:
     - `paper_trading`: Whether to use paper trading (default: true)
     - `api_key`: Your Alpaca API key
     - `api_secret`: Your Alpaca API secret

2. **MockBroker** - For mock/simulated trading (testing)
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

Several scheduling options are available via the `ScheduleIterator` classes:

1. **DailyMarketScheduleIterator** - For scheduling execution at a specific time relative to market close
   - Parameters:
     - `exchange`: Exchange to use for market hours (default: "NYSE")
     - `minutes_before_close`: Number of minutes before market close to execute (default: 15)
     - `timezone`: Timezone for execution times (default: UTC)
     - `start_date`: Optional start date (default: current time)

More schedule iterators can be created by implementing the `ScheduleIterator` abstract base class.

## Running the Examples

The new example script demonstrates how to use the integrated scheduling approach:

```bash
python examples/run_execution_with_schedule.py --config example_config.json
```

To run once without scheduling:

```bash
python examples/run_execution_with_schedule.py --config example_config.json --run-once
```

To run for a limited number of iterations:

```bash
python examples/run_execution_with_schedule.py --config example_config.json --iterations 3
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

## Creating Custom Schedules

You can create custom scheduling by implementing your own `ScheduleIterator`. For example:

```python
from portwine.utils.schedule_iterator import ScheduleIterator
import pandas as pd

class HourlyScheduleIterator(ScheduleIterator):
    """Iterator that yields times on the hour, every hour during market hours."""
    
    def __next__(self) -> pd.Timestamp:
        # Calculate next hour
        next_time = self.current_time.floor('H') + pd.Timedelta(hours=1)
        self.current_time = next_time
        return next_time
```

## Troubleshooting

If you encounter issues:

1. Check your API keys and make sure they have the necessary permissions
2. Verify that your configuration file is correctly formatted JSON
3. Look at the log file for detailed error messages
4. Ensure that you're using compatible versions of all dependencies

For more detailed information about specific classes, consult the module docstrings and source code. 