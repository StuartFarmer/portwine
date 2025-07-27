# Universe System Changelog

## Overview

This document outlines the implementation of a dynamic universe management system for handling delisted stocks and historical constituents in the portwine backtesting framework. The system allows strategies to work with rotating universes (like S&P 500 constituents over time) while maintaining backward compatibility.

## New Components

### Universe Class (`portwine/universe.py`)

#### Base Universe Class
The `Universe` class provides efficient lookup of constituents at any given date using binary search on pre-sorted dates.

**Key Features:**
- **Immutable**: Once loaded, the universe cannot be modified
- **Efficient**: Pre-sorted dates enable O(log n) binary search lookups
- **Set-based**: Uses sets internally for optimal performance
- **Flexible**: Handles various date formats and edge cases

**Core Methods:**
- `get_constituents(dt) -> Set[str]`: Get the basket for a given date
- `all_tickers -> set`: Get all unique tickers that have ever been in the universe

**Usage:**
```python
from portwine.universe import Universe
from datetime import date

# Create universe directly
constituents = {
    date(2020, 1, 1): {"AAPL", "GOOGL", "MSFT"},
    date(2020, 2, 1): {"AAPL", "GOOGL", "MSFT", "AMZN"},
}
universe = Universe(constituents)

# Get constituents at specific date
tickers = universe.get_constituents("2020-01-15")  # Returns {"AAPL", "GOOGL", "MSFT"}
all_tickers = universe.all_tickers  # Returns {"AAPL", "GOOGL", "MSFT", "AMZN"}
```

#### CSVUniverse Class
The `CSVUniverse` class loads constituent data from CSV files.

**CSV Format:**
```csv
date,ticker1,ticker2,ticker3,...
2020-01-01,AAPL,GOOGL,MSFT
2020-02-01,AAPL,GOOGL,MSFT,AMZN
2020-03-01,AAPL,MSFT,AMZN,TSLA
```

**Features:**
- Comments: Lines starting with '#' are ignored
- Empty lines: Automatically skipped
- Whitespace: Stripped from tickers
- Invalid dates: Skipped with warning
- Duplicate dates: Last entry wins

**Usage:**
```python
from portwine.universe import CSVUniverse

universe = CSVUniverse("sp500_constituents.csv")
tickers = universe.get_constituents("2020-06-15")
```

## Strategy Base Class Changes (`portwine/strategies/base.py`)

### Unified Interface
The `StrategyBase` class now uses a unified universe interface. All strategies internally use a universe object.

**Key Changes:**
- **List Conversion**: Passing a list of tickers automatically creates a static universe
- **Set-based Tickers**: `strategy.tickers` is now always a set
- **Universe Access**: All strategies have access to `strategy.universe`

**Behavior:**
- **Static Universe**: Lists are converted to static universes with tickers from 1970-01-01 onwards
- **Dynamic Universe**: Universe objects are used directly
- **Backward Compatibility**: Existing strategies continue to work without changes

**Usage Examples:**
```python
# Static universe (from list)
strategy = MyStrategy(["AAPL", "GOOGL", "MSFT"])
# Internally creates: Universe({date(1970, 1, 1): {"AAPL", "GOOGL", "MSFT"}})

# Dynamic universe (from CSV)
universe = CSVUniverse("sp500_constituents.csv")
strategy = MyStrategy(universe)
# Uses the dynamic universe directly
```

## Backtester Changes (`portwine/backtester.py`)

### Universe-Aware Data Loading
The backtester now filters data based on the current universe at each step.

**Key Changes:**
- **Unified Interface**: All strategies use universe objects internally
- **Dynamic Filtering**: Data is filtered at each step based on current universe
- **Validation**: Backtester validates that strategies only assign weights to tickers in current universe
- **Set-based Processing**: `_split_tickers` now accepts sets instead of lists

**Behavior:**
1. **Initial Data Loading**: Loads all possible tickers from `strategy.universe.all_tickers`
2. **Step-by-step Filtering**: At each step, gets current universe with `strategy.universe.get_constituents(ts)`
3. **Data Filtering**: Only passes data for current universe tickers to strategy
4. **Weight Validation**: Fails if strategy tries to assign weights to tickers not in current universe
5. **Automatic Zero-filling**: Sets weights to 0 for tickers not in current universe

**Error Handling:**
```python
# This will raise ValueError:
ValueError: Strategy assigned weights to tickers not in current universe at 2024-01-02 00:00:00: ['MSFT']. Current universe: {'AAPL'}
```

**Performance Optimizations:**
- Binary search for date lookups: O(log n) instead of O(n)
- Pre-sorted dates for efficient searching
- Set-based operations for optimal performance
- Single data loading pass with filtering at each step

## Usage Examples

### Static Universe (Traditional Behavior)
```python
class MyStrategy(StrategyBase):
    def step(self, current_date, daily_data):
        # daily_data contains all tickers from the static universe
        valid_tickers = [t for t in daily_data.keys() if daily_data.get(t) is not None]
        n = len(valid_tickers)
        weight = 1.0 / n if n > 0 else 0.0
        return {ticker: weight for ticker in valid_tickers}

# Create strategy with static universe
strategy = MyStrategy(["AAPL", "GOOGL", "MSFT"])
backtester = Backtester(market_data_loader=data_loader)
results = backtester.run_backtest(strategy=strategy, start_date="2024-01-01", end_date="2024-12-31")
```

### Dynamic Universe (New Behavior)
```python
# Create universe CSV file: sp500.csv
# 2024-01-01,AAPL,GOOGL
# 2024-02-01,AAPL,GOOGL,MSFT
# 2024-03-01,MSFT,AMZN

class DynamicStrategy(StrategyBase):
    def step(self, current_date, daily_data):
        # daily_data only contains tickers currently in the universe
        # On 2024-01-15: daily_data = {"AAPL": {...}, "GOOGL": {...}}
        # On 2024-02-15: daily_data = {"AAPL": {...}, "GOOGL": {...}, "MSFT": {...}}
        valid_tickers = [t for t in daily_data.keys() if daily_data.get(t) is not None]
        n = len(valid_tickers)
        weight = 1.0 / n if n > 0 else 0.0
        return {ticker: weight for ticker in valid_tickers}

# Create strategy with dynamic universe
universe = CSVUniverse("sp500.csv")
strategy = DynamicStrategy(universe)
backtester = Backtester(market_data_loader=data_loader)
results = backtester.run_backtest(strategy=strategy, start_date="2024-01-01", end_date="2024-12-31")
```

## Migration Guide

### For Existing Strategies
No changes required! Existing strategies continue to work exactly as before:

```python
# This still works exactly the same
class MyStrategy(StrategyBase):
    def step(self, current_date, daily_data):
        # Your existing logic works unchanged
        pass

strategy = MyStrategy(["AAPL", "GOOGL", "MSFT"])  # Automatically creates static universe
```

### For New Dynamic Universe Strategies
1. Create a CSV file with your universe data
2. Use `CSVUniverse` to load the data
3. Pass the universe to your strategy

```python
universe = CSVUniverse("my_universe.csv")
strategy = MyStrategy(universe)
```

## Testing

### Unit Tests
- `tests/test_universe.py`: Comprehensive tests for Universe and CSVUniverse classes
- `tests/test_universe_integration.py`: Integration tests for universe with backtester and strategy

### Key Test Cases
- Static universe creation from lists
- Dynamic universe loading from CSV
- Date-based constituent lookup
- Binary search edge cases
- Invalid data handling
- Universe filtering in backtester
- Weight validation
- Backward compatibility

## Performance Characteristics

### Time Complexity
- **Date Lookup**: O(log n) using binary search
- **Data Filtering**: O(k) where k is number of current universe tickers
- **Memory Usage**: O(n) where n is total number of unique tickers

### Memory Efficiency
- Sets for optimal membership testing
- Pre-computed `all_tickers` for efficient data loading
- Single data loading pass with filtering at each step

## Future Enhancements

### Potential Improvements
- **Caching**: Cache frequently accessed date ranges
- **Compression**: Compress large universe datasets
- **Streaming**: Support for streaming universe updates
- **Validation**: Additional validation for universe data integrity
- **Analytics**: Universe change analytics and reporting

### API Extensions
- **Universe Comparison**: Compare universes across dates
- **Change Detection**: Detect additions/removals between dates
- **Universe Merging**: Merge multiple universe sources
- **Universe Validation**: Validate universe data quality

## Breaking Changes

### None
This implementation maintains full backward compatibility. All existing code continues to work without modification.

## Dependencies

### New Dependencies
- None (uses only standard library)

### Updated Dependencies
- None

## Version Compatibility

- **Python**: 3.8+
- **pandas**: No changes required
- **numpy**: No changes required
- **Other dependencies**: No changes required 