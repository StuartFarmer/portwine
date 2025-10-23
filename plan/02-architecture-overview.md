# Data Refactor - Architecture Overview

**Branch**: `loader` (28 commits ahead of main)
**Status**: 80-85% Complete
**Files Changed**: 78 files (additions/deletions/modifications)

---

## High-Level Architecture

### Old System (Deprecated)

```
┌─────────────────────────────────────┐
│   Monolithic Loader Classes         │
│   (portwine/loaders/)               │
│                                     │
│   - AlpacaLoader                    │
│   - EODHDLoader                     │
│   - FREDLoader                      │
│   - PolygonLoader                   │
│   - BrokerDataLoader                │
└──────────┬──────────────────────────┘
           │ Direct coupling
           ▼
┌─────────────────────────────────────┐
│        Backtester                   │
└─────────────────────────────────────┘
```

**Problems with old system**:
- Tight coupling between data source and backtester
- No abstraction for different storage backends
- Hard to add alternative data sources
- Limited flexibility for multi-source strategies

---

### New System (Current Refactor)

```
┌──────────────────────────────────────────────────────────┐
│                   Data Providers                         │
│              (portwine/data/providers/)                  │
│                                                          │
│  AlpacaProvider  EODHDProvider  FREDProvider  PolygonProvider
│        │              │              │              │
│        └──────────────┴──────────────┴──────────────┘
│                           │
│                           ▼
└───────────────────────────┼──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                    Data Stores                           │
│               (portwine/data/stores/)                    │
│                                                          │
│  CSVDataStore    ParquetDataStore    NoisyDataStore     │
│  (files)         (columnar)          (decorator)         │
│                                                          │
│  + MarketDataLoaderAdapter (legacy loader → store)      │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                  Data Interfaces                         │
│              (portwine/data/interface.py)                │
│                                                          │
│  • DataInterface          - Single source                │
│  • MultiDataInterface     - Multiple prefixed sources    │
│  • RestrictedDataInterface- Filtered ticker access       │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│              Backtester & Strategy                       │
│         (portwine/backtester/core.py)                    │
└──────────────────────────────────────────────────────────┘
```

**Benefits of new system**:
- Clean separation of concerns (source / storage / access)
- Pluggable storage backends (CSV, Parquet, future: databases)
- Support for multiple data sources in single strategy
- Backward compatible via adapter pattern
- Testability (NoisyDataStore decorator for robustness testing)

---

## Component Details

### 1. Data Providers Layer

**Purpose**: Abstract data source acquisition

**Location**: [portwine/data/providers/](../portwine/data/providers/)

**Key Files**:
- [base.py](../portwine/data/providers/base.py) - `DataProvider` abstract base class
- [alpaca.py](../portwine/data/providers/alpaca.py) - Alpaca market data
- [eodhd.py](../portwine/data/providers/eodhd.py) - EODHD market data
- [fred.py](../portwine/data/providers/fred.py) - Federal Reserve economic data
- [polygon.py](../portwine/data/providers/polygon.py) - Polygon.io market data
- [loader_adapters.py](../portwine/data/providers/loader_adapters.py) - Legacy compatibility

**Interface**:
```python
class DataProvider(ABC):
    @abstractmethod
    def fetch(self, identifier: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data for an identifier within date range."""
        pass
```

**Backward Compatibility**:
- `ProviderBasedLoader` - Base class adapting providers to old loader interface
- `AlpacaMarketDataLoader`, `EODHDMarketDataLoader`, etc. - Drop-in replacements
- Legacy imports show deprecation warnings but still work

---

### 2. Data Stores Layer

**Purpose**: Abstract data persistence and retrieval

**Location**: [portwine/data/stores/](../portwine/data/stores/)

**Key Files**:
- [base.py](../portwine/data/stores/base.py) - `DataStore` abstract base class
- [csvstore.py](../portwine/data/stores/csvstore.py) - File-based CSV storage
- [parquet.py](../portwine/data/stores/parquet.py) - Columnar Parquet storage
- [noisy.py](../portwine/data/stores/noisy.py) - Decorator for noise injection
- [adapter.py](../portwine/data/stores/adapter.py) - `MarketDataLoaderAdapter` (loader → store)

**Interface**:
```python
class DataStore(ABC):
    @abstractmethod
    def get(self, identifier: str, dt: datetime) -> Optional[Dict]:
        """Get data for identifier at specific datetime."""
        pass

    @abstractmethod
    def get_range(self, identifier: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Get data for identifier over date range."""
        pass

    @abstractmethod
    def add(self, identifier: str, data: pd.DataFrame):
        """Add/update data for identifier."""
        pass

    # TODO: Missing in current implementation!
    # def earliest(self, identifier: str) -> Optional[datetime]:
    #     """Get earliest available date for identifier."""
    #     pass
```

**Store Types**:

1. **CSVDataStore**
   - Stores each identifier as separate CSV file
   - Simple, human-readable
   - Good for small-medium datasets
   - File: `{base_dir}/{identifier}.csv`

2. **ParquetDataStore**
   - Columnar storage format
   - Excellent compression
   - Fast read performance for column subsets
   - File: `{base_dir}/{identifier}.parquet`

3. **NoisyDataStore** (Decorator)
   - Wraps any DataStore
   - Injects realistic market noise for testing
   - Tests strategy robustness

4. **MarketDataLoaderAdapter**
   - Wraps legacy loader classes
   - Implements DataStore interface
   - Enables old loaders to work with new system

---

### 3. Data Interfaces Layer

**Purpose**: High-level data access patterns for backtester

**Location**: [portwine/data/interface.py](../portwine/data/interface.py)

**Interface Types**:

#### DataInterface
Single data source access.

```python
class DataInterface:
    def __init__(self, data_loader: DataStore):
        self.data_loader = data_loader

    def __getitem__(self, ticker: str) -> Dict:
        """Access data at current timestamp: data[ticker]"""
        pass

    def earliest_any_date(self, tickers: List[str]) -> datetime:
        """Earliest date where ANY ticker has data."""
        pass

    def earliest_common_date(self, tickers: List[str]) -> datetime:
        """Earliest date where ALL tickers have data."""
        pass
```

**Usage**:
```python
data = DataInterface(ParquetDataStore('/data/market'))
backtester = Backtester(data)
```

#### MultiDataInterface
Multiple prefixed data sources (e.g., stocks + economic data).

```python
class MultiDataInterface(DataInterface):
    def __init__(self, loaders: Dict[Optional[str], DataStore]):
        # None key = default/market data
        # "ECON:" = economic data prefix
        # "INDEX:" = index data prefix
        self.loaders = loaders
```

**Usage**:
```python
data = MultiDataInterface({
    None: ParquetDataStore('/data/market'),      # SPY, AAPL access
    "ECON": CSVDataStore('/data/fred'),          # ECON:GDP, ECON:CPI
    "INDEX": MarketDataLoaderAdapter(loader),    # INDEX:VIX
})
backtester = Backtester(data)

# In strategy:
spy_price = data['SPY']['close']
gdp = data['ECON:GDP']['value']
```

#### RestrictedDataInterface
Filtered access to specific tickers (used internally by backtester).

```python
class RestrictedDataInterface(MultiDataInterface):
    def restrict(self, tickers: Set[str]):
        """Limit access to only specified tickers."""
        pass
```

**Purpose**: Prevent lookahead bias by restricting strategy to current universe.

---

## Data Flow Example

### Scenario: Backtest a strategy using Alpaca data

```python
# 1. Create provider
provider = AlpacaProvider(api_key='xxx', secret='yyy')

# 2. Create store with provider
store = ParquetDataStore('/path/to/data', provider=provider)

# 3. Populate store (one-time)
store.add('SPY', provider.fetch('SPY', '2020-01-01', '2023-12-31'))
store.add('AAPL', provider.fetch('AAPL', '2020-01-01', '2023-12-31'))

# 4. Create data interface
data = DataInterface(store)

# 5. Run backtest
backtester = Backtester(data)
results = backtester.run(strategy, tickers=['SPY', 'AAPL'],
                         start_date='2021-01-01', end_date='2022-12-31')
```

### What happens during backtest:

```
Day 1 (2021-01-01):
  Backtester → data['SPY'] → store.get('SPY', datetime(2021,1,1))
               → returns {'open': 373.12, 'close': 373.88, ...}

  Strategy computes signals: {'SPY': 0.5, 'AAPL': 0.5}

Day 2 (2021-01-04):
  Backtester → data['SPY'] → store.get('SPY', datetime(2021,1,4))
  Backtester → data['AAPL'] → store.get('AAPL', datetime(2021,1,4))

  Strategy computes signals based on new data

... continues for each trading day
```

---

## Backward Compatibility Strategy

### Legacy Code (Still Works)

```python
# Old way - still works with deprecation warning
from portwine.loaders import MarketDataLoader, AlpacaLoader

loader = AlpacaLoader(api_key='xxx', secret='yyy')
# DeprecationWarning: Use AlpacaMarketDataLoader from portwine.data.providers.loader_adapters
```

### Migration Path

```python
# New way - recommended
from portwine.data.providers.loader_adapters import AlpacaMarketDataLoader
from portwine.data.stores import ParquetDataStore
from portwine.data.interface import DataInterface

loader = AlpacaMarketDataLoader(api_key='xxx', secret='yyy')
store = ParquetDataStore('/data', provider=loader)
data = DataInterface(store)
```

### Adapter Pattern

The system maintains compatibility through:

1. **ProviderBasedLoader** - Old loader interface → new provider
2. **MarketDataLoaderAdapter** - Old loader → new DataStore interface
3. **Deprecation warnings** - Guide users to new API
4. **Example code** - [loader_migration_example.py](../examples/loader_migration_example.py)

---

## Key Design Patterns Used

### 1. Strategy Pattern
Different data stores implement common `DataStore` interface.

### 2. Adapter Pattern
- `MarketDataLoaderAdapter`: Legacy loaders → DataStore
- `ProviderBasedLoader`: New providers → old loader interface

### 3. Decorator Pattern
`NoisyDataStore` wraps any store to add noise injection.

### 4. Facade Pattern
`DataInterface` provides simple unified access to complex store/provider system.

### 5. Factory Pattern
Loaders create appropriate providers based on configuration.

---

## Testing Architecture

### Test Organization

**Store Tests**:
- [tests/test_stores.py](../tests/test_stores.py) - CSVStore, ParquetStore
- [tests/test_parquet_data_store.py](../tests/test_parquet_data_store.py) - Comprehensive Parquet tests

**Adapter Tests**:
- [tests/test_adapter.py](../tests/test_adapter.py) - MarketDataLoaderAdapter
- [tests/test_loader_adapters_compatibility.py](../tests/test_loader_adapters_compatibility.py) - Legacy compatibility

**Interface Tests**:
- [tests/test_multidata_interface.py](../tests/test_multidata_interface.py) - Multi-source access

**Integration Tests**:
- [tests/test_backtester_integration.py](../tests/test_backtester_integration.py) - End-to-end

### Test Status
- Most unit tests passing ("all tests pass" in commit 851e9d5)
- Integration tests need verification after blockers fixed
- Performance tests exist in [tests/performance_decorator.py](../tests/performance_decorator.py)

---

## Performance Optimizations

### Backtester Core
The refactor coincided with major backtester performance work:

- **Vectorized operations** using NumPy arrays
- **Numba JIT compilation** for returns calculation
- **Commit 521732d**: "10x performance speedup!!!"

### Data Access
- Parquet columnar format for fast column reads
- Lazy loading (only load data when accessed)
- Caching at interface level (when needed)

---

## Documentation

**User Guide**:
- [docs/user-guide/data-management.md](../docs/user-guide/data-management.md) - Complete guide

**Examples**:
- [examples/loader_migration_example.py](../examples/loader_migration_example.py) - Migration tutorial

**Performance**:
- [docs/performance_optimization_guide.md](../docs/performance_optimization_guide.md) - Optimization techniques

---

## Files Added/Modified Summary

### New Directories
- `portwine/data/` - Entire new data layer
- `portwine/backtester/` - Refactored from single file

### Removed Files
- `portwine/backtester.py` - Moved to `portwine/backtester/core.py`
- Legacy loader files (moved to deprecated with warnings)
- Old example files (outdated workflows)

### Key Modified Files
- `portwine/__init__.py` - New imports
- `portwine/universe.py` - Compatible with new interfaces
- `portwine/strategies/base.py` - Uses new data access
- All test files - Updated to new API

### Total Changes
- **79 files changed**
- **14,708 insertions**
- **11,167 deletions**
- **Net: +3,541 lines**

---

## Migration Timeline

Based on commit history:

1. **Early Phase** (commits 45aae9f - bc7127c)
   - Backtester optimization pt 1
   - Performance speedup work
   - Refactoring groundwork

2. **Core Refactor** (commits ebe8514 - 67930c5)
   - Storage layer
   - Feed system
   - "big one" commit with major changes

3. **Testing Phase** (commits 253ec0f - 5beb334)
   - Test fixes and refactoring
   - Universe integration
   - Full loop testing

4. **Stabilization** (commits 68d4080 - 52bf875)
   - "nearly everything tested"
   - Old backtester deprecated
   - New providers added

5. **Final Polish** (commits 153b435 - 3f6020d)
   - Loader adapters
   - Backward compatible
   - CSV store implementation
   - **Current state**

---

## What Makes This Refactor Good

1. **Clear separation of concerns** - provider / store / interface
2. **Backward compatible** - existing code still works
3. **Extensible** - easy to add new providers/stores
4. **Well-tested** - comprehensive test coverage
5. **Documented** - user guide and examples
6. **Performance-conscious** - designed for speed
7. **Future-proof** - supports alternative data, multi-source strategies

The architecture is solid. Just needs the critical blockers resolved before merge.
