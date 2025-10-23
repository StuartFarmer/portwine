# Data Management

Portwine provides flexible data management through its data loader system, making it easy to work with various data sources and formats. Data loaders are responsible for fetching and providing market data to the backtester.

## Base Market Data Loader

The `MarketDataLoader` is the foundation of Portwine's data system. It provides a standardized interface for loading and accessing market data, with built-in caching and efficient data retrieval.

### Core Functionality

The base loader provides three main capabilities:

1. **Data Loading**: Fetch and cache market data for multiple tickers
2. **Date Management**: Build unified trading calendars across multiple assets
3. **Time-based Access**: Retrieve data at specific timestamps with efficient lookups

### Data Format

All data loaders must return data in a standardized format:

```python
# DataFrame with required columns and index
DataFrame:
    Index: pd.Timestamp (datetime)
    Columns: ['open', 'high', 'low', 'close', 'volume']
    
# Example:
#            open    high     low   close    volume
# 2020-01-02  100.0  102.5   99.0   101.2  1000000
# 2020-01-03  101.2  103.8  100.8   102.9  1200000
# 2020-01-06  102.9  105.2  102.1   104.5  1100000
```

### Key Methods

#### `load_ticker(ticker: str) -> pd.DataFrame | None`

**Purpose**: Load data for a single ticker (must be implemented by subclasses)

**Returns**: 
- `pd.DataFrame` with OHLCV data indexed by timestamp
- `None` if data is unavailable

**Example Implementation**:
```python
def load_ticker(self, ticker: str) -> pd.DataFrame | None:
    # Load from CSV file
    file_path = f"data/{ticker}.csv"
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    return df[['open', 'high', 'low', 'close', 'volume']]
```

---

#### `fetch_data(tickers: list[str]) -> dict[str, pd.DataFrame]`

**Purpose**: Load and cache data for multiple tickers

**Returns**: Dictionary mapping ticker symbols to their DataFrames

**Features**:
- Automatic caching (data loaded once, reused)
- Handles missing data gracefully
- Returns only available tickers

```python
# Usage
loader = MyDataLoader()
data = loader.fetch_data(['AAPL', 'GOOGL', 'MSFT'])

# Returns:
# {
#     'AAPL': DataFrame(...),
#     'GOOGL': DataFrame(...),
#     'MSFT': DataFrame(...)
# }
```

---

#### `get_all_dates(tickers: list[str]) -> list[pd.Timestamp]`

**Purpose**: Build a unified trading calendar across multiple tickers

**Returns**: Sorted list of all unique timestamps

**Use Case**: Creating trading calendars for backtesting

```python
# Get all trading dates across multiple assets
dates = loader.get_all_dates(['AAPL', 'GOOGL', 'MSFT'])
# Returns: [2020-01-02, 2020-01-03, 2020-01-06, ...]
```

---

#### `next(tickers: list[str], ts: pd.Timestamp) -> dict[str, dict[str, float] | None]`

**Purpose**: Get OHLCV data at or immediately before a specific timestamp

**Returns**: Dictionary with ticker data at the requested time

**Features**:
- Efficient binary search using `searchsorted`
- Returns data from the most recent bar before the timestamp
- Handles missing data with `None` values

```python
# Get data at specific time
bar_data = loader.next(['AAPL', 'GOOGL'], pd.Timestamp('2020-01-02 10:30:00'))

# Returns:
# {
#     'AAPL': {
#         'open': 100.0,
#         'high': 102.5,
#         'low': 99.0,
#         'close': 101.2,
#         'volume': 1000000.0
#     },
#     'GOOGL': {
#         'open': 1500.0,
#         'high': 1520.0,
#         'low': 1495.0,
#         'close': 1510.0,
#         'volume': 500000.0
#     }
# }
```

---

### Data Caching

The base loader includes automatic caching to improve performance:

```python
# First call loads from source
data1 = loader.fetch_data(['AAPL'])  # Loads from file/API

# Subsequent calls use cached data
data2 = loader.fetch_data(['AAPL'])  # Uses cache, no I/O
```

### Error Handling

The loader handles missing data gracefully:

```python
# Missing ticker returns empty dict
data = loader.fetch_data(['INVALID_TICKER'])
# Returns: {}

# Missing timestamp returns None
bar = loader.next(['AAPL'], pd.Timestamp('1900-01-01'))
# Returns: {'AAPL': None}
```

### Creating Custom Loaders

To create a custom data loader, inherit from `MarketDataLoader` and implement `load_ticker`:

```python
from portwine.data.providers.loader_adapters import MarketDataLoader
import pandas as pd

class MyCustomLoader(MarketDataLoader):
    def __init__(self, api_key):
        self.api_key = api_key
        super().__init__()
    
    def load_ticker(self, ticker: str) -> pd.DataFrame | None:
        # Your custom data loading logic here
        # Must return DataFrame with ['open', 'high', 'low', 'close', 'volume']
        # or None if data unavailable
        pass
```

### Integration with Backtester

The backtester uses the data loader's methods automatically:

```python
# Backtester calls these methods internally:
loader.fetch_data(strategy.tickers)  # Load all required data
loader.get_all_dates(strategy.tickers)  # Build trading calendar
loader.next(tickers, current_time)  # Get data for each step
```

This design provides a clean separation between data management and strategy execution, making it easy to switch data sources or add new ones without changing the backtesting logic.

## Out-of-the-Box Loaders

Portwine comes with several out of the box loaders for loading saved data from different providers. These are not *downloaders*, which actually fetch the data from the source. You will need one of those prior (coming soon...)

### EODHD Market Data Loader

The EODHD loader reads historical market data from CSV files downloaded from EODHD (End of Day Historical Data). It automatically handles price adjustments for splits and dividends.

```python
from portwine.data.providers.loader_adapters import EODHDMarketDataLoader

# Initialize with your data directory
data_loader = EODHDMarketDataLoader(
    data_path='path/to/your/eodhd/data/',
    exchange_code='US'  # Default is 'US'
)

# Fetch data for specific tickers
data = data_loader.fetch_data(['AAPL', 'GOOGL', 'MSFT'])
```

#### File Structure

CSV files must be named as `TICKER.EXCHANGE.csv` and contain these columns:

```
data_path/
├── AAPL.US.csv
├── GOOGL.US.csv
├── MSFT.US.csv
└── SPY.US.csv
```

#### Required CSV Columns

Each CSV file must contain these columns:
- `date` - Date column (will become the index)
- `open` - Opening price
- `high` - High price
- `low` - Low price
- `close` - Closing price
- `adjusted_close` - Split/dividend adjusted closing price
- `volume` - Trading volume

#### Price Adjustment

The loader automatically adjusts all OHLC prices using the adjusted_close ratio:

```python
# The loader calculates: adj_ratio = adjusted_close / close
# Then applies: open = open * adj_ratio, high = high * adj_ratio, etc.
# Final close = adjusted_close
```

This ensures all prices are adjusted for stock splits and dividends, making them suitable for backtesting.

#### Example CSV Format

```csv
date,open,high,low,close,adjusted_close,volume
2020-01-02,100.0,102.5,99.0,101.2,101.2,1000000
2020-01-03,101.2,103.8,100.8,102.9,102.9,1200000
2020-01-06,102.9,105.2,102.1,104.5,104.5,1100000
```

### Polygon Market Data Loader

For Polygon.io data:

```python
from portwine.data.providers.loader_adapters import PolygonMarketDataLoader

data_loader = PolygonMarketDataLoader(data_path='path/to/polygon/data/')
```

## Data Format Requirements

### OHLCV Data Structure

Portwine expects data in the following format:

```python
# DataFrame with columns: open, high, low, close, volume
data = {
    'AAPL': pd.DataFrame({
        'open': [150.0, 151.0, 152.0],
        'high': [152.0, 153.0, 154.0],
        'low': [149.0, 150.0, 151.0],
        'close': [151.0, 152.0, 153.0],
        'volume': [1000000, 1100000, 1200000]
    }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03']))
}
```

### Data Quality Requirements

- **No missing values**: All OHLCV fields should be present
- **Valid dates**: Index should be datetime objects
- **Sorted index**: Dates should be in ascending order
- **Consistent timezone**: All data should use the same timezone

## Alternative Data

Portwine supports alternative data sources through a specifier system that distinguishes between different types of data. This allows you to combine market data with alternative data sources like sentiment, news, economic indicators, and more.

### Data Source Specifiers

Data sources are identified using a `SOURCE:TICKER` format, where:
- `SOURCE` - Identifies the data provider or type
- `TICKER` - The specific identifier for that data source

```python
# Market data (no specifier needed)
market_tickers = ['AAPL', 'GOOGL', 'MSFT']

# Alternative data (with specifier)
alt_tickers = ['sentiment:AAPL', 'news:GOOGL', 'fred:GDP', 'custom:my_data']
```

### Built-in Alternative Data Loaders

#### FRED Economic Data

Load economic indicators from the Federal Reserve Economic Data (FRED):

```python
from portwine.data.providers.loader_adapters import FREDDataLoader

# Initialize FRED loader
fred_loader = FREDDataLoader(api_key='your_fred_api_key')

# Fetch economic data
data = fred_loader.fetch_data(['fred:GDP', 'fred:UNRATE', 'fred:CPIAUCSL'])

# Returns:
# {
#     'fred:GDP': DataFrame(...),      # Gross Domestic Product
#     'fred:UNRATE': DataFrame(...),   # Unemployment Rate
#     'fred:CPIAUCSL': DataFrame(...)  # Consumer Price Index
# }
```

#### Custom Alternative Data

Create your own alternative data loader:

```python
from portwine.data.providers.loader_adapters import MarketDataLoader

class SentimentDataLoader(MarketDataLoader):
    def __init__(self, data_path):
        self.data_path = data_path
        super().__init__()
    
    def load_ticker(self, ticker):
        # Remove 'sentiment:' prefix to get actual ticker
        base_ticker = ticker.replace('sentiment:', '')
        file_path = f"{self.data_path}/{base_ticker}_sentiment.csv"
        
        if not os.path.exists(file_path):
            return None
        
        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        # Alternative data can have different columns
        return df[['sentiment_score', 'volume']]  # Not OHLCV format

# Use your custom loader
sentiment_loader = SentimentDataLoader('path/to/sentiment/data/')
```

### Combining Market and Alternative Data

The backtester can handle both market data and alternative data simultaneously:

```python
from portwine.backtester import Backtester
from portwine.data.providers.loader_adapters import EODHDMarketDataLoader

# Set up market data loader
market_loader = EODHDMarketDataLoader('path/to/market/data/')

# Set up alternative data loader
alt_loader = SentimentDataLoader('path/to/sentiment/data/')

# Create backtester with both loaders
backtester = Backtester(
    market_data_loader=market_loader,
    alternative_data_loader=alt_loader
)

# Strategy can access both types of data
class HybridStrategy(StrategyBase):
    def __init__(self, tickers):
        # Market tickers
        self.market_tickers = ['AAPL', 'GOOGL']
        # Alternative data tickers
        self.alt_tickers = ['sentiment:AAPL', 'sentiment:GOOGL']
        # Combined list
        tickers = self.market_tickers + self.alt_tickers
        super().__init__(tickers)
    
    def step(self, current_date, daily_data):
        allocations = {}
        
        for ticker in self.market_tickers:
            if ticker in daily_data and daily_data[ticker]:
                # Market data has OHLCV format
                price = daily_data[ticker]['close']
                sentiment_key = f'sentiment:{ticker}'
                
                if sentiment_key in daily_data and daily_data[sentiment_key]:
                    # Alternative data has custom format
                    sentiment = daily_data[sentiment_key]['sentiment_score']
                    
                    # Use both market price and sentiment
                    if sentiment > 0.5 and price > 100:
                        allocations[ticker] = 1.0
                    else:
                        allocations[ticker] = 0.0
                else:
                    allocations[ticker] = 0.0
        
        return allocations
```

### Data Format Differences

#### Market Data Format
Market data always follows the standard OHLCV format:

```python
# Market data structure
{
    'AAPL': {
        'open': 100.0,
        'high': 102.5,
        'low': 99.0,
        'close': 101.2,
        'volume': 1000000.0
    }
}
```

#### Alternative Data Format
Alternative data can have any structure, but should be consistent:

```python
# Sentiment data structure
{
    'sentiment:AAPL': {
        'sentiment_score': 0.75,
        'volume': 5000,
        'confidence': 0.9
    }
}

# Economic data structure
{
    'fred:GDP': {
        'value': 21433.2,
        'change': 0.5,
        'units': 'Billions of Dollars'
    }
}
```

### Data Source Management

#### Automatic Source Detection

The backtester automatically detects and routes data to the appropriate loader:

```python
# The backtester splits tickers based on specifiers
market_tickers, alt_tickers = backtester._split_tickers(strategy.tickers)

# Market tickers: ['AAPL', 'GOOGL']
# Alt tickers: ['sentiment:AAPL', 'fred:GDP']
```

#### Multiple Alternative Data Sources

You can combine multiple alternative data sources:

```python
from portwine.data.providers.loader_adapters import FREDDataLoader, SentimentDataLoader, NewsDataLoader, AlternativeDataLoader

# Multiple alternative data loaders
fred_loader = FREDDataLoader(api_key='your_fred_key')
sentiment_loader = SentimentDataLoader('path/to/sentiment/')
news_loader = NewsDataLoader('path/to/news/')

# Initialize the main alternative data loader with all sub-loaders
alt_loader = AlternativeDataLoader()
alt_loader.add_loader('fred', fred_loader)
alt_loader.add_loader('sentiment', sentiment_loader)
alt_loader.add_loader('news', news_loader)

# Strategy with multiple data sources
class MultiSourceStrategy(StrategyBase):
    def __init__(self):
        self.tickers = [
            # Market data
            'AAPL', 'GOOGL',
            # Economic data
            'fred:GDP', 'fred:UNRATE',
            # Sentiment data
            'sentiment:AAPL', 'sentiment:GOOGL',
            # News data
            'news:AAPL', 'news:GOOGL'
        ]
        super().__init__(self.tickers)
    
    def step(self, current_date, daily_data):
        # Access different data types
        aapl_price = daily_data.get('AAPL', {}).get('close')
        gdp = daily_data.get('fred:GDP', {}).get('value')
        sentiment = daily_data.get('sentiment:AAPL', {}).get('sentiment_score')
        news_count = daily_data.get('news:AAPL', {}).get('article_count')
        
        # Combine all signals
        if (aapl_price and gdp and sentiment and news_count):
            # Your strategy logic here
            pass
```

### Best Practices for Alternative Data

#### 1. Consistent Naming

Use consistent specifier prefixes:

```python
# Good: Consistent naming
alt_tickers = [
    'sentiment:AAPL', 'sentiment:GOOGL',
    'news:AAPL', 'news:GOOGL',
    'fred:GDP', 'fred:UNRATE'
]

# Avoid: Inconsistent naming
alt_tickers = [
    'sentiment:AAPL', 'news_GOOGL',  # Mixed separators
    'FRED:GDP', 'fred:UNRATE'        # Mixed case
]
```

#### 2. Data Synchronization

Ensure alternative data aligns with market data dates:

```python
def synchronize_alt_data(market_data, alt_data):
    """Align alternative data with market data dates."""
    market_dates = set(market_data['AAPL'].index)
    
    for ticker, df in alt_data.items():
        # Filter to market trading days
        alt_data[ticker] = df[df.index.isin(market_dates)]
    
    return alt_data
```

#### 3. Missing Data Handling

Handle missing alternative data gracefully:

```python
def step(self, current_date, daily_data):
    allocations = {}
    
    for ticker in self.market_tickers:
        # Check if we have both market and alternative data
        has_market = ticker in daily_data and daily_data[ticker] is not None
        has_sentiment = f'sentiment:{ticker}' in daily_data and daily_data[f'sentiment:{ticker}'] is not None
        
        if has_market and has_sentiment:
            # Use both data sources
            price = daily_data[ticker]['close']
            sentiment = daily_data[f'sentiment:{ticker}']['sentiment_score']
            allocations[ticker] = self.calculate_weight(price, sentiment)
        elif has_market:
            # Fall back to market data only
            allocations[ticker] = self.calculate_weight_market_only(daily_data[ticker])
        else:
            # No data available
            allocations[ticker] = 0.0
    
    return allocations
```

### Creating Custom Alternative Data Loaders

To create a custom alternative data loader:

```python
from portwine.data.providers.loader_adapters import MarketDataLoader

class CustomAltDataLoader(MarketDataLoader):
    def __init__(self, data_path, source_name):
        self.data_path = data_path
        self.source_name = source_name  # e.g., 'sentiment', 'news'
        super().__init__()
    
    def load_ticker(self, ticker):
        # Extract base ticker from specifier
        if not ticker.startswith(f'{self.source_name}:'):
            return None
        
        base_ticker = ticker.replace(f'{self.source_name}:', '')
        file_path = f"{self.data_path}/{base_ticker}.csv"
        
        if not os.path.exists(file_path):
            return None
        
        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        # Return whatever columns your alternative data has
        return df

# Usage
sentiment_loader = CustomAltDataLoader('path/to/data/', 'sentiment')
data = sentiment_loader.fetch_data(['sentiment:AAPL', 'sentiment:GOOGL'])
```

This system provides flexibility to combine any type of alternative data with market data while maintaining clean separation between different data sources.

## Data Caching

### Implementing Caching

```python
import pickle
import os

class CachedDataLoader(MarketDataLoader):
    def __init__(self, base_loader, cache_dir='./cache'):
        self.base_loader = base_loader
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch_data(self, tickers):
        """Fetch data with caching."""
        cached_data = {}
        uncached_tickers = []
        
        # Check cache first
        for ticker in tickers:
            cache_file = f"{self.cache_dir}/{ticker}.pkl"
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_data[ticker] = pickle.load(f)
            else:
                uncached_tickers.append(ticker)
        
        # Fetch uncached data
        if uncached_tickers:
            new_data = self.base_loader.fetch_data(uncached_tickers)
            
            # Cache new data
            for ticker, df in new_data.items():
                cache_file = f"{self.cache_dir}/{ticker}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                cached_data[ticker] = df
        
        return cached_data

# Use cached loader
cached_loader = CachedDataLoader(base_loader)
```

## Best Practices

### 1. Data Quality Checks

```python
def check_data_quality(data_dict):
    """Comprehensive data quality check."""
    issues = []
    
    for ticker, df in data_dict.items():
        # Check for required columns
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            issues.append(f"{ticker}: Missing required columns")
        
        # Check for negative prices
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            issues.append(f"{ticker}: Negative prices detected")
        
        # Check for price consistency
        if not ((df['low'] <= df['open']) & (df['low'] <= df['close']) & 
                (df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
            issues.append(f"{ticker}: Price consistency issues")
        
        # Check for reasonable volume
        if (df['volume'] < 0).any():
            issues.append(f"{ticker}: Negative volume detected")
    
    return issues

# Run quality checks
issues = check_data_quality(data)
if issues:
    print("Data quality issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

### 2. Data Synchronization

```python
def synchronize_data(data_dict):
    """Ensure all tickers have data for the same date range."""
    # Find common date range
    all_dates = []
    for df in data_dict.values():
        all_dates.extend(df.index.tolist())
    
    common_start = max(df.index.min() for df in data_dict.values())
    common_end = min(df.index.max() for df in data_dict.values())
    
    # Filter to common range
    synchronized_data = {}
    for ticker, df in data_dict.items():
        mask = (df.index >= common_start) & (df.index <= common_end)
        synchronized_data[ticker] = df[mask]
    
    return synchronized_data

# Synchronize your data
sync_data = synchronize_data(data)
```

### 3. Memory Management

```python
# For large datasets, consider loading data in chunks
def load_data_in_chunks(data_loader, tickers, chunk_size=100):
    """Load data in chunks to manage memory."""
    all_data = {}
    
    for i in range(0, len(tickers), chunk_size):
        chunk_tickers = tickers[i:i+chunk_size]
        chunk_data = data_loader.fetch_data(chunk_tickers)
        all_data.update(chunk_data)
        
        # Optional: clear memory
        del chunk_data
    
    return all_data
```

## Troubleshooting

### Common Issues

1. **Missing Data Files**
   ```python
   # Check if files exist
   import os
   for ticker in tickers:
       file_path = f"data_path/{ticker}.csv"
       if not os.path.exists(file_path):
           print(f"Warning: {file_path} not found")
   ```

2. **Date Format Issues**
   ```python
   # Ensure proper date parsing
   df.index = pd.to_datetime(df.index, errors='coerce')
   df = df.dropna()  # Remove rows with invalid dates
   ```

3. **Timezone Issues**
   ```python
   # Normalize timezones
   df.index = df.index.tz_localize(None)  # Remove timezone info
   ```

## Next Steps

- Learn about [building strategies](strategies.md)
- Explore [backtesting](backtesting.md)
- Check out [performance analysis](analysis.md) 