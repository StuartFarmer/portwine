# Data Management

Portwine provides flexible data management through its data loader system, making it easy to work with various data sources and formats.

## Data Loaders

Data loaders are responsible for fetching and providing market data to the backtester. Portwine supports multiple data loader types:

### EODHD Market Data Loader

The most common data loader for historical market data:

```python
from portwine import EODHDMarketDataLoader

# Initialize with your data directory
data_loader = EODHDMarketDataLoader(data_path='path/to/your/eodhd/data/')

# Fetch data for specific tickers
data = data_loader.fetch_data(['AAPL', 'GOOGL', 'MSFT'])
```

### Polygon Market Data Loader

For Polygon.io data:

```python
from portwine import PolygonMarketDataLoader

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

## Data Directory Structure

### EODHD Format

```
data_path/
├── US/
│   ├── AAPL.csv
│   ├── GOOGL.csv
│   ├── MSFT.csv
│   └── ...
├── ETF/
│   ├── SPY.csv
│   ├── QQQ.csv
│   └── ...
└── INDEX/
    ├── ^GSPC.csv
    └── ^VIX.csv
```

### Polygon Format

```
data_path/
├── stocks/
│   ├── AAPL/
│   │   ├── 2023-01-01.csv
│   │   ├── 2023-01-02.csv
│   │   └── ...
│   └── GOOGL/
│       ├── 2023-01-01.csv
│       └── ...
└── etfs/
    └── SPY/
        └── ...
```

## Working with Data

### Fetching Data

```python
# Fetch data for multiple tickers
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
data = data_loader.fetch_data(tickers)

# Check what data is available
for ticker, df in data.items():
    print(f"{ticker}: {df.index.min()} to {df.index.max()}")
```

### Data Validation

```python
def validate_data(data_dict):
    """Validate data quality and consistency."""
    for ticker, df in data_dict.items():
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            print(f"Warning: {ticker} missing columns: {missing_cols}")
        
        # Check for missing values
        missing_values = df[required_cols].isnull().sum()
        if missing_values.sum() > 0:
            print(f"Warning: {ticker} has missing values:\n{missing_values}")
        
        # Check date range
        print(f"{ticker}: {df.index.min()} to {df.index.max()} ({len(df)} days)")

# Validate your data
validate_data(data)
```

### Data Preprocessing

```python
def preprocess_data(df):
    """Clean and prepare data for backtesting."""
    # Remove rows with missing values
    df = df.dropna()
    
    # Ensure data types are correct
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date
    df = df.sort_index()
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    return df

# Apply preprocessing to all data
cleaned_data = {}
for ticker, df in data.items():
    cleaned_data[ticker] = preprocess_data(df)
```

## Alternative Data

### Custom Data Loaders

You can create custom data loaders for alternative data sources:

```python
from portwine.loaders.base import MarketDataLoader

class CustomDataLoader(MarketDataLoader):
    def __init__(self, data_path):
        self.data_path = data_path
    
    def fetch_data(self, tickers):
        """Fetch data for the given tickers."""
        data = {}
        for ticker in tickers:
            # Your custom data loading logic here
            file_path = f"{self.data_path}/{ticker}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                data[ticker] = df
        return data

# Use your custom loader
custom_loader = CustomDataLoader('path/to/alternative/data/')
```

### Alternative Data Integration

```python
# Set up alternative data loader
alt_loader = CustomDataLoader('path/to/alt/data/')

# Create backtester with both market and alternative data
backtester = Backtester(
    market_data_loader=market_loader,
    alternative_data_loader=alt_loader
)

# Strategy can access alternative data
class AltDataStrategy(StrategyBase):
    def step(self, current_date, daily_data):
        # Market data
        aapl_price = daily_data.get('AAPL', {}).get('close')
        
        # Alternative data
        sentiment = daily_data.get('alt:sentiment', 0)
        
        # Use both in your strategy
        if sentiment > 0.5 and aapl_price:
            # Bullish sentiment and valid price
            return {'AAPL': 1.0}
        else:
            return {'AAPL': 0.0}
```

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