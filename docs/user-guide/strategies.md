# Building Strategies

Strategies in portwine are the heart of your backtesting system. They define how your portfolio allocates capital based on market conditions.

Strategies are the core components of portwine that define how your portfolio allocates capital based on market conditions.

## Online Architecture

Portwine uses an **online architecture** where strategies process data one time period at a time. This means:

- **Single Time Period Data**: Each call to `step()` receives data for only the current trading day
- **No Historical Context**: Strategies do not have access to previous time periods' data unless they explicitly store it
- **Self-Contained Indicators**: All technical indicators, moving averages, momentum calculations, and other analysis must be implemented within the strategy itself
- **State Management**: Strategies must maintain their own state (price history, indicators, etc.) across time periods

This architecture is designed for:

- **Real-time trading**: Strategies can be deployed in live trading environments
- **Memory efficiency**: Only current data is processed at each step
- **Scalability**: Strategies can handle large datasets without loading everything into memory

### Why This Matters

```python
# ❌ WRONG - This won't work in portwine
def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict]) -> Dict[str, float]:
    # This assumes you have access to historical data
    # But portwine only gives you current day data!
    historical_prices = get_historical_prices(ticker, start_date, end_date)  # Not available
    sma_20 = calculate_sma(historical_prices, 20)  # Can't calculate without history
    
    return allocations

# ✅ CORRECT - Store and update indicators within the strategy
class ProperStrategy(StrategyBase):
    def __init__(self, tickers: List[str]):
        super().__init__(tickers)
        # Store price history within the strategy
        self.price_history = {ticker: [] for ticker in tickers}
        self.sma_20 = {ticker: None for ticker in tickers}
    
    def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict]) -> Dict[str, float]:
        # Update price history with current data
        for ticker in self.tickers:
            if daily_data.get(ticker):
                self.price_history[ticker].append(daily_data[ticker]['close'])
                
                # Calculate indicators using stored history
                if len(self.price_history[ticker]) >= 20:
                    recent_prices = self.price_history[ticker][-20:]
                    self.sma_20[ticker] = sum(recent_prices) / len(recent_prices)
        
        # Use the calculated indicators for allocation decisions
        allocations = self.calculate_allocations()
        return allocations
```

## Strategy Basics

All strategies in portwine inherit from `StrategyBase` and implement a `step` method. This method receives the current date and market data, then returns allocation weights.

```python
from portwine import StrategyBase

class MyStrategy(StrategyBase):
    def __init__(self, tickers):
        super().__init__(tickers)
        # Initialize your strategy state here
    
    def step(self, current_date, daily_data):
        """
        Process daily data and return allocations
        
        Parameters
        ----------
        current_date : datetime
            Current trading date
        daily_data : dict
            Dictionary with ticker -> OHLCV data
            
        Returns
        -------
        dict
            Ticker -> allocation weight (0.0 to 1.0)
        """
        # Your strategy logic here
        allocations = {}
        for ticker in self.tickers:
            allocations[ticker] = 0.0
        
        return allocations
```

## The Step Method

The `step` method is called for each trading day and receives:

- **`current_date`**: The current trading date as a datetime object
- **`daily_data`**: A dictionary where keys are tickers and values are OHLCV data dictionaries

### Daily Data Format

```python
daily_data = {
    'AAPL': {
        'open': 150.0,
        'high': 152.0,
        'low': 149.0,
        'close': 151.0,
        'volume': 1000000
    },
    'GOOGL': {
        'open': 2800.0,
        'high': 2820.0,
        'low': 2790.0,
        'close': 2810.0,
        'volume': 500000
    },
    'FRED:SP500': {
        'open': 4500.0,
        'high': 4510.0,
        'low': 4495.0,
        'close': 4505.0,
        'volume': None  # Some alternative data may not have volume
    },
    'SENTIMENT:AAPL': {
        'sentiment_score': 0.75,
        'confidence': 0.92,
        'source_count': 150
    },
    'EARNINGS:GOOGL': {
        'eps': 2.45,
        'revenue': 75000000000,
        'guidance': 'positive'
    }
    # ... more tickers including alternative data sources
}
```

**Important Notes:**

- The `daily_data` dictionary is guaranteed to contain keys for all tickers in your strategy, even if no market data exists for that day (values will be `None`)
- Alternative data sources appear as separate tickers with the format `"SOURCE:TICKER"` (e.g., `"FRED:SP500"`, `"SENTIMENT:AAPL"`)
- **Alternative data sources can define their own schemas** - they are not limited to OHLCV format and can include any fields relevant to their data type
- Each alternative data loader should document its schema so you know what fields are available
- Some alternative data sources may not have all OHLCV fields (e.g., volume might be `None`)
- Always check for `None` values and handle missing data gracefully in your strategy logic

## Example: Simple Equal Weight Strategy

```python
class EqualWeightStrategy(StrategyBase):
    def __init__(self, tickers):
        super().__init__(tickers)
    
    def step(self, current_date, daily_data):
        # Equal weight allocation
        weight = 1.0 / len(self.tickers)
        return {ticker: weight for ticker in self.tickers}
```

## Example: Moving Average Crossover

```python
class MACrossoverStrategy(StrategyBase):
    def __init__(self, tickers, short_window=10, long_window=50):
        super().__init__(tickers)
        self.short_window = short_window
        self.long_window = long_window
        self.price_history = {ticker: [] for ticker in tickers}
    
    def step(self, current_date, daily_data):
        # Update price history
        for ticker in self.tickers:
            if daily_data.get(ticker):
                self.price_history[ticker].append(daily_data[ticker]['close'])
        
        allocations = {}
        for ticker in self.tickers:
            prices = self.price_history[ticker]
            
            if len(prices) >= self.long_window:
                short_ma = sum(prices[-self.short_window:]) / self.short_window
                long_ma = sum(prices[-self.long_window:]) / self.long_window
                
                # Buy signal when short MA > long MA
                if short_ma > long_ma:
                    allocations[ticker] = 1.0 / len(self.tickers)
                else:
                    allocations[ticker] = 0.0
            else:
                allocations[ticker] = 0.0
        
        return allocations
```

## Strategy State Management

Strategies can maintain state between calls to `step`:

```python
class StatefulStrategy(StrategyBase):
    def __init__(self, tickers):
        super().__init__(tickers)
        self.position_history = []
        self.last_rebalance_date = None
    
    def step(self, current_date, daily_data):
        # Use state to make decisions
        if self.should_rebalance(current_date):
            self.last_rebalance_date = current_date
            # ... rebalancing logic
        
        # ... rest of strategy logic
```

## Best Practices

### 1. Handle Missing Data
```python
def step(self, current_date, daily_data):
    allocations = {}
    for ticker in self.tickers:
        if ticker in daily_data and daily_data[ticker] is not None:
            # Process valid data
            allocations[ticker] = self.calculate_weight(ticker, daily_data[ticker])
        else:
            # Handle missing data
            allocations[ticker] = 0.0
    return allocations
```

### 2. Validate Allocations
```python
def step(self, current_date, daily_data):
    allocations = self.calculate_allocations(daily_data)
    
    # Ensure weights sum to 1.0 (or 0.0 for cash)
    total_weight = sum(allocations.values())
    if total_weight > 0:
        # Normalize weights
        for ticker in allocations:
            allocations[ticker] /= total_weight
    
    return allocations
```

### 3. Use Efficient Data Structures
```python
def __init__(self, tickers):
    super().__init__(tickers)
    # Pre-allocate data structures
    self.price_history = {ticker: [] for ticker in tickers}
    self.signals = {ticker: 0.0 for ticker in tickers}
```

## Advanced Features

### Alternative Data Support
Strategies can access alternative data through the backtester:

```python
def step(self, current_date, daily_data):
    # Access alternative data with custom schemas
    if 'FRED:SP500' in daily_data:
        sp500_data = daily_data['FRED:SP500']
        # Use alternative data in your strategy
        if sp500_data and sp500_data['close'] > 4500:
            # SP500 is above threshold, adjust allocations
            pass
    
    # Access sentiment data with custom schema
    if 'SENTIMENT:AAPL' in daily_data:
        sentiment_data = daily_data['SENTIMENT:AAPL']
        if sentiment_data and sentiment_data['sentiment_score'] > 0.7:
            # High sentiment, consider overweighting
            pass
    
    # Access earnings data with custom schema
    if 'EARNINGS:GOOGL' in daily_data:
        earnings_data = daily_data['EARNINGS:GOOGL']
        if earnings_data and earnings_data['guidance'] == 'positive':
            # Positive guidance, bullish signal
            pass
```

### Calendar Awareness
Strategies can be aware of trading calendars:

```python
def step(self, current_date, daily_data):
    # Check if it's a rebalancing day
    if current_date.weekday() == 4:  # Friday
        # Weekly rebalancing logic
        pass
```

## Next Steps

- Learn about [backtesting your strategies](backtesting.md)
- Explore [data management](data-management.md)
- Check out [performance analysis](analysis.md) 