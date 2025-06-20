# Building Strategies

Strategies in portwine are the heart of your backtesting system. They define how your portfolio allocates capital based on market conditions.

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
    }
    # ... more tickers
}
```

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
    # Access alternative data if available
    if 'alt:signal' in daily_data:
        alt_signal = daily_data['alt:signal']
        # Use alternative data in your strategy
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