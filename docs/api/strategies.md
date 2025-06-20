# Strategies API

Strategies are the core components of portwine that define how your portfolio allocates capital based on market conditions.

## StrategyBase Class

All strategies in portwine inherit from the `StrategyBase` class:

```python
from portwine import StrategyBase

class StrategyBase:
    """
    Base class for all strategies in portwine.
    
    All strategies must inherit from this class and implement the `step` method.
    """
    
    def __init__(self, tickers: List[str]):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols this strategy will trade
        """
        self.tickers = tickers
    
    def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict]) -> Dict[str, float]:
        """
        Process daily data and return allocations.
        
        This method is called for each trading day and must return
        allocation weights for each ticker.
        
        Parameters
        ----------
        current_date : pd.Timestamp
            Current trading date
        daily_data : Dict[str, Dict]
            Dictionary with ticker -> OHLCV data
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping ticker symbols to allocation weights (0.0 to 1.0)
        """
        raise NotImplementedError("Subclasses must implement step method")
```

## Built-in Strategies

### SimpleMomentumStrategy

A simple momentum strategy that invests in the best-performing asset:

```python
from portwine import SimpleMomentumStrategy

class SimpleMomentumStrategy(StrategyBase):
    """
    A simple momentum strategy that:
    1. Calculates N-day momentum for each ticker
    2. Invests in the top performing ticker
    3. Rebalances weekly (every Friday)
    """
    
    def __init__(self, tickers: List[str], lookback_days: int = 10):
        """
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols to consider for investment
        lookback_days : int, default 10
            Number of days to use for momentum calculation
        """
        super().__init__(tickers)
        self.lookback_days = lookback_days
        self.price_history = {ticker: [] for ticker in tickers}
        self.current_signals = {ticker: 0.0 for ticker in tickers}
        self.dates = []
    
    def is_friday(self, date: pd.Timestamp) -> bool:
        """Check if given date is a Friday (weekday 4)"""
        return date.weekday() == 4
    
    def calculate_momentum(self, ticker: str) -> float:
        """Calculate simple price momentum over lookback period"""
        prices = self.price_history[ticker]
        
        # Need at least lookback_days+1 data points
        if len(prices) <= self.lookback_days:
            return -999.0
        
        # Get starting and ending prices for momentum calculation
        start_price = prices[-self.lookback_days-1]
        end_price = prices[-1]
        
        # Check for valid prices
        if start_price is None or end_price is None or start_price <= 0:
            return -999.0
        
        # Return simple momentum (end/start - 1)
        return end_price / start_price - 1.0
    
    def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict]) -> Dict[str, float]:
        """
        Process daily data and determine allocations
        """
        # Track dates for rebalancing logic
        self.dates.append(current_date)
        
        # Update price history for each ticker
        for ticker in self.tickers:
            price = None
            if daily_data.get(ticker) is not None:
                price = daily_data[ticker].get('close', None)
            
            # Forward fill missing data
            if price is None and len(self.price_history[ticker]) > 0:
                price = self.price_history[ticker][-1]
                
            self.price_history[ticker].append(price)
        
        # Only rebalance on Fridays
        if self.is_friday(current_date):
            # Calculate momentum for each ticker
            momentum_scores = {}
            for ticker in self.tickers:
                momentum_scores[ticker] = self.calculate_momentum(ticker)
            
            # Find best performing ticker
            best_ticker = max(momentum_scores.items(), 
                             key=lambda x: x[1] if x[1] != -999.0 else -float('inf'))[0]
            
            # Reset all allocations to zero
            self.current_signals = {ticker: 0.0 for ticker in self.tickers}
            
            # Allocate 100% to best performer if we have valid momentum
            if momentum_scores[best_ticker] != -999.0:
                self.current_signals[best_ticker] = 1.0
        
        # Return current allocations
        return self.current_signals.copy()
```

## Creating Custom Strategies

### Basic Strategy Template

```python
class MyCustomStrategy(StrategyBase):
    def __init__(self, tickers: List[str], **parameters):
        super().__init__(tickers)
        # Initialize your strategy parameters and state
        self.parameters = parameters
        self.state = {}
    
    def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict]) -> Dict[str, float]:
        """
        Your strategy logic goes here.
        
        Parameters
        ----------
        current_date : pd.Timestamp
            Current trading date
        daily_data : Dict[str, Dict]
            Dictionary with ticker -> OHLCV data
            
        Returns
        -------
        Dict[str, float]
            Allocation weights for each ticker
        """
        # Your strategy implementation
        allocations = {}
        
        # Example: Equal weight allocation
        weight = 1.0 / len(self.tickers)
        for ticker in self.tickers:
            allocations[ticker] = weight
        
        return allocations
```

### Advanced Strategy Example

```python
class MeanReversionStrategy(StrategyBase):
    """
    A mean reversion strategy that:
    1. Calculates rolling z-scores for each asset
    2. Goes long assets with negative z-scores (oversold)
    3. Goes short assets with positive z-scores (overbought)
    """
    
    def __init__(self, tickers: List[str], lookback_days: int = 60, z_threshold: float = 1.0):
        super().__init__(tickers)
        self.lookback_days = lookback_days
        self.z_threshold = z_threshold
        self.price_history = {ticker: [] for ticker in tickers}
    
    def calculate_z_score(self, ticker: str) -> float:
        """Calculate z-score for a ticker"""
        prices = self.price_history[ticker]
        
        if len(prices) < self.lookback_days:
            return 0.0
        
        recent_prices = prices[-self.lookback_days:]
        current_price = recent_prices[-1]
        
        if current_price is None:
            return 0.0
        
        mean_price = np.mean([p for p in recent_prices if p is not None])
        std_price = np.std([p for p in recent_prices if p is not None])
        
        if std_price == 0:
            return 0.0
        
        return (current_price - mean_price) / std_price
    
    def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict]) -> Dict[str, float]:
        # Update price history
        for ticker in self.tickers:
            if daily_data.get(ticker):
                self.price_history[ticker].append(daily_data[ticker]['close'])
            else:
                # Forward fill if no new data
                if len(self.price_history[ticker]) > 0:
                    self.price_history[ticker].append(self.price_history[ticker][-1])
                else:
                    self.price_history[ticker].append(None)
        
        # Calculate allocations based on z-scores
        allocations = {}
        total_weight = 0.0
        
        for ticker in self.tickers:
            z_score = self.calculate_z_score(ticker)
            
            if z_score < -self.z_threshold:
                # Oversold - go long
                allocations[ticker] = 1.0
                total_weight += 1.0
            elif z_score > self.z_threshold:
                # Overbought - go short
                allocations[ticker] = -1.0
                total_weight += 1.0
            else:
                # Neutral
                allocations[ticker] = 0.0
        
        # Normalize weights
        if total_weight > 0:
            for ticker in allocations:
                allocations[ticker] /= total_weight
        
        return allocations
```

## Strategy Best Practices

### 1. Handle Missing Data

```python
def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict]) -> Dict[str, float]:
    allocations = {}
    
    for ticker in self.tickers:
        if ticker in daily_data and daily_data[ticker] is not None:
            # Process valid data
            allocations[ticker] = self.calculate_weight(ticker, daily_data[ticker])
        else:
            # Handle missing data gracefully
            allocations[ticker] = 0.0
    
    return allocations
```

### 2. Validate Allocations

```python
def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict]) -> Dict[str, float]:
    # Calculate raw allocations
    allocations = self.calculate_allocations(daily_data)
    
    # Ensure weights sum to 1.0 (or 0.0 for cash)
    total_weight = sum(abs(weight) for weight in allocations.values())
    
    if total_weight > 0:
        # Normalize weights
        for ticker in allocations:
            allocations[ticker] /= total_weight
    
    return allocations
```

### 3. Use Efficient Data Structures

```python
def __init__(self, tickers: List[str]):
    super().__init__(tickers)
    # Pre-allocate data structures for efficiency
    self.price_history = {ticker: [] for ticker in tickers}
    self.signals = {ticker: 0.0 for ticker in tickers}
    self.last_rebalance = None
```

### 4. Implement State Management

```python
class StatefulStrategy(StrategyBase):
    def __init__(self, tickers: List[str]):
        super().__init__(tickers)
        self.position_history = []
        self.last_rebalance_date = None
        self.current_positions = {ticker: 0.0 for ticker in tickers}
    
    def should_rebalance(self, current_date: pd.Timestamp) -> bool:
        """Determine if rebalancing is needed"""
        if self.last_rebalance_date is None:
            return True
        
        # Rebalance weekly
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        return days_since_rebalance >= 7
    
    def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict]) -> Dict[str, float]:
        if self.should_rebalance(current_date):
            # Perform rebalancing
            self.last_rebalance_date = current_date
            new_allocations = self.calculate_new_allocations(daily_data)
            
            # Track position changes
            for ticker in self.tickers:
                old_position = self.current_positions.get(ticker, 0.0)
                new_position = new_allocations.get(ticker, 0.0)
                
                if abs(new_position - old_position) > 0.01:  # 1% threshold
                    self.position_history.append({
                        'date': current_date,
                        'ticker': ticker,
                        'old_position': old_position,
                        'new_position': new_position
                    })
            
            self.current_positions = new_allocations.copy()
        
        return self.current_positions.copy()
```

## Strategy Testing

### Unit Testing Your Strategy

```python
import pytest
import pandas as pd
from unittest.mock import Mock

def test_strategy_initialization():
    """Test strategy initialization"""
    strategy = MyCustomStrategy(['AAPL', 'GOOGL'])
    assert strategy.tickers == ['AAPL', 'GOOGL']

def test_strategy_step():
    """Test strategy step method"""
    strategy = MyCustomStrategy(['AAPL', 'GOOGL'])
    
    # Mock daily data
    daily_data = {
        'AAPL': {'open': 150, 'high': 152, 'low': 149, 'close': 151, 'volume': 1000000},
        'GOOGL': {'open': 2800, 'high': 2820, 'low': 2790, 'close': 2810, 'volume': 500000}
    }
    
    current_date = pd.Timestamp('2023-01-01')
    allocations = strategy.step(current_date, daily_data)
    
    # Verify allocations
    assert 'AAPL' in allocations
    assert 'GOOGL' in allocations
    assert sum(allocations.values()) == 1.0  # Should sum to 1.0

def test_strategy_missing_data():
    """Test strategy handles missing data gracefully"""
    strategy = MyCustomStrategy(['AAPL', 'GOOGL'])
    
    # Mock daily data with missing ticker
    daily_data = {
        'AAPL': {'open': 150, 'high': 152, 'low': 149, 'close': 151, 'volume': 1000000}
        # GOOGL missing
    }
    
    current_date = pd.Timestamp('2023-01-01')
    allocations = strategy.step(current_date, daily_data)
    
    # Should handle missing data
    assert 'GOOGL' in allocations
    assert allocations['GOOGL'] == 0.0  # Should be 0 for missing data
```

## Performance Considerations

### Memory Management

```python
class MemoryEfficientStrategy(StrategyBase):
    def __init__(self, tickers: List[str], max_history: int = 1000):
        super().__init__(tickers)
        self.max_history = max_history
        self.price_history = {ticker: [] for ticker in tickers}
    
    def add_price(self, ticker: str, price: float):
        """Add price while maintaining maximum history size"""
        self.price_history[ticker].append(price)
        
        # Keep only the most recent prices
        if len(self.price_history[ticker]) > self.max_history:
            self.price_history[ticker] = self.price_history[ticker][-self.max_history:]
```

### Computational Efficiency

```python
class EfficientStrategy(StrategyBase):
    def __init__(self, tickers: List[str]):
        super().__init__(tickers)
        # Pre-calculate constants
        self.n_tickers = len(tickers)
        self.equal_weight = 1.0 / self.n_tickers
        
        # Use numpy arrays for faster computation
        self.price_array = np.zeros((self.n_tickers, 1000))  # Pre-allocate
        self.current_index = 0
    
    def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict]) -> Dict[str, float]:
        # Use vectorized operations when possible
        prices = np.array([daily_data.get(ticker, {}).get('close', 0) for ticker in self.tickers])
        
        # Your efficient strategy logic here
        # ...
        
        return {ticker: self.equal_weight for ticker in self.tickers}
```

## Next Steps

- Learn about [backtesting strategies](backtester.md)
- Explore [performance analysis](analysis.md)
- Check out [data management](data-management.md) 