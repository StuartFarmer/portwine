# Quick Start

This guide will walk you through creating and running your first strategy with portwine.

## Your First Strategy

Let's create a simple momentum strategy that invests in the best-performing asset from the previous period.

```python
from portwine.backtester import Backtester
from portwine.loaders import EODHDMarketDataLoader
from portwine.strategies import StrategyBase

class SimpleMomentumStrategy(StrategyBase):
    """
    A simple momentum strategy that:
    1. Calculates N-day momentum for each ticker
    2. Invests in the top performing ticker
    3. Rebalances weekly (every Friday)
    """
    
    def __init__(self, tickers, lookback_days=10):
        """
        Parameters
        ----------
        tickers : list
            List of ticker symbols to consider for investment
        lookback_days : int, default 10
            Number of days to use for momentum calculation
        """
        # Pass tickers to parent class (StrategyBase) for initialization
        super().__init__(tickers)
        
        # Set the lookback window size for momentum calculation
        self.lookback_days = lookback_days
        
        # Initialize price history storage for each ticker
        # Because portwine is ONLY walkforward, we must store data after every timestep
        # to build up a history to run our analysis on
        self.price_history = {ticker: [] for ticker in tickers}
    
    def is_friday(self, date):
        """Check if given date is a Friday (weekday 4)"""
        return date.weekday() == 4
    
    def calculate_momentum(self, ticker):
        """Calculate simple price momentum over lookback period"""
        # Get the price history for this specific ticker
        prices = self.price_history[ticker]
        
        # Need at least lookback_days+1 data points to calculate momentum
        # (we need start_price and end_price with lookback_days between them)
        if len(prices) <= self.lookback_days:
            return -999.0  # Sentinel value indicating insufficient data
        
        # Get starting price (lookback_days ago) and ending price (today)
        start_price = prices[-self.lookback_days-1]  # Price from lookback_days+1 ago
        end_price = prices[-1]  # Most recent price (today)
        
        # Check for valid prices (not None and positive)
        if start_price is None or end_price is None or start_price <= 0:
            return -999.0  # Sentinel value for invalid data
        
        # Calculate momentum: (end_price / start_price) - 1
        # This gives us the percentage change over the lookback period
        return end_price / start_price - 1.0
    
    def step(self, current_date, daily_data):
        """
        Process daily data and determine allocations
        Called by portwine for each trading day
        """
        # Update price history for each ticker with today's data
        for ticker in self.tickers:
            price = None
            
            # Extract close price from daily data if available
            if daily_data.get(ticker) is not None:
                price = daily_data[ticker].get('close', None)
            
            # Forward fill missing data: if no price today, use yesterday's price
            if price is None and len(self.price_history[ticker]) > 0:
                price = self.price_history[ticker][-1]  # Last known price
                
            # Add today's price (or forward-filled price) to history
            self.price_history[ticker].append(price)
        
        # Only rebalance on Fridays to reduce trading costs
        if self.is_friday(current_date):
            # Calculate momentum score for each ticker
            momentum_scores = {}
            for ticker in self.tickers:
                momentum_scores[ticker] = self.calculate_momentum(ticker)
            
            # Find the ticker with the highest momentum score
            # Handle sentinel values (-999.0) by treating them as negative infinity
            best_ticker = max(momentum_scores.items(), 
                             key=lambda x: x[1] if x[1] != -999.0 else -float('inf'))[0]
            
            # Create new allocation signals
            signals = {ticker: 0.0 for ticker in self.tickers}
            
            # Allocate 100% to best performer if we have valid momentum data
            if momentum_scores[best_ticker] != -999.0:
                signals[best_ticker] = 1.0
            
            return signals
        else:
            # On non-Friday days, return zero allocation (cash)
            # This maintains the previous Friday's allocation until next rebalance
            return {ticker: 0.0 for ticker in self.tickers}

# Define your investment universe
universe = ['MTUM', 'VTV', 'VUG', 'IJR', 'MDY']

# Create a momentum strategy
strategy = SimpleMomentumStrategy(
    tickers=universe, 
    lookback_days=10
)

# Set up your data loader
data_loader = EODHDMarketDataLoader(
    data_path='path/to/your/eodhd/data/'
)

# Create the backtester
backtester = Backtester(market_data_loader=data_loader)

# Run the backtest
results = backtester.run_backtest(
    strategy=strategy,
    benchmark_ticker='SPY',
    start_date='2020-01-01',
    end_date='2023-12-31',
    verbose=True
)
```

For a more detailed tutorial on writing strategies, [click here](tutorial.md).

## Understanding the Results

The backtest returns a dictionary with several key components:

```python
# Strategy signals over time
signals_df = results['signals_df']

# Individual asset returns
ticker_returns = results['tickers_returns']

# Strategy performance
strategy_returns = results['strategy_returns']

# Benchmark performance
benchmark_returns = results['benchmark_returns']
```

These components contain all the information you need to analyze your strategy in any capacity.

## Analyzing Performance

Portwine comes with built-in analyzers to help you understand your strategy's performance:

```python
from portwine.analyzers import EquityDrawdownAnalyzer, MonteCarloAnalyzer

# Equity and drawdown analysis
EquityDrawdownAnalyzer().plot(results)

# Monte Carlo simulation
MonteCarloAnalyzer().plot(results)
```

For more information on the available analyzers and how to write your own analyzer, [click here](analyzers.md).

## What's Happening Under the Hood

1. **Data Loading**: The data loader fetches historical price data for your universe
2. **Strategy Execution**: Each day, your strategy receives the latest prices and decides allocations
3. **Signal Processing**: Portwine handles the mechanics of applying your signals to the market
4. **Performance Calculation**: Returns are calculated and compared against your benchmark

## Next Steps

- Learn more about [building strategies](user-guide/strategies.md)
- Explore [different analyzers](user-guide/analysis.md)
- Check out [advanced examples](examples/advanced-strategies.md) 