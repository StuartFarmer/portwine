# portwine - a clean, elegant portfolio backtester
![The Triumph of Bacchus](imgs/header.jpg)
Portfolio construction, optimization, and backtesting can be a complicated web of data wrangling, signal generation, lookahead bias reduction, and parameter tuning.

But with `portwine`, strategies are clear and written in an 'online' fashion that removes most of the complexity that comes with backtesting, analyzing, and deploying your trading strategies.

---

### Simple Strategies
Strategies are only given the last day of prices to make their determinations and allocate weights. This allows them to be completely encapsulated and portable.

```python
class SimpleMomentumStrategy(StrategyBase):
    """
    A simple momentum strategy that:
    1. Calculates N-day momentum for each ticker
    2. Invests in the top performing ticker
    3. Rebalances weekly (every Friday)
    
    This demonstrates a step-based strategy implementation in a concise, easy-to-understand way.
    """
    
    def __init__(self, tickers, lookback_days=10):
        """
        Parameters
        ----------
        tickers : list
            List of ticker symbols to consider for investment
        lookback_days : int
            Number of days to use for momentum calculation
        """
        super().__init__(tickers)
        self.lookback_days = lookback_days
        self.price_history = {ticker: [] for ticker in tickers}
        self.current_signals = {ticker: 0.0 for ticker in tickers}
        self.dates = []
    
    def is_friday(self, date):
        """Check if given date is a Friday (weekday 4)"""
        return date.weekday() == 4
    
    def calculate_momentum(self, ticker):
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
    
    def step(self, current_date, daily_data):
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

---

### Breezy Backtesting

Backtesting strategies is a breeze, as well. Simply tell the backtester where your data is located with a data loader manager and give it a strategy. You get results immediately.

```python
universe = ['MTUM', 'VTV', 'VUG', 'IJR', 'MDY']
strategy = SimpleMomentumStrategy(tickers=universe, lookback_days=10)

data_loader = EODHDMarketDataLoader(data_path='../../../Developer/Data/EODHD/us_sorted/US/')
backtester = Backtester(market_data_loader=data_loader)
results = backtester.run_backtest(strategy, benchmark_ticker='SPY')
```
---
### Streamlined Data
Managing data can be a massive pain. But as long as you have your daily flat files from EODHD or Polygon saved in a directory, the data loaders will manage the rest. You don't have to worry about anything except writing code.

---

### Effortless Analysis

After running a strategy through the backtester, put it through an array of analyzers that are simple, visual, and clear. You can easily add your own analyzers to discover anything you need to know about your portfolio's performance, risk management, volatility, etc.

Check out what comes out of the box:

##### Equity Drawdown Analysis
```python
EquityDrawdownAnalyzer().plot(results)
```
![Equity Drawdown](imgs/equitydrawdown.jpg)

---
##### Monte Carlo Analysis
```python
MonteCarloAnalyzer().plot(results)
```
![Equity Drawdown](imgs/montecarlo.jpg)

---
##### Seasonality Analysis
```python
EquityDrawdownAnalyzer().plot(results)
```
![Equity Drawdown](imgs/seasonality.jpg)

With more on the way!

---
*Docs and tutorials coming soon.*
