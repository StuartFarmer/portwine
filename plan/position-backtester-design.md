# Position-Based Backtester - Design Document

**Branch**: `position-backtester`
**Status**: Planning Phase
**Goal**: Add explicit position-based backtesting alongside existing weights-based system

---

## Executive Summary

This document outlines the design for a new **position-based backtesting system** that runs parallel to the existing weights-based backtester. Instead of portfolio weights (e.g., `{'AAPL': 0.25}`), strategies will output position changes (e.g., `{'AAPL': 10}` = buy 10 shares).

**Key Differences from Current System**:
- **Weights-based**: Allocate 25% of capital to AAPL → normalized equity curve starting at 1.0
- **Position-based**: Buy 10 shares of AAPL → explicit monetary P&L tracking with infinite capital

**Design Principles**:
1. **API Compatibility**: Minimal changes to strategy interface and backtest flow
2. **Parallel Systems**: Position-based system coexists with weights-based (no replacement)
3. **Infinite Capital**: Assume unlimited buying power; track unrealized/realized P&L explicitly
4. **Analyzer Compatibility**: Similar output structure to enable analyzer adaptation

---

## Current System Analysis

### Weights-Based Backtesting Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Strategy.step() returns Dict[str, float]                │
│    Example: {'AAPL': 0.3, 'MSFT': 0.7}                     │
│    Meaning: Allocate 30% to AAPL, 70% to MSFT              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Backtester validates signals                            │
│    - Total weight <= 1.0                                    │
│    - All tickers in universe                                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Collect signals into DataFrame (days × tickers)         │
│    - One row per trading day                                │
│    - One column per ticker                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Calculate ticker returns from close prices               │
│    - returns[t] = (close[t] - close[t-1]) / close[t-1]     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Calculate strategy returns with signal shifting         │
│    - Shift signals forward by 1 day (avoid lookahead)      │
│    - strategy_return[t] = Σ(weight[t-1] × return[t])       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Return results dict                                     │
│    {                                                        │
│      'signals_df': DataFrame,       # weights              │
│      'tickers_returns': DataFrame,  # daily returns        │
│      'strategy_returns': Series,    # portfolio returns    │
│      'benchmark_returns': Series    # benchmark returns    │
│    }                                                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Analyzers compute equity curves ON-DEMAND                │
│    equity = (1 + strategy_returns).cumprod()                │
│    - Starts at 1.0 (normalized)                             │
│    - Value of 2.0 = 100% gain                               │
│    - Value of 0.5 = 50% loss                                │
└─────────────────────────────────────────────────────────────┘
```

**Key Properties**:
- **Percentage-based**: All returns are percentages (e.g., 0.05 = 5% gain)
- **Normalized equity**: Equity curve always starts at 1.0
- **Capital-agnostic**: No concept of dollar amounts or cash balance
- **Position-agnostic**: No tracking of number of shares held

---

## Position-Based System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PositionStrategy.step() returns Dict[str, int/float]    │
│    Example: {'AAPL': 10, 'MSFT': -5}                       │
│    Meaning: Buy 10 shares AAPL, sell 5 shares MSFT         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PositionBacktester validates actions                    │
│    - All tickers in universe                                │
│    - Position changes are numeric (int/float)               │
│    - NO capital constraint (infinite cash)                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Update position tracking                                │
│    - positions[ticker] += action[ticker]                    │
│    - Track: current positions, actions, prices              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Calculate P&L (unrealized + realized)                   │
│    For each ticker:                                         │
│      unrealized_pnl = position × (current_price - avg_cost) │
│      realized_pnl = Σ(closed_position_pnl)                  │
│    Total portfolio P&L = Σ(unrealized + realized)           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Calculate dollar returns (NOT percentages)               │
│    dollar_return[t] = portfolio_value[t] - portfolio_value[t-1]
│    - No normalization                                       │
│    - Explicit dollar amounts                                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Return position-based results dict                      │
│    {                                                        │
│      'positions_df': DataFrame,     # shares held          │
│      'actions_df': DataFrame,       # buy/sell actions     │
│      'prices_df': DataFrame,        # execution prices     │
│      'unrealized_pnl': Series,      # daily unrealized P&L │
│      'realized_pnl': Series,        # cumulative realized  │
│      'total_pnl': Series,           # unrealized + realized│
│      'portfolio_value': Series,     # total portfolio value│
│      'benchmark_returns': Series    # benchmark (% or $?)  │
│    }                                                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. PositionAnalyzers consume results                       │
│    - Equity curve = total_pnl (starts at 0.0)               │
│    - OR portfolio_value (starts at initial margin req)      │
│    - Different metrics: max position size, turnover, etc.   │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Design

### 1. Strategy Interface

#### Current: StrategyBase (Weights)
```python
class StrategyBase(abc.ABC):
    def step(self, current_date, daily_data) -> Dict[str, float]:
        """
        Returns portfolio weights.

        Returns:
            Dict[str, float]: {ticker: weight} where sum(weights) <= 1.0
        """
        pass
```

#### Proposed: PositionStrategyBase
```python
class PositionStrategyBase(abc.ABC):
    def __init__(self, tickers: Union[List[str], Universe]):
        self.universe = tickers if isinstance(tickers, Universe) else self._create_static_universe(tickers)
        self._positions = {}  # Track current positions for strategy logic

    @property
    def positions(self) -> Dict[str, float]:
        """Current positions (read-only view for strategy)."""
        return dict(self._positions)

    def step(self, current_date, daily_data) -> Dict[str, float]:
        """
        Returns position changes (BUY/SELL actions).

        Args:
            current_date: Current timestamp
            daily_data: RestrictedDataInterface for current universe

        Returns:
            Dict[str, float]: {ticker: shares_to_buy_or_sell}
                - Positive = BUY shares
                - Negative = SELL shares
                - Can go short (negative positions allowed)
                - Can use fractional shares

        Example:
            {'AAPL': 10}     # Buy 10 shares of AAPL
            {'AAPL': -5}     # Sell 5 shares of AAPL
            {'AAPL': 10, 'MSFT': -20}  # Buy AAPL, sell MSFT
        """
        pass

    def _update_positions(self, actions: Dict[str, float]):
        """Called by backtester after each step (internal use)."""
        for ticker, action in actions.items():
            self._positions[ticker] = self._positions.get(ticker, 0) + action
```

**Design Decision: Position Tracking in Strategy**
- Strategy has read-only access to current positions via `self.positions`
- Backtester updates `self._positions` after validation
- Enables strategies like: "Buy more if position < threshold"

**Alternative Approach** (rejected):
- Strategy manages positions entirely
- **Problem**: Backtester can't validate or track positions centrally
- **Problem**: Harder to ensure consistency

---

### 2. Position Backtester Core

#### File Location
`portwine/backtester/position_core.py` (new file, parallel to `core.py`)

#### Key Classes

```python
class PositionBacktestResult:
    """Stores position-based backtest results."""

    def __init__(self, datetime_index, tickers):
        self.datetime_index = datetime_index
        self.tickers = sorted(tickers)

        # Position tracking
        self.positions_array = np.zeros((len(datetime_index), len(tickers)))
        self.actions_array = np.zeros((len(datetime_index), len(tickers)))
        self.prices_array = np.full((len(datetime_index), len(tickers)), np.nan)

        # P&L tracking
        self.unrealized_pnl = np.zeros(len(datetime_index))
        self.realized_pnl = np.zeros(len(datetime_index))
        self.total_pnl = np.zeros(len(datetime_index))
        self.portfolio_value = np.zeros(len(datetime_index))

        # Benchmark
        self.benchmark_returns = np.zeros(len(datetime_index))

    def calculate_pnl(self):
        """Calculate unrealized and realized P&L using Numba JIT."""
        self._calculate_position_pnl(
            self.positions_array,
            self.actions_array,
            self.prices_array,
            self.unrealized_pnl,
            self.realized_pnl,
            self.total_pnl,
            self.portfolio_value
        )

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_position_pnl(positions, actions, prices,
                                 unrealized_pnl, realized_pnl,
                                 total_pnl, portfolio_value):
        """
        Numba-optimized P&L calculation.

        For each day and ticker:
          1. Track average cost basis
          2. Calculate unrealized P&L on current position
          3. Calculate realized P&L on closed positions
          4. Sum across all tickers for portfolio totals
        """
        n_days, n_tickers = positions.shape

        # Track cost basis per ticker
        cost_basis = np.zeros(n_tickers)
        cumulative_realized = 0.0

        for day in range(n_days):
            day_unrealized = 0.0

            for ticker_idx in range(n_tickers):
                pos = positions[day, ticker_idx]
                action = actions[day, ticker_idx]
                price = prices[day, ticker_idx]

                if np.isnan(price):
                    continue

                # Update cost basis when position changes
                if action != 0:
                    if pos == action:  # Opening new position
                        cost_basis[ticker_idx] = price
                    else:  # Adding to or closing position
                        # FIFO or average cost basis logic
                        prev_pos = pos - action
                        if prev_pos * action > 0:  # Same direction
                            # Average cost basis
                            total_cost = (cost_basis[ticker_idx] * abs(prev_pos) +
                                         price * abs(action))
                            cost_basis[ticker_idx] = total_cost / abs(pos)
                        else:  # Closing or reversing
                            # Realize P&L on closed portion
                            closed_shares = min(abs(prev_pos), abs(action))
                            if prev_pos > 0:  # Closing long
                                pnl = closed_shares * (price - cost_basis[ticker_idx])
                            else:  # Closing short
                                pnl = closed_shares * (cost_basis[ticker_idx] - price)
                            cumulative_realized += pnl

                            # Update cost basis for remaining position
                            if abs(action) > abs(prev_pos):  # Reversing
                                cost_basis[ticker_idx] = price

                # Calculate unrealized P&L on current position
                if pos != 0:
                    if pos > 0:  # Long position
                        pnl = pos * (price - cost_basis[ticker_idx])
                    else:  # Short position
                        pnl = abs(pos) * (cost_basis[ticker_idx] - price)
                    day_unrealized += pnl

            unrealized_pnl[day] = day_unrealized
            realized_pnl[day] = cumulative_realized
            total_pnl[day] = day_unrealized + cumulative_realized
            portfolio_value[day] = total_pnl[day]  # Starting capital = 0

    def to_dict(self) -> dict:
        """Convert to results dictionary."""
        return {
            'positions_df': pd.DataFrame(
                self.positions_array,
                index=self.datetime_index,
                columns=self.tickers
            ),
            'actions_df': pd.DataFrame(
                self.actions_array,
                index=self.datetime_index,
                columns=self.tickers
            ),
            'prices_df': pd.DataFrame(
                self.prices_array,
                index=self.datetime_index,
                columns=self.tickers
            ),
            'unrealized_pnl': pd.Series(
                self.unrealized_pnl,
                index=self.datetime_index,
                name='unrealized_pnl'
            ),
            'realized_pnl': pd.Series(
                self.realized_pnl,
                index=self.datetime_index,
                name='realized_pnl'
            ),
            'total_pnl': pd.Series(
                self.total_pnl,
                index=self.datetime_index,
                name='total_pnl'
            ),
            'portfolio_value': pd.Series(
                self.portfolio_value,
                index=self.datetime_index,
                name='portfolio_value'
            ),
            'benchmark_returns': pd.Series(
                self.benchmark_returns,
                index=self.datetime_index,
                name='benchmark_returns'
            )
        }


class PositionBacktester:
    """Position-based backtester (parallel to Backtester)."""

    def __init__(self, data_interface, calendar=None):
        self.data = data_interface
        self.calendar = calendar or DailyMarketCalendar()

    def run_backtest(
        self,
        strategy: PositionStrategyBase,
        start_date=None,
        end_date=None,
        benchmark=None,
        execution_price='close',  # 'open', 'close', 'average'
        verbose=False
    ):
        """
        Run position-based backtest.

        Args:
            strategy: PositionStrategyBase instance
            start_date: Start date (auto-detect if None)
            end_date: End date (auto-detect if None)
            benchmark: Benchmark specification (ticker, callable, or None)
            execution_price: Which price to use for executions
            verbose: Print progress

        Returns:
            dict: Position-based results
        """
        # Similar structure to Backtester.run_backtest()

        # 1. Validate strategy
        # 2. Determine date range
        # 3. Get datetime index
        # 4. Create PositionBacktestResult
        # 5. Loop through dates:
        #    - Update universe
        #    - Call strategy.step()
        #    - Validate actions
        #    - Update positions
        #    - Record prices
        #    - Update strategy._positions
        # 6. Calculate P&L
        # 7. Calculate benchmark
        # 8. Return results dict

        pass

    def validate_actions(self, actions: Dict[str, float],
                        current_universe_tickers: List[str]):
        """
        Validate position actions.

        Checks:
          - All tickers exist in current universe
          - Actions are numeric (int or float)
          - NO capital constraint (infinite cash assumption)
        """
        for ticker in actions.keys():
            if ticker not in current_universe_tickers:
                raise ValueError(f"Ticker {ticker} not in universe")

        for ticker, action in actions.items():
            if not isinstance(action, (int, float)):
                raise ValueError(f"Action for {ticker} must be numeric")
            if np.isnan(action) or np.isinf(action):
                raise ValueError(f"Invalid action for {ticker}: {action}")
```

---

### 3. Benchmark Handling

**Challenge**: Current benchmarks return percentage-based returns. Position-based system uses dollar amounts.

**Solutions**:

#### Option 1: Dual-Mode Benchmarks (Recommended)
```python
# Benchmark returns percentage-based returns as usual
# Convert to dollar-equivalent for comparison

def _calculate_position_benchmark(benchmark_pct_returns, total_pnl):
    """
    Convert percentage benchmark to dollar-equivalent.

    Strategy: Apply benchmark returns to a "virtual portfolio" with same
    capital deployment as actual strategy.

    Example:
      - Strategy has $10,000 deployed (sum of abs(position × price))
      - Benchmark return = 5%
      - Benchmark dollar return = $10,000 × 0.05 = $500
    """
    pass
```

#### Option 2: Position-Based Benchmark Strategy
```python
class PositionBenchmarkStrategy(PositionStrategyBase):
    """Equal-weight buy-and-hold benchmark in position space."""

    def __init__(self, tickers, initial_shares_per_ticker=100):
        super().__init__(tickers)
        self.initial_shares = initial_shares_per_ticker
        self._initialized = False

    def step(self, current_date, daily_data):
        if not self._initialized:
            # Buy initial positions on first day
            self._initialized = True
            return {ticker: self.initial_shares for ticker in self.tickers}
        return {}  # Hold forever
```

#### Option 3: No Benchmark Conversion (Simplest)
```python
# Keep benchmark as percentage returns
# Let analyzers decide how to compare:
#   - Convert strategy to percentage returns (total_pnl / deployed_capital)
#   - Or show separately (strategy in $, benchmark in %)
```

**Recommendation**: Start with **Option 3** (simplest), add conversion later if needed.

---

### 4. Output API Design

#### Current Output (Weights-Based)
```python
{
    'signals_df': DataFrame,        # (days × tickers) portfolio weights
    'tickers_returns': DataFrame,   # (days × tickers) daily % returns
    'strategy_returns': Series,     # (days,) daily portfolio % returns
    'benchmark_returns': Series     # (days,) daily benchmark % returns
}
```

#### Proposed Output (Position-Based)
```python
{
    # Core position data
    'positions_df': DataFrame,      # (days × tickers) shares held
    'actions_df': DataFrame,        # (days × tickers) buy/sell actions
    'prices_df': DataFrame,         # (days × tickers) execution prices

    # P&L data
    'unrealized_pnl': Series,       # (days,) daily unrealized P&L ($)
    'realized_pnl': Series,         # (days,) cumulative realized P&L ($)
    'total_pnl': Series,            # (days,) total P&L ($)
    'portfolio_value': Series,      # (days,) total portfolio value ($)

    # Benchmark (optional)
    'benchmark_returns': Series,    # (days,) benchmark % returns

    # Optional: ticker-level P&L
    'ticker_unrealized_pnl': DataFrame,  # (days × tickers) per-ticker unrealized
    'ticker_realized_pnl': DataFrame,    # (days × tickers) per-ticker realized
}
```

**Similarities to Weights-Based**:
- Same DataFrame structure (days × tickers)
- Same Series structure (days,)
- Same datetime index
- Similar column structure

**Differences**:
- `positions_df` instead of `signals_df` (shares vs weights)
- `actions_df` is NEW (buy/sell actions)
- `prices_df` is NEW (execution prices)
- P&L in dollars, not percentages
- No `tickers_returns` (can be calculated from prices_df if needed)
- No `strategy_returns` (replaced by `total_pnl`)

**API Compatibility Score**: ~60%
- DataFrames and Series structure preserved
- Column semantics completely different
- Analyzers need substantial rewrites

---

### 5. Execution Prices

**Question**: When do we execute trades and at what price?

**Current System (Weights)**:
- Uses **close prices** exclusively
- Signal shifting: today's signal → tomorrow's returns
- Implicit assumption: execute at tomorrow's open (approximately)

**Position System Options**:

#### Option A: Close-to-Close (Simplest)
```python
# Day 1 close: Strategy decides to buy 10 AAPL
# Day 2 open: Execute at Day 2 close price
# Same as current system's signal shifting
```

**Pros**: Matches current system, simple
**Cons**: Unrealistic (can't execute at close after seeing close price)

#### Option B: Close-to-Open (More Realistic)
```python
# Day 1 close: Strategy decides to buy 10 AAPL based on Day 1 close
# Day 2 open: Execute at Day 2 open price
```

**Pros**: More realistic execution model
**Cons**: Requires open prices in data

#### Option C: Configurable Execution Price
```python
def run_backtest(..., execution_price='close'):
    """
    execution_price options:
      - 'close': Execute at same day's close (unrealistic but consistent)
      - 'next_open': Execute at next day's open (realistic)
      - 'next_close': Execute at next day's close (current system equivalent)
      - 'average': Execute at (open + close) / 2
    """
```

**Recommendation**: Implement **Option C** with default `'next_close'` (matches current system).

---

### 6. Cost Basis Tracking

**Challenge**: Track average cost basis for P&L calculation when positions are accumulated.

**Example**:
```
Day 1: Buy 10 AAPL @ $150 → Cost basis = $150
Day 2: Buy 5 AAPL @ $155 → Cost basis = (10×$150 + 5×$155) / 15 = $151.67
Day 3: Sell 5 AAPL @ $160 → Realized P&L = 5 × ($160 - $151.67) = $41.65
                            → Remaining 10 @ $151.67 cost basis
```

**Accounting Methods**:

1. **Average Cost** (Recommended for simplicity)
   - Average all buys into single cost basis
   - Simple calculation
   - Used in mutual funds

2. **FIFO (First In, First Out)**
   - Sell oldest shares first
   - More complex tracking (need queue)
   - Required for tax reporting

3. **LIFO (Last In, First Out)**
   - Sell newest shares first
   - Also complex
   - Less common

**Recommendation**: Start with **Average Cost**, add FIFO option later.

**Short Position Handling**:
```
Day 1: Sell short 10 AAPL @ $150 → Cost basis = $150
Day 2: Buy to cover 5 AAPL @ $145 → Realized P&L = 5 × ($150 - $145) = $25
Day 3: Sell short 5 more AAPL @ $155 → New cost basis for -10 position?
```

**Solution**: Track cost basis separately for long and short positions. When reversing (long→short or short→long), close old position and open new one.

---

### 7. Infinite Capital Assumption

**Philosophy**: Position-based system assumes **infinite capital** to focus on strategy logic.

**Implications**:

1. **No Margin Calls**
   - Can hold any position size
   - No forced liquidations

2. **No Borrowing Costs**
   - Shorting is free
   - Leverage is free

3. **No Cash Tracking**
   - No need to track cash balance
   - All P&L is paper gains/losses

4. **Post-Analysis Capital Calculation**
   ```python
   # After backtest, calculate capital required
   max_deployed_capital = max(Σ|position[t] × price[t]|)
   max_drawdown = min(total_pnl[t])
   required_margin = max_deployed_capital + abs(max_drawdown)
   ```

**Alternative Approach** (Future Enhancement):
```python
def run_backtest(..., initial_capital=None):
    """
    If initial_capital is provided:
      - Track cash balance
      - Reject trades that exceed capital
      - Apply margin requirements

    If initial_capital is None (default):
      - Infinite capital mode
      - Pure strategy analysis
    """
```

**Recommendation**: Start with **infinite capital**, add capital constraints as optional feature later.

---

### 8. Analyzer Migration Strategy

**Current Analyzers** (25+ classes):
- All assume percentage returns
- All compute equity curves as `(1 + returns).cumprod()`
- All expect `strategy_returns` Series

**Migration Approaches**:

#### Approach 1: Adapter Pattern (Recommended)
```python
class PositionToWeightsAdapter:
    """Convert position-based results to weights-based format."""

    @staticmethod
    def convert(position_results: dict, deployed_capital: float) -> dict:
        """
        Convert position results to weights-based format.

        Args:
            position_results: Position-based backtest results
            deployed_capital: Total capital to normalize against

        Returns:
            Weights-based format for legacy analyzers
        """
        # Calculate percentage returns from dollar P&L
        pnl = position_results['total_pnl']
        strategy_returns = pnl.diff() / deployed_capital

        # Convert positions to weights
        positions = position_results['positions_df']
        prices = position_results['prices_df']
        position_values = positions * prices
        total_value = position_values.sum(axis=1)
        signals_df = position_values.div(total_value, axis=0)

        return {
            'signals_df': signals_df,
            'strategy_returns': strategy_returns,
            'benchmark_returns': position_results['benchmark_returns'],
            'tickers_returns': prices.pct_change()  # Approximate
        }

# Usage
position_results = position_backtester.run_backtest(strategy)
weights_results = PositionToWeightsAdapter.convert(
    position_results,
    deployed_capital=100000
)
analyzer = EquityDrawdownAnalyzer()
analyzer.analyze(weights_results)  # Works with existing analyzer!
```

**Pros**: Reuse all existing analyzers immediately
**Cons**: Requires choosing a `deployed_capital` value (somewhat arbitrary)

#### Approach 2: Position-Specific Analyzers
```python
class PositionEquityAnalyzer(Analyzer):
    """Equity analysis for position-based results."""

    def analyze(self, results: dict):
        total_pnl = results['total_pnl']
        portfolio_value = results['portfolio_value']

        # Metrics
        max_pnl = total_pnl.max()
        min_pnl = total_pnl.min()
        final_pnl = total_pnl.iloc[-1]

        # Drawdown in dollars (not percentage)
        running_max = total_pnl.cummax()
        drawdown_dollars = total_pnl - running_max
        max_drawdown_dollars = drawdown_dollars.min()

        return {
            'max_pnl': max_pnl,
            'min_pnl': min_pnl,
            'final_pnl': final_pnl,
            'max_drawdown_dollars': max_drawdown_dollars,
            'total_pnl_series': total_pnl
        }

    def plot(self, results: dict):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve (total P&L)
        results['total_pnl'].plot(ax=axes[0], label='Total P&L')
        results['realized_pnl'].plot(ax=axes[0], label='Realized P&L', alpha=0.7)
        axes[0].set_ylabel('P&L ($)')
        axes[0].legend()

        # Position sizes over time
        positions = results['positions_df']
        prices = results['prices_df']
        position_values = (positions * prices).abs().sum(axis=1)
        position_values.plot(ax=axes[1])
        axes[1].set_ylabel('Deployed Capital ($)')

        plt.tight_layout()
        return fig
```

**Pros**: Tailored metrics for position-based strategies
**Cons**: Need to rewrite all 25+ analyzers

#### Approach 3: Hybrid (Recommended)
```python
# 1. Create adapter for quick compatibility
# 2. Create position-specific analyzers for new metrics
# 3. Gradually migrate existing analyzers

# Use adapter for quick wins
weights_results = PositionToWeightsAdapter.convert(position_results, 100000)
sharpe = SharpeRatioAnalyzer().analyze(weights_results)

# Use position analyzers for new insights
position_metrics = PositionEquityAnalyzer().analyze(position_results)
turnover = PositionTurnoverAnalyzer().analyze(position_results)
```

---

### 9. New Position-Specific Metrics

**Metrics NOT possible in weights-based system**:

1. **Position Turnover (in shares)**
   ```python
   daily_turnover = abs(actions_df).sum(axis=1)
   total_turnover = daily_turnover.sum()
   avg_daily_turnover = daily_turnover.mean()
   ```

2. **Deployed Capital Over Time**
   ```python
   deployed = (positions_df.abs() * prices_df).sum(axis=1)
   max_deployed = deployed.max()
   avg_deployed = deployed.mean()
   ```

3. **Position Concentration**
   ```python
   # Number of positions held
   num_positions = (positions_df != 0).sum(axis=1)

   # Largest position as % of deployed capital
   position_values = positions_df * prices_df
   deployed = position_values.abs().sum(axis=1)
   max_position_pct = position_values.abs().max(axis=1) / deployed
   ```

4. **Trade Analysis**
   ```python
   # Extract individual trades from actions_df
   trades = extract_trades(actions_df, prices_df)
   # [{ticker, entry_date, exit_date, shares, entry_price, exit_price, pnl}]

   win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades)
   avg_win = mean([t['pnl'] for t in trades if t['pnl'] > 0])
   avg_loss = mean([t['pnl'] for t in trades if t['pnl'] < 0])
   ```

5. **Holding Period Analysis**
   ```python
   avg_holding_period = mean([t['exit_date'] - t['entry_date'] for t in trades])
   ```

6. **Long/Short Exposure**
   ```python
   long_exposure = (positions_df.clip(lower=0) * prices_df).sum(axis=1)
   short_exposure = (positions_df.clip(upper=0).abs() * prices_df).sum(axis=1)
   net_exposure = long_exposure - short_exposure
   gross_exposure = long_exposure + short_exposure
   ```

**These metrics are IMPOSSIBLE in weights-based system** because:
- No concept of number of shares
- No concept of trade entry/exit
- No distinction between initial position and rebalancing

---

## API Comparison Summary

### Strategy Interface

| Aspect | Weights-Based | Position-Based |
|--------|---------------|----------------|
| Base Class | `StrategyBase` | `PositionStrategyBase` |
| `step()` returns | `Dict[str, float]` weights | `Dict[str, float]` shares |
| Constraint | `sum(weights) <= 1.0` | None (infinite capital) |
| Position tracking | None | `self.positions` property |
| Validation | Total weight <= 1.0 | All tickers in universe |

### Backtest Results

| Field | Weights-Based | Position-Based |
|-------|---------------|----------------|
| Positions/Weights | `signals_df` (weights) | `positions_df` (shares) |
| Actions | N/A | `actions_df` (buy/sell) |
| Prices | Implicit in returns | `prices_df` (execution prices) |
| Returns | `strategy_returns` (%) | `total_pnl` ($) |
| Ticker returns | `tickers_returns` (%) | `prices_df.pct_change()` |
| Benchmark | `benchmark_returns` (%) | `benchmark_returns` (%) |
| P&L breakdown | N/A | `unrealized_pnl`, `realized_pnl` |

### Equity Curves

| Aspect | Weights-Based | Position-Based |
|--------|---------------|----------------|
| Formula | `(1 + returns).cumprod()` | `total_pnl` (direct) |
| Starting value | 1.0 (normalized) | 0.0 (no initial capital) |
| Units | Dimensionless (multiplier) | Dollars |
| Interpretation | 2.0 = 100% gain | $10,000 = $10k profit |

### Analyzers

| Aspect | Weights-Based | Position-Based |
|--------|---------------|----------------|
| Count | 25+ existing | New (TBD) |
| Compatibility | 100% | 0% without adapter |
| Adapter available | N/A | Yes (`PositionToWeightsAdapter`) |
| New metrics | N/A | Turnover, deployed capital, trades |

---

## Potential Issues and Edge Cases

### 1. Fractional Shares

**Question**: Allow fractional shares (e.g., 10.5 shares)?

**Options**:
- **Yes**: More flexible, matches real brokers (Robinhood, etc.)
- **No**: Simpler, matches traditional trading

**Recommendation**: **Allow fractional shares** (use `float`, not `int`).

---

### 2. Short Selling

**Question**: Allow negative positions (short selling)?

**Answer**: **Yes**, definitely allow shorts.

**Implementation**:
```python
# Positive position = long
# Negative position = short
# Zero = flat

position = 10   # Long 10 shares
position = -10  # Short 10 shares
position = 0    # No position
```

**P&L Calculation**:
```python
# Long: profit when price rises
long_pnl = shares × (current_price - cost_basis)

# Short: profit when price falls
short_pnl = abs(shares) × (cost_basis - current_price)
```

---

### 3. Price Gaps and Execution

**Problem**: Price gaps between decision and execution.

**Example**:
```
Day 1 close: $100 → Strategy decides to buy 10 shares
Day 2 open: $110 (gap up 10%)
```

**Current System**: Signal shifting handles this automatically (today's signal → tomorrow's return).

**Position System**: Need to be explicit about execution price.

**Solution**: Use configurable execution price (see Section 5).

---

### 4. Dividends and Corporate Actions

**Question**: How to handle dividends, splits, spin-offs?

**Current System**: Data is pre-adjusted (split-adjusted, dividend-adjusted).

**Position System**: Same approach.

**Future Enhancement**: Track dividends separately for cash flow analysis.

```python
# Future feature
{
    'dividend_df': DataFrame,  # (days × tickers) dividend payments
    'cash_flows': Series,      # (days,) net cash flows
}
```

---

### 5. Benchmark Comparison

**Problem**: Strategy in dollars, benchmark in percentages.

**Solutions** (see Section 3):
1. Convert benchmark to dollars (complex)
2. Convert strategy to percentages (requires deployed capital)
3. Keep separate (show both)

**Recommendation**: Keep separate initially, add conversion later.

---

### 6. Transaction Costs

**Current System**: `TransactionCostAnalyzer` applies costs post-hoc.

**Position System**: Same approach.

```python
# Calculate turnover in dollars
turnover_dollars = (abs(actions_df) * prices_df).sum()

# Apply cost per share or percentage
cost_per_share = 0.01  # $0.01 per share
total_costs = (abs(actions_df) * cost_per_share).sum()

# Adjust P&L
net_pnl = total_pnl - total_costs
```

**Future Enhancement**: Apply costs during backtest, not post-hoc.

---

### 7. Data Requirements

**Current System**: Requires close prices only.

**Position System**:
- Minimum: close prices (same as current)
- Better: open and close (for realistic execution)
- Best: OHLCV (for slippage modeling)

**Recommendation**: Start with close prices, add open prices later.

---

### 8. Universe Changes

**Current System**: Handles dynamic universes (constituents change over time).

**Position System**: **Challenge** - what happens to positions when ticker exits universe?

**Example**:
```
Day 1: Hold 100 shares AAPL (in universe)
Day 2: AAPL removed from universe
  → Force liquidation?
  → Hold position but can't trade?
  → Error?
```

**Solution**:
```python
def handle_universe_exit(ticker, position, price):
    """
    When ticker exits universe:
    1. Force liquidation at last available price
    2. Realize P&L
    3. Log warning
    """
    if position != 0:
        logger.warning(f"{ticker} exited universe with {position} shares held. "
                      f"Forced liquidation at ${price}")
        # Sell/cover entire position
        action = -position
        # Calculate realized P&L
        # ...
```

---

### 9. Initial Positions

**Question**: Start with existing positions or empty portfolio?

**Current System**: Always starts with no positions (weights = 0).

**Position System**: Same, but allow override.

```python
def run_backtest(..., initial_positions=None):
    """
    initial_positions: Dict[str, float] = initial position sizes

    Example:
      initial_positions = {'AAPL': 100, 'MSFT': 50}
      # Start backtest with 100 AAPL, 50 MSFT already held
    """
```

**Use Cases**:
- Testing "what if I held this position?" scenarios
- Continuing from previous backtest

**Recommendation**: Allow optional initial positions (default empty).

---

### 10. Performance Optimization

**Current System**: Uses Numba JIT for returns calculation (~10x speedup).

**Position System**: More complex calculations (cost basis tracking, P&L).

**Optimization Strategy**:
1. **First pass**: Pure Python, correct implementation
2. **Second pass**: Numba-fy hot loops (P&L calculation)
3. **Third pass**: Profile and optimize bottlenecks

**Expected Performance**:
- Slower than weights-based (more complex)
- Still acceptable (target: <1s for 1000 days, 100 tickers)

---

## Implementation Plan

### Phase 1: Core Position Backtester (Week 1-2)

**Files to Create**:
1. `portwine/strategies/position_base.py` - `PositionStrategyBase`
2. `portwine/backtester/position_core.py` - `PositionBacktester`, `PositionBacktestResult`
3. `tests/test_position_backtester.py` - Unit tests

**Deliverables**:
- Working position backtester
- Simple example strategy (buy-and-hold)
- Tests passing

**Success Criteria**:
```python
# Example usage works
strategy = BuyAndHoldPositionStrategy(['AAPL', 'MSFT'], shares_per_ticker=100)
backtester = PositionBacktester(data)
results = backtester.run_backtest(strategy, '2020-01-01', '2023-12-31')

assert 'positions_df' in results
assert 'total_pnl' in results
assert len(results['total_pnl']) > 0
```

---

### Phase 2: P&L Calculation (Week 2-3)

**Tasks**:
1. Implement cost basis tracking (average cost method)
2. Calculate unrealized and realized P&L
3. Numba optimization for hot loops
4. Validate against hand-calculated examples

**Deliverables**:
- Accurate P&L calculation
- Unit tests for edge cases (shorts, reversals, etc.)
- Performance benchmarks

**Success Criteria**:
- P&L matches hand-calculated examples (within floating point precision)
- Performance acceptable (<5s for 5 years, 50 tickers)

---

### Phase 3: Position Analyzers (Week 3-4)

**Tasks**:
1. Create `PositionToWeightsAdapter` for backward compatibility
2. Create core position analyzers:
   - `PositionEquityAnalyzer` - P&L curves, drawdowns
   - `PositionTurnoverAnalyzer` - Turnover metrics
   - `PositionTradeAnalyzer` - Individual trade stats
   - `PositionExposureAnalyzer` - Long/short/net/gross exposure

**Deliverables**:
- 4 position-specific analyzers
- Adapter for legacy analyzers
- Example notebook demonstrating usage

**Success Criteria**:
```python
# Adapter works with legacy analyzers
weights_results = PositionToWeightsAdapter.convert(position_results, 100000)
sharpe = SharpeRatioAnalyzer().analyze(weights_results)  # Works!

# Position analyzers provide new insights
trades = PositionTradeAnalyzer().analyze(position_results)
assert 'win_rate' in trades
assert 'avg_holding_period' in trades
```

---

### Phase 4: Example Strategies (Week 4-5)

**Tasks**:
1. Port existing strategies to position-based:
   - `SimpleMovingAverageCrossover` (weights) → `PositionSMACrossover`
   - `MeanReversion` (weights) → `PositionMeanReversion`
2. Create new position-specific strategies:
   - `FixedShareBuyAndHold` - Buy N shares and hold
   - `PercentileRebalance` - Rebalance to percentile-based targets
   - `PairsTradingPosition` - Market-neutral pairs trading

**Deliverables**:
- 5 working position strategies
- Examples showing equivalent weights vs positions strategies
- Performance comparison

**Success Criteria**:
- Strategies run successfully
- Results make intuitive sense
- Documentation explains usage

---

### Phase 5: Benchmarks and Validation (Week 5-6)

**Tasks**:
1. Implement benchmark handling (decide on Option 1, 2, or 3 from Section 3)
2. Create validation suite:
   - Compare weights-based vs position-based on same strategy
   - Validate P&L against known-good examples
   - Edge case testing (shorts, reversals, universe changes)
3. Documentation and examples

**Deliverables**:
- Benchmark integration
- Comprehensive test suite
- User guide for position-based backtesting
- Migration guide for existing strategies

**Success Criteria**:
- All tests passing
- Documentation complete
- Ready for production use

---

### Phase 6: Advanced Features (Future)

**Optional Enhancements** (post-MVP):
1. Capital constraints mode (`initial_capital` parameter)
2. FIFO cost basis (in addition to average cost)
3. Transaction costs during backtest (not post-hoc)
4. Slippage modeling (execution price adjustment)
5. Dividend tracking and cash flows
6. Margin requirements and borrowing costs
7. Position limits and risk controls
8. More execution price options (VWAP, TWAP, etc.)

---

## File Structure

```
portwine/
├── strategies/
│   ├── base.py                          # StrategyBase (existing)
│   ├── position_base.py                 # PositionStrategyBase (NEW)
│   ├── simple_moving_average.py         # Weights-based (existing)
│   └── position_examples.py             # Position-based examples (NEW)
│
├── backtester/
│   ├── core.py                          # Backtester (existing)
│   ├── position_core.py                 # PositionBacktester (NEW)
│   ├── benchmarks.py                    # Benchmark handling (existing)
│   └── __init__.py                      # Exports both backtesters
│
├── analyzers/
│   ├── base.py                          # Analyzer base (existing)
│   ├── position_adapter.py              # PositionToWeightsAdapter (NEW)
│   ├── position_equity.py               # PositionEquityAnalyzer (NEW)
│   ├── position_turnover.py             # PositionTurnoverAnalyzer (NEW)
│   ├── position_trades.py               # PositionTradeAnalyzer (NEW)
│   ├── position_exposure.py             # PositionExposureAnalyzer (NEW)
│   └── ... (25+ existing analyzers)
│
└── tests/
    ├── test_backtester.py               # Weights backtester tests (existing)
    ├── test_position_backtester.py      # Position backtester tests (NEW)
    ├── test_position_strategies.py      # Position strategy tests (NEW)
    └── test_position_analyzers.py       # Position analyzer tests (NEW)
```

---

## Testing Strategy

### Unit Tests

1. **PositionBacktester Core**
   - Date range detection
   - Universe handling
   - Action validation
   - Position updates

2. **P&L Calculation**
   - Long positions
   - Short positions
   - Position reversals (long → short)
   - Cost basis averaging
   - Realized vs unrealized

3. **Edge Cases**
   - Empty actions (no trades)
   - Universe entry/exit
   - Price gaps
   - Missing prices (NaN handling)

### Integration Tests

1. **End-to-End Backtest**
   - Simple buy-and-hold strategy
   - Active trading strategy
   - Long-short strategy

2. **Analyzer Integration**
   - Adapter converts correctly
   - Legacy analyzers work
   - Position analyzers produce valid output

### Validation Tests

1. **Hand-Calculated Examples**
   - Simple 3-day, 2-ticker scenario
   - Known P&L outcomes
   - Verify exact matches

2. **Equivalence Tests**
   - Same strategy in weights vs positions
   - Results should be proportional
   - Validate conversion logic

---

## Migration Guide for Users

### Converting Weights-Based Strategy to Position-Based

**Before (Weights-Based)**:
```python
class MyStrategy(StrategyBase):
    def step(self, current_date, daily_data):
        # Compute some signal
        signal_strength = compute_signal(daily_data)

        # Return weights (percentage allocation)
        if signal_strength > 0:
            return {'AAPL': 0.5, 'MSFT': 0.5}
        return {}
```

**After (Position-Based)**:
```python
class MyPositionStrategy(PositionStrategyBase):
    def __init__(self, tickers, shares_per_trade=100):
        super().__init__(tickers)
        self.shares_per_trade = shares_per_trade

    def step(self, current_date, daily_data):
        # Same signal computation
        signal_strength = compute_signal(daily_data)

        # Return share quantities (buy/sell actions)
        if signal_strength > 0:
            # Check if we already have positions
            if self.positions.get('AAPL', 0) == 0:
                return {'AAPL': self.shares_per_trade,
                       'MSFT': self.shares_per_trade}
        elif signal_strength < 0:
            # Exit positions
            return {'AAPL': -self.positions.get('AAPL', 0),
                   'MSFT': -self.positions.get('MSFT', 0)}
        return {}
```

**Key Differences**:
1. Extend `PositionStrategyBase` instead of `StrategyBase`
2. Return share quantities instead of weights
3. Can access current positions via `self.positions`
4. No constraint on total allocation
5. Need to decide how many shares to trade (configurable parameter)

---

## Open Questions

### 1. Benchmark Handling
**Question**: How should we handle benchmarks when strategy is in dollars but benchmarks are percentages?

**Options**:
- A) Convert benchmark to dollars (using deployed capital)
- B) Keep separate (show benchmark in %, strategy in $)
- C) Convert strategy to % (using deployed capital)

**Recommendation**: Start with **B** (simplest), gather user feedback.

---

### 2. Execution Price
**Question**: Which execution price model should be the default?

**Options**:
- A) `'next_close'` - matches current system
- B) `'next_open'` - more realistic
- C) `'close'` - unrealistic but simple

**Recommendation**: Default to **A** (`'next_close'`) for consistency, allow configuration.

---

### 3. Cost Basis Method
**Question**: Should we support FIFO in addition to average cost?

**Options**:
- A) Average cost only (simpler)
- B) Average cost + FIFO (more realistic for taxes)

**Recommendation**: Start with **A**, add FIFO in Phase 6 if needed.

---

### 4. Initial Capital
**Question**: Should we support capital constraints from day 1, or make it a future enhancement?

**Options**:
- A) Infinite capital only (MVP)
- B) Infinite capital + optional capital constraints (more complex)

**Recommendation**: **A** for MVP, **B** in Phase 6.

---

### 5. Analyzer Strategy
**Question**: Should we prioritize:
- A) Adapter for legacy analyzers (fast backward compatibility)
- B) New position-specific analyzers (better metrics)
- C) Both in parallel

**Recommendation**: **C** - create adapter first (quick wins), then build position analyzers.

---

## Success Metrics

**Phase 1-2 Success** (Core Working):
- [ ] Position backtester runs without errors
- [ ] P&L calculation matches hand-calculated examples
- [ ] Performance acceptable (<5s for 5 years, 50 tickers)
- [ ] All unit tests passing

**Phase 3-4 Success** (Usability):
- [ ] Adapter allows using legacy analyzers
- [ ] 4+ position-specific analyzers created
- [ ] 5+ example position strategies working
- [ ] Example notebook demonstrates full workflow

**Phase 5 Success** (Production Ready):
- [ ] Comprehensive test suite (unit + integration + validation)
- [ ] Documentation complete (user guide + API reference)
- [ ] Migration guide for existing users
- [ ] Benchmark handling decided and implemented
- [ ] Ready to merge to main

---

## Risks and Mitigations

### Risk 1: Performance Degradation
**Risk**: Position tracking and P&L calculation slower than weights-based.

**Mitigation**:
- Use Numba JIT for hot loops (proven effective in current system)
- Profile early and optimize bottlenecks
- Accept 2-3x slowdown as reasonable trade-off for features

---

### Risk 2: Complexity Creep
**Risk**: Position system becomes too complex with all features.

**Mitigation**:
- Implement MVP first (infinite capital, average cost, simple execution)
- Add advanced features incrementally in later phases
- Keep API simple even as internals grow

---

### Risk 3: Analyzer Incompatibility
**Risk**: Users expect all analyzers to work but they don't.

**Mitigation**:
- Create adapter for backward compatibility
- Document which analyzers work via adapter
- Provide migration guide for creating position-specific versions
- Create core position analyzers to cover common use cases

---

### Risk 4: User Confusion
**Risk**: Two backtesting systems confuse users.

**Mitigation**:
- Clear naming: `Backtester` vs `PositionBacktester`
- Documentation explains when to use which
- Examples show both approaches
- Consider making one the "primary" recommendation over time

---

### Risk 5: Benchmark Comparison Issues
**Risk**: Can't meaningfully compare position-based strategy to benchmarks.

**Mitigation**:
- Implement multiple benchmark comparison methods
- Let users choose what makes sense for their use case
- Provide examples of each approach
- Document trade-offs clearly

---

## Alternatives Considered

### Alternative 1: Extend Current Backtester (Rejected)
**Idea**: Add position mode to existing `Backtester` class.

**Pros**: Single system, no duplication.

**Cons**:
- Tight coupling of two different paradigms
- Complex conditional logic throughout
- Harder to maintain
- Breaking changes risk

**Rejection Reason**: Too much complexity, violates single responsibility principle.

---

### Alternative 2: Replace Weights with Positions (Rejected)
**Idea**: Deprecate weights-based, make position-based the only system.

**Pros**: Single system, modern approach.

**Cons**:
- Breaks all existing code
- Weights-based is still valuable for many use cases
- Forces migration on all users

**Rejection Reason**: Unnecessary disruption, weights-based has valid use cases.

---

### Alternative 3: Capital-Constrained from Start (Considered)
**Idea**: Always track cash balance and enforce capital constraints.

**Pros**: More realistic, prepares for production trading.

**Cons**:
- More complex to implement
- Harder to debug (is issue in strategy or capital logic?)
- Requires users to specify initial capital

**Decision**: Start with infinite capital (simpler), add constraints later.

---

## Conclusion

The position-based backtesting system is a **natural evolution** of the existing weights-based system, not a replacement. By running in parallel, we gain:

1. **Explicit position tracking** - know exactly how many shares are held
2. **Dollar-based P&L** - real monetary outcomes, not normalized returns
3. **New metrics** - turnover, deployed capital, trade statistics
4. **Realistic modeling** - explicit executions, cost basis, long/short

The design maintains **API compatibility** where possible while acknowledging that position-based and weights-based are fundamentally different paradigms. The adapter pattern allows immediate reuse of existing analyzers while we build position-specific ones.

**Recommendation**: Proceed with implementation in 5 phases as outlined, starting with MVP (infinite capital, average cost, simple execution) and adding advanced features incrementally.

---

## Appendix: Example Position-Based Backtest

```python
# 1. Create data interface (same as weights-based)
from portwine.data.stores import ParquetDataStore
from portwine.data.interface import DataInterface

store = ParquetDataStore('./data/stocks')
data = DataInterface(store)

# 2. Create position-based strategy
from portwine.strategies.position_base import PositionStrategyBase

class SimplePositionStrategy(PositionStrategyBase):
    def __init__(self, tickers, shares=100):
        super().__init__(tickers)
        self.shares = shares
        self.initialized = False

    def step(self, current_date, daily_data):
        # Buy initial positions on first step
        if not self.initialized:
            self.initialized = True
            return {ticker: self.shares for ticker in self.tickers}

        # Hold forever (buy and hold)
        return {}

strategy = SimplePositionStrategy(['AAPL', 'MSFT'], shares=100)

# 3. Run position backtest
from portwine.backtester import PositionBacktester

backtester = PositionBacktester(data)
results = backtester.run_backtest(
    strategy=strategy,
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# 4. Analyze results

# 4a. Use position-specific analyzer
from portwine.analyzers import PositionEquityAnalyzer

analyzer = PositionEquityAnalyzer()
metrics = analyzer.analyze(results)
print(f"Final P&L: ${metrics['final_pnl']:,.2f}")
print(f"Max Drawdown: ${metrics['max_drawdown_dollars']:,.2f}")

analyzer.plot(results)  # Show equity curve

# 4b. Use legacy analyzer via adapter
from portwine.analyzers import PositionToWeightsAdapter, SharpeRatioAnalyzer

weights_results = PositionToWeightsAdapter.convert(
    results,
    deployed_capital=100000  # Normalize to $100k
)
sharpe = SharpeRatioAnalyzer().analyze(weights_results)
print(f"Sharpe Ratio: {sharpe['sharpe']:.2f}")

# 5. Examine detailed results
print("\nPositions over time:")
print(results['positions_df'].head())

print("\nDaily P&L:")
print(results['total_pnl'].describe())

print("\nDeployed capital:")
deployed = (results['positions_df'].abs() * results['prices_df']).sum(axis=1)
print(f"Max: ${deployed.max():,.2f}")
print(f"Avg: ${deployed.mean():,.2f}")
```

**Output Example**:
```
Final P&L: $45,234.56
Max Drawdown: $-12,345.67
Sharpe Ratio: 1.23

Positions over time:
            AAPL  MSFT
2020-01-02   100   100
2020-01-03   100   100
2020-01-06   100   100
...

Daily P&L:
count    1008.000000
mean       44.882937
std       567.234512
min     -2345.678901
max      3456.789012

Deployed capital:
Max: $87,654.32
Avg: $65,432.10
```
