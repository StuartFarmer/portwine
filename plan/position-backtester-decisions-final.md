# Position-Based Backtester - Final Decisions

**Date**: 2025-10-23
**Status**: Decisions finalized, ready for implementation

---

## Design Decisions (Finalized)

### Decision 1: Strategy Position Access
**Choice**: Strategies can access positions via **alternative data loader** (existing pattern)

**Implication**:
- **NO NEW `PositionStrategyBase` CLASS NEEDED**
- Use existing `StrategyBase`
- Strategies output same `Dict[str, float]` format
- Backtester interprets values as **shares** instead of **weights**
- Strategies wanting position info use alternative data (e.g., `daily_data['POSITIONS:AAPL']`)

**Example**:
```python
class MyStrategy(StrategyBase):  # Use existing base class!
    def step(self, current_date, daily_data):
        # Access position via alternative data if needed
        # current_pos = daily_data.get('POSITIONS:AAPL', {}).get('shares', 0)

        # Return share quantities (not weights!)
        return {'AAPL': 10, 'MSFT': -5}  # Buy 10 AAPL, sell 5 MSFT
```

---

### Decision 2: Execution Price Model
**Choice**: Execute on **close** (same as weights backtester)

**Note**: Check if original backtester has open/close option. If it does, replicate that.

**Action Item**: Verify if `Backtester` supports execution price configuration
- [ ] Check current `Backtester` for execution price parameter
- [ ] If exists, replicate same parameter in `PositionBacktester`
- [ ] If not, default to close only for MVP

---

### Decision 3: Cost Basis Method
**Choice**: **Defer to external analyzer** - not calculated in core backtester

**Implication**:
- Core backtester does NOT track cost basis
- Core backtester does NOT calculate realized vs unrealized P&L
- Backtester only tracks: positions, actions, prices, total value
- **Analyzer** can calculate cost basis, realized P&L post-hoc using any method (average, FIFO, LIFO, etc.)

**Simplification**: This significantly reduces backtester complexity!

---

### Decision 4: Short Selling
**Choice**: **Yes** - allow negative positions

**Implication**: No validation preventing negative positions

---

### Decision 5: Fractional Shares
**Choice**: **Yes** - allow fractional shares

**Implication**: Use `float` for position quantities, not `int`

---

### Decision 6: Benchmark Comparison
**Choice**: Follow existing backtester pattern (extensible functions)

**Default Approach**:
- Benchmark holds position size of 1 per ticker, OR
- Benchmark holds average notional position of strategy over time

**Implication**: Same benchmark architecture as weights backtester
- Support benchmark functions
- Allow custom callables
- Defer sophisticated comparison to analyzers

---

### Decision 7: Initial Positions
**Choice**: **No** - always start with empty portfolio

**Implication**: All backtests start with zero positions

---

### Decision 8: Universe Exit Handling
**Choice**: **Follow weights backtester behavior**

**Action Item**: Verify what weights backtester does
- [ ] Check `Backtester.run_backtest()` for universe exit handling
- [ ] Replicate same logic in `PositionBacktester`

**Hypothesis**: Likely force-sells positions when ticker exits universe

---

### Decision 9: File Structure
**Clarification**: No new strategy files needed (using existing `StrategyBase`)

**New Files Required**:
```
portwine/backtester/position_core.py    # PositionBacktester only
tests/test_position_backtester.py       # Tests
```

**No new files in**:
- `strategies/` - Use existing `StrategyBase`
- `analyzers/` - Defer to later phase

---

### Decision 10: Data Requirements
**Choice**: Use **exact same data interface** as weights backtester

**Implication**:
- Same `DataInterface`, `MultiDataInterface`, `RestrictedDataInterface`
- Same price access patterns
- If weights backtester uses close only, position backtester uses close only
- If weights backtester supports OHLCV, position backtester gets it automatically

---

### Decision 11: Testing Approach
**Choice**: **Hybrid** (write feature, test it, build next feature on top)

---

### Decision 12: MVP Feature Set
**Choice**: Start with **minimal working backtester**, iterate rapidly

**MVP Definition**:
- `PositionBacktester` class with `run_backtest()` method
- Interprets strategy output as share quantities
- Tracks positions over time
- Tracks actions over time
- Tracks prices over time
- Calculates portfolio value (sum of position × price)
- Returns results dict with DataFrames
- NO cost basis tracking (defer to analyzer)
- NO realized/unrealized split (defer to analyzer)
- NO benchmarks initially (add if easy)

---

### Decision 13: Numba Optimization Timing
**Choice**: **Python first**, optimize later

**Implication**: Get correctness first, add Numba JIT after tests pass

---

## Revised Architecture

### Core Insight: Much Simpler Than Original Design!

**Original Plan** (too complex):
- New `PositionStrategyBase` class
- Complex cost basis tracking in backtester
- Realized vs unrealized P&L split
- Position injection into strategy

**Revised Plan** (your approach):
- Use existing `StrategyBase` ✓
- No cost basis in backtester (defer to analyzer) ✓
- Simple portfolio value calculation ✓
- Position access via alternative data (existing pattern) ✓

---

## Simplified Position Backtester Architecture

### Input
```python
strategy = StrategyBase(['AAPL', 'MSFT'])  # Existing class!

# Strategy returns shares (backtester interprets as quantities)
def step(self, current_date, daily_data):
    return {'AAPL': 10}  # Buy 10 shares
```

### Processing
```python
# PositionBacktester.run_backtest()
for date in datetime_index:
    actions = strategy.step(date, data)  # {'AAPL': 10}

    # Update positions
    for ticker, action in actions.items():
        positions[ticker] += action

    # Record state
    record_positions(date, positions)
    record_actions(date, actions)
    record_prices(date, current_prices)

    # Calculate portfolio value
    portfolio_value = sum(positions[t] * prices[t] for t in tickers)
```

### Output
```python
results = {
    'positions_df': DataFrame,   # (days × tickers) share quantities
    'actions_df': DataFrame,     # (days × tickers) buy/sell quantities
    'prices_df': DataFrame,      # (days × tickers) execution prices
    'portfolio_value': Series,   # (days,) total portfolio value
    'benchmark_returns': Series  # (days,) if benchmark provided
}
```

### Post-Processing (Analyzers)
```python
# Analyzer calculates cost basis, realized P&L, etc.
# Using whatever method desired (average, FIFO, LIFO)

analyzer = PositionCostBasisAnalyzer(method='average')
pnl = analyzer.analyze(results)
# Returns: realized_pnl, unrealized_pnl, cost_basis_history, etc.
```

---

## Simplified Output API

### Minimal Output (MVP)
```python
{
    'positions_df': pd.DataFrame,      # Share positions over time
    'actions_df': pd.DataFrame,        # Buy/sell actions over time
    'prices_df': pd.DataFrame,         # Execution prices
    'portfolio_value': pd.Series,      # Total value (Σ position × price)
}
```

### With Benchmark (MVP+)
```python
{
    # ... everything above, plus:
    'benchmark_returns': pd.Series,    # Or benchmark_portfolio_value?
}
```

**Note**: No realized/unrealized split, no cost basis - that's analyzer territory!

---

## Key Questions to Verify Before Implementation

### Question 1: Weights Backtester Execution Price
**Need to check**: Does current `Backtester` have execution price configuration?

**File to check**: `portwine/backtester/core.py` - look for parameters in `run_backtest()`

**Expected**: Likely just uses close, maybe has option for next open

---

### Question 2: Weights Backtester Universe Exit
**Need to check**: What happens when ticker exits universe with open position?

**File to check**: `portwine/backtester/core.py` - look in main loop for universe changes

**Expected**: Likely force-liquidates or raises warning

---

### Question 3: Strategy Output Format
**Need to verify**: Current strategies return `Dict[str, float]` for weights

**Confirmation**: Position strategies will return same format, just interpreted as shares

**Validation**: Should we add any new validation besides "ticker in universe"?

---

## Implementation Simplifications

### Removed from Original Plan:
1. ❌ `PositionStrategyBase` class (use existing `StrategyBase`)
2. ❌ Cost basis tracking in backtester (defer to analyzer)
3. ❌ Realized vs unrealized P&L calculation (defer to analyzer)
4. ❌ Position injection into strategy (use alternative data)
5. ❌ Complex benchmark conversion (use simple approach)
6. ❌ Initial positions support (not needed)
7. ❌ New data interfaces (use existing)

### Kept in Plan:
1. ✅ `PositionBacktester` class (new file)
2. ✅ Position tracking (core feature)
3. ✅ Action tracking (core feature)
4. ✅ Price tracking (core feature)
5. ✅ Portfolio value calculation (simple sum)
6. ✅ Results dict with DataFrames (match existing API)
7. ✅ Benchmark support (same extensible pattern)

---

## Estimated Complexity Reduction

**Original Estimate**: 5-6 weeks, 2000+ lines of code

**Revised Estimate**: 2-3 weeks, 500-800 lines of code

**Reason**:
- No new strategy base class (~200 lines saved)
- No cost basis tracking (~300 lines saved)
- No realized/unrealized split (~200 lines saved)
- Reuse existing data interfaces (~400 lines saved)
- Simpler validation (~100 lines saved)

---

## Next Steps

1. ✅ Decisions finalized
2. ⏳ Verify weights backtester behavior (execution price, universe exit)
3. ⏳ Create iteration-by-iteration implementation plan
4. ⏳ Begin implementation

---

## Implementation Philosophy

**Your approach** (lean, iterative):
- Start with simplest possible working version
- Test immediately
- Add one feature at a time
- Build on tested foundation
- Defer complexity to analyzers

**This aligns perfectly with**:
- Unix philosophy (do one thing well)
- Separation of concerns (backtester ≠ analyzer)
- Existing portwine architecture (analyzers consume backtest results)

---

## Summary

The position backtester is **much simpler** than originally designed:

1. **No new strategy class** - use existing `StrategyBase`
2. **No cost basis** - defer to analyzers
3. **Simple position tracking** - just accumulate actions
4. **Simple portfolio value** - just sum position × price
5. **Same data interfaces** - reuse everything
6. **Same patterns** - execution, benchmarks, validation

This is a **weekend project**, not a month-long effort!
