# Position-Based Backtester - Project Summary

**Branch**: `position-backtester`
**Status**: Ready to implement
**Estimated Time**: 2-3 weeks (part-time)

---

## What We're Building

A **position-based backtesting system** that runs parallel to the existing weights-based backtester.

**Key Difference**:
- **Weights backtester**: Strategy returns `{'AAPL': 0.25, 'MSFT': 0.75}` (portfolio allocation %)
- **Position backtester**: Strategy returns `{'AAPL': 10, 'MSFT': -5}` (buy 10 AAPL, sell 5 MSFT)

---

## Major Design Decisions (Finalized)

### 1. No New Strategy Base Class ✅
**Use existing `StrategyBase`** - just interpret output as shares instead of weights

**Impact**: Saves ~500 lines of code, simpler implementation

### 2. No Cost Basis in Backtester ✅
**Defer to analyzers** - backtester only tracks positions, actions, prices

**Impact**: Backtester stays simple, analyzers handle complex P&L calculations

### 3. Same Data Interfaces ✅
**Reuse existing** `DataInterface`, `MultiDataInterface`, `RestrictedDataInterface`

**Impact**: No new data code needed, full compatibility

### 4. Same Patterns ✅
**Follow existing backtester** for execution price, universe handling, benchmarks

**Impact**: Consistency, predictable behavior

---

## Core Architecture

```
Input:
  Strategy returns: {'AAPL': 10}  # Buy 10 shares

Processing:
  For each trading day:
    1. Call strategy.step()
    2. Validate actions (ticker in universe)
    3. Record actions
    4. Update positions (cumulative sum of actions)
    5. Record prices (close price)

  After loop:
    6. Calculate portfolio value (sum of position × price)

Output:
  {
    'positions_df': DataFrame,     # (days × tickers) share positions
    'actions_df': DataFrame,       # (days × tickers) buy/sell actions
    'prices_df': DataFrame,        # (days × tickers) execution prices
    'portfolio_value': Series      # (days,) total portfolio value
  }
```

---

## Implementation Approach

**Philosophy**: Lean, iterative, test-driven

### Iteration Strategy
Each iteration:
1. Build simplest version of one feature
2. Test it immediately
3. Build next feature on top
4. **Never move forward with failing tests**

### 13 Iterations

**Phase 1: Foundation** (Iterations 0-3)
- File setup
- `PositionBacktestResult` data structure
- Array storage and manipulation
- Output formatting

**Phase 2: Backtester Shell** (Iterations 4-5)
- `PositionBacktester` class
- Basic loop structure
- Date range handling

**Phase 3: Core Logic** (Iterations 6-8)
- Test data fixtures
- Process actions
- Track positions (cumulative sum)
- Track prices
- Calculate portfolio value

**Phase 4: Robustness** (Iteration 9)
- Input validation
- Edge cases (fractional shares, shorts, empty strategies)
- Error handling

**Phase 5: Integration** (Iterations 10-12)
- Module exports
- Example scripts
- Documentation
- User guide

**Phase 6: Performance** (Iteration 13)
- Performance testing
- Optimization if needed (Numba)

---

## Files to Create

### New Files
```
portwine/backtester/position_core.py          # ~400 lines
tests/test_position_backtester.py             # ~600 lines
examples/position_backtest_example.py         # ~80 lines
docs/user-guide/position-backtesting.md       # Documentation
```

### Modified Files
```
portwine/backtester/__init__.py               # Add PositionBacktester export
```

**Total new code**: ~1000 lines (vs. 2000+ in original plan)

---

## Key Features

### Supported
- ✅ Share quantities (int or float)
- ✅ Long positions
- ✅ Short positions (negative quantities)
- ✅ Fractional shares
- ✅ Position accumulation (buy multiple times)
- ✅ Position reduction (sell partially)
- ✅ Portfolio value tracking
- ✅ Same universe handling as weights backtester
- ✅ Same data interfaces
- ✅ Same execution model (close prices)

### Not Supported (MVP)
- ❌ Cost basis tracking → defer to analyzer
- ❌ Realized/unrealized P&L split → defer to analyzer
- ❌ Benchmarks → add in Iteration 10 if easy, else defer
- ❌ Initial positions → add later if needed
- ❌ Capital constraints → add later if needed
- ❌ Transaction costs → defer to analyzer
- ❌ Slippage modeling → defer to analyzer

---

## Output Format

```python
results = {
    # Core data
    'positions_df': pd.DataFrame(
        index=datetime_index,    # Trading days
        columns=tickers,         # Ticker symbols
        values=positions         # Share quantities
    ),

    'actions_df': pd.DataFrame(
        index=datetime_index,
        columns=tickers,
        values=actions           # Buy/sell quantities
    ),

    'prices_df': pd.DataFrame(
        index=datetime_index,
        columns=tickers,
        values=prices            # Execution prices
    ),

    'portfolio_value': pd.Series(
        index=datetime_index,
        values=sum(positions × prices)  # Total value
    ),

    # Optional
    'benchmark_returns': pd.Series(...)  # If benchmark provided
}
```

---

## Example Usage

```python
from portwine.backtester import PositionBacktester
from portwine.strategies.base import StrategyBase
from portwine.data.stores.csvstore import CSVStore
from portwine.data.interface import DataInterface


# 1. Create strategy (returns share quantities!)
class MyStrategy(StrategyBase):
    def __init__(self, tickers):
        super().__init__(tickers)
        self.bought = False

    def step(self, current_date, daily_data):
        if not self.bought:
            self.bought = True
            return {'AAPL': 100, 'MSFT': 50}  # Buy 100 AAPL, 50 MSFT
        return {}  # Hold


# 2. Setup data (same as weights backtester)
store = CSVStore("./data")
data = DataInterface(store)


# 3. Run backtest
strategy = MyStrategy(['AAPL', 'MSFT'])
backtester = PositionBacktester(data)

results = backtester.run_backtest(
    strategy,
    start_date='2020-01-01',
    end_date='2023-12-31',
    verbose=True
)


# 4. Analyze results
print("Final positions:", results['positions_df'].iloc[-1])
print("Final portfolio value:", results['portfolio_value'].iloc[-1])

# Plot
results['portfolio_value'].plot(title='Portfolio Value Over Time')
```

---

## Testing Strategy

### Test Pyramid

**Unit Tests** (Iterations 1-3):
- `PositionBacktestResult` initialization
- Array operations
- Data methods
- Output formatting

**Integration Tests** (Iterations 6-8):
- End-to-end backtest
- Position tracking
- Portfolio value calculation
- Real data processing

**Edge Case Tests** (Iteration 9):
- Invalid inputs
- Empty strategies
- Fractional shares
- Short positions
- Series/None returns

**Performance Tests** (Iteration 13):
- Large datasets
- Many tickers
- Long time periods

### Test Coverage Goal
- 90%+ coverage on core logic
- 100% coverage on public API
- All edge cases covered

---

## Timeline

### Week 1: Core Implementation
**Mon-Tue**: Iterations 0-3 (Foundation)
- Setup files
- Build `PositionBacktestResult`
- Test data structures

**Wed-Thu**: Iterations 4-5 (Shell)
- Build `PositionBacktester`
- Implement loop structure
- Test initialization

**Fri-Sun**: Iterations 6-8 (Logic)
- Process actions
- Track positions
- Calculate portfolio value
- Integration tests

**Checkpoint**: MVP works end-to-end

---

### Week 2: Polish and Integration
**Mon-Tue**: Iteration 9 (Robustness)
- Edge cases
- Validation
- Error handling

**Wed**: Iterations 10-11 (Integration)
- Module exports
- Example scripts
- Basic documentation

**Thu**: Iteration 12 (Documentation)
- Complete docstrings
- User guide
- Type hints

**Fri**: Iteration 13 (Performance)
- Performance testing
- Optimization if needed
- Final validation

**Checkpoint**: Production ready

---

### Week 3 (Optional): Analyzers
**If needed**:
- Cost basis analyzer
- Trade analysis
- Position metrics
- Adapter for legacy analyzers

---

## Success Metrics

### MVP Complete
- [ ] All 13 iterations complete
- [ ] All tests pass (90%+ coverage)
- [ ] Example script works
- [ ] Documentation exists
- [ ] Performance acceptable (<5s for 5 years, 50 tickers)
- [ ] No known bugs

### Production Ready
- [ ] User guide complete
- [ ] Integrated with main codebase
- [ ] At least 1 example strategy
- [ ] Reviewed and approved

---

## Risk Mitigation

### Risk: Complexity creep
**Mitigation**: Strict scope adherence, defer complex features to analyzers

### Risk: Performance issues
**Mitigation**: Performance testing in Iteration 13, Numba ready if needed

### Risk: Integration problems
**Mitigation**: Follow existing patterns, use same interfaces

### Risk: Incomplete testing
**Mitigation**: Test at every iteration, no forward progress without green tests

---

## Future Enhancements (Post-MVP)

### Phase 7: Analyzers
- Cost basis analyzer (average cost, FIFO)
- Trade analysis (win rate, holding period)
- Position metrics (turnover, concentration)
- Adapter (position → weights format)

### Phase 8: Advanced Features
- Capital constraints mode
- Transaction costs during backtest
- Slippage modeling
- Margin requirements
- Dividend tracking

### Phase 9: Alternative Execution
- Open price execution
- VWAP/TWAP
- Custom execution models

---

## Key Insights

### What Makes This Simple
1. **No new strategy class** - reuse existing
2. **No cost basis** - defer to analyzers
3. **Same data APIs** - no new infrastructure
4. **Follow existing patterns** - predictable behavior

### What Makes This Powerful
1. **Explicit positions** - know exact shares held
2. **Dollar values** - real monetary outcomes
3. **New metrics** - trades, turnover, concentration
4. **Realistic modeling** - actual share quantities

### What Makes This Maintainable
1. **Parallel to existing** - no breaking changes
2. **Clear separation** - backtester ≠ analyzer
3. **Well tested** - test at every step
4. **Well documented** - guide and examples

---

## Questions? Issues?

If stuck during implementation:

1. **Check the design doc**: [position-backtester-design.md](position-backtester-design.md)
2. **Check implementation plan**: [position-backtester-implementation.md](position-backtester-implementation.md)
3. **Look at existing backtester**: `portwine/backtester/core.py`
4. **Review decisions**: [position-backtester-decisions-final.md](position-backtester-decisions-final.md)

---

## Getting Started

Ready to implement? Start here:

```bash
# 1. Ensure you're on the right branch
git checkout position-backtester

# 2. Create the files
touch portwine/backtester/position_core.py
touch tests/test_position_backtester.py

# 3. Begin Iteration 0
# Follow position-backtester-implementation.md step-by-step

# 4. Run tests after each iteration
pytest tests/test_position_backtester.py -v

# 5. Commit when tests pass
git add .
git commit -m "Complete Iteration N: [description]"
```

**First commit**: Iteration 0 (project setup)
**Last commit**: Iteration 13 (performance validated)

---

## Final Notes

This is a **weekend project**, not a month-long effort, thanks to:
- Reusing existing infrastructure
- Deferring complexity to analyzers
- Following existing patterns
- Lean, iterative approach

**Estimated actual coding time**: 15-20 hours
**Estimated testing time**: 10-15 hours
**Estimated documentation time**: 3-5 hours

**Total**: ~30-40 hours over 2-3 weeks part-time

Good luck! 🚀
