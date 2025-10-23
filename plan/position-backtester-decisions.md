# Position-Based Backtester - Decision Document

**Purpose**: Clarify design choices before implementation
**Status**: Awaiting user decisions

---

## Critical Decisions (Must Answer Before Starting)

### Decision 1: Strategy Position Access

**Question**: Should strategies have access to their current positions?

**Context**: Strategies might want to make decisions based on existing positions (e.g., "only buy if I don't already own it", "add to position if price drops").

**Options**:

**A) Yes - Strategy can read positions**
```python
class MyStrategy(PositionStrategyBase):
    def step(self, current_date, daily_data):
        # Can check current positions
        if self.positions.get('AAPL', 0) < 100:
            return {'AAPL': 50}  # Add to position
        return {}
```

**Pros**: More flexible, enables position-aware strategies
**Cons**: Slightly more complex, strategy needs internal tracking

**B) No - Strategy is stateless**
```python
class MyStrategy(PositionStrategyBase):
    def step(self, current_date, daily_data):
        # Can't see current positions, just return desired changes
        return {'AAPL': 50}  # Backtester tracks everything
```

**Pros**: Simpler, cleaner separation
**Cons**: Can't make position-aware decisions

**Recommendation**: **Option A** (strategies can read positions)

**Your Choice**: _____________

---

### Decision 2: Execution Price Model

**Question**: When executing trades, which price do we use?

**Context**: Strategy decides at end of day, but can't execute at that moment in real life.

**Options**:

**A) Next day's close** (matches current weights-based system)
- Strategy sees Day 1 close → executes at Day 2 close
- Consistent with existing system
- Unrealistic (can't see price before executing)

**B) Next day's open** (more realistic)
- Strategy sees Day 1 close → executes at Day 2 open
- Requires open price data (may not be available for all data sources)
- More realistic execution model

**C) Same day's close** (simplest but unrealistic)
- Strategy sees Day 1 close → executes at Day 1 close
- Lookahead bias (can't execute at price you just observed)
- Only use for quick testing

**D) Configurable with default** (most flexible)
- Allow user to choose execution model
- Default to one of the above

**Recommendation**: **Option D** with default **A** (next close, for consistency)

**Your Choice**: _____________

---

### Decision 3: Cost Basis Method (MVP)

**Question**: How do we calculate cost basis when accumulating positions?

**Context**: Needed for P&L calculation when buying/selling same stock multiple times.

**Options**:

**A) Average Cost** (simpler)
```
Day 1: Buy 10 @ $100 → Avg cost = $100
Day 2: Buy 10 @ $110 → Avg cost = (10×$100 + 10×$110) / 20 = $105
Day 3: Sell 5 @ $120 → Realized P&L = 5 × ($120 - $105) = $75
                      → Remaining 15 @ $105 avg cost
```

**Pros**: Simple calculation, no position tracking needed
**Cons**: Not tax-accurate (IRS requires FIFO)

**B) FIFO (First In, First Out)** (tax-accurate)
```
Day 1: Buy 10 @ $100 → Queue: [(10, $100)]
Day 2: Buy 10 @ $110 → Queue: [(10, $100), (10, $110)]
Day 3: Sell 5 @ $120 → Sell from first lot
                      → Realized P&L = 5 × ($120 - $100) = $100
                      → Queue: [(5, $100), (10, $110)]
```

**Pros**: Tax-accurate, realistic
**Cons**: More complex (need queue per ticker)

**C) Both (most flexible but more work)**
- Implement average cost first (MVP)
- Add FIFO as option later

**Recommendation**: **Option A** for MVP (average cost), add FIFO in Phase 6

**Your Choice**: _____________

---

### Decision 4: Short Selling

**Question**: Allow negative positions (short selling)?

**Context**: Some strategies need to short. Adds complexity to P&L calculation.

**Options**:

**A) Yes - Allow shorts from MVP**
```python
position = -100  # Short 100 shares
# P&L = 100 × (entry_price - current_price)
```

**Pros**: Full flexibility, complete feature set
**Cons**: More complex P&L logic, need to test short scenarios

**B) No - Only long positions in MVP**
```python
# Reject negative position changes in validate_actions()
```

**Pros**: Simpler MVP, easier testing
**Cons**: Limits strategy types, need to add later anyway

**Recommendation**: **Option A** (allow shorts from start) - not much extra complexity

**Your Choice**: _____________

---

### Decision 5: Fractional Shares

**Question**: Allow fractional shares (e.g., 10.5 shares)?

**Context**: Modern brokers support fractional shares. Makes some strategies easier.

**Options**:

**A) Yes - Allow fractional shares**
```python
return {'AAPL': 10.5}  # Buy 10.5 shares
```

**Pros**: More flexible, easier to allocate exact dollar amounts
**Cons**: Slightly more complex validation (use float, not int)

**B) No - Only whole shares**
```python
return {'AAPL': 10}  # Must be integer
```

**Pros**: Simpler, traditional
**Cons**: Less flexible, harder to do dollar-based allocation

**Recommendation**: **Option A** (allow fractional) - no significant complexity

**Your Choice**: _____________

---

### Decision 6: Benchmark Comparison

**Question**: How do we compare position-based strategy (in dollars) to benchmarks (in percentages)?

**Context**: Current benchmarks return percentage returns. Position strategies return dollar P&L.

**Options**:

**A) Keep separate** (simplest)
- Strategy results in dollars
- Benchmark in percentages
- Show both, let user interpret
- Can calculate conversion manually if needed

**B) Convert benchmark to dollars** (more work)
- Calculate "virtual deployed capital" for benchmark
- Apply benchmark % returns to that capital
- Both in dollars for direct comparison

**C) Convert strategy to percentages** (requires assumption)
- Divide strategy dollar P&L by "deployed capital"
- Both in percentages
- Requires choosing what "deployed capital" means

**D) Defer decision** (punt for now)
- No benchmark support in MVP
- Add in later phase once we see what makes sense

**Recommendation**: **Option A** or **D** for MVP

**Your Choice**: _____________

---

### Decision 7: Initial Positions

**Question**: Can backtest start with existing positions?

**Context**: Sometimes want to test "what if I held these positions" scenarios.

**Options**:

**A) Yes - Allow optional initial positions**
```python
backtester.run_backtest(
    strategy,
    initial_positions={'AAPL': 100, 'MSFT': 50}
)
```

**Pros**: More flexible, enables interesting analyses
**Cons**: Slightly more complex initialization

**B) No - Always start with empty portfolio**
```python
# All strategies start with no positions
```

**Pros**: Simpler, one less thing to test
**Cons**: Limits use cases

**Recommendation**: **Option B** for MVP (always start empty), add A in Phase 6

**Your Choice**: _____________

---

### Decision 8: Universe Exit Handling

**Question**: What happens when a ticker exits the universe while we hold a position?

**Context**: Dynamic universes (e.g., S&P 500 constituents) change over time. Ticker might be removed while we hold shares.

**Options**:

**A) Force liquidation** (conservative)
```python
# When AAPL exits universe:
# - Sell entire position at last available price
# - Realize P&L
# - Log warning
```

**Pros**: Clean, no orphaned positions
**Cons**: Forced sale might not be strategy intent

**B) Hold position but freeze** (flexible)
```python
# When AAPL exits universe:
# - Keep position (don't force sale)
# - Mark price as last known price
# - Prevent new trades
# - Allow strategy to see and manage it
```

**Pros**: Strategy controls exit
**Cons**: More complex, need to track "frozen" positions

**C) Error/Raise exception** (strict)
```python
# When ticker exits with open position:
# - Raise error
# - Force strategy to handle before removal
```

**Pros**: Explicit, no surprises
**Cons**: Annoying, breaks backtests

**Recommendation**: **Option A** for MVP (force liquidation with warning)

**Your Choice**: _____________

---

### Decision 9: File Structure

**Question**: Where do new files go?

**Context**: Need to integrate with existing structure without breaking things.

**Options**:

**A) Parallel structure** (separate but equal)
```
portwine/
├── strategies/
│   ├── base.py                    # StrategyBase (existing)
│   └── position_base.py           # PositionStrategyBase (NEW)
├── backtester/
│   ├── core.py                    # Backtester (existing)
│   └── position_core.py           # PositionBacktester (NEW)
└── analyzers/
    ├── (existing analyzers)
    └── position/                  # New subdirectory
        ├── adapter.py             # PositionToWeightsAdapter
        ├── equity.py              # Position-specific analyzers
        └── ...
```

**Pros**: Clear separation, no conflicts
**Cons**: More files

**B) Extend existing files** (integrated)
```
portwine/
├── strategies/
│   └── base.py                    # Add PositionStrategyBase here
├── backtester/
│   └── core.py                    # Add PositionBacktester here
└── analyzers/
    └── (add position analyzers alongside existing)
```

**Pros**: Fewer files, side-by-side comparison
**Cons**: Large files, potential for conflicts

**Recommendation**: **Option A** (parallel structure) - cleaner separation

**Your Choice**: _____________

---

### Decision 10: Data Requirements

**Question**: What price data do we need for MVP?

**Context**: Current system only needs close prices. Position system might need more.

**Options**:

**A) Close only** (minimum, matches existing)
- Only require close prices
- All execution at close
- Simplest, works with existing data

**B) Open + Close** (better)
- Require both open and close
- More realistic execution models
- Might not be available for all data sources

**C) OHLCV** (complete)
- Full OHLCV data
- Enables slippage modeling, etc.
- Overkill for MVP

**Recommendation**: **Option A** for MVP (close only), check for open/high/low but don't require

**Your Choice**: _____________

---

## Implementation Strategy Decisions

### Decision 11: Testing Approach

**Question**: What testing strategy should we use?

**Options**:

**A) Test-Driven Development (TDD)** (strict)
- Write test first
- Implement feature to pass test
- Refactor
- Repeat

**B) Hybrid Approach** (flexible)
- Write simplest feature first
- Write tests for that feature
- Add next feature on top
- Test again
- Allows exploration while maintaining test coverage

**C) Implementation-first** (cowboy)
- Build everything
- Test at end
- Faster but risky

**Recommendation**: **Option B** (hybrid) - as you suggested

**Your Choice**: _____________

---

### Decision 12: MVP Feature Set

**Question**: What's the absolute minimum for a "working" position backtester?

**Context**: Want to get something testable ASAP, then iterate.

**Proposed MVP (Minimal)**:
- [ ] `PositionStrategyBase` with single abstract method `step()`
- [ ] `PositionBacktester.run_backtest()` - basic loop
- [ ] Position tracking (positions array)
- [ ] Action tracking (actions array)
- [ ] Price tracking (execution prices)
- [ ] Simple P&L calculation (unrealized only, no cost basis yet)
- [ ] Return results dict with DataFrames
- [ ] One simple test strategy (buy-and-hold)
- [ ] NO benchmarks, NO analyzers, NO fancy features

**Proposed MVP+ (Usable)**:
- Everything in MVP, plus:
- [ ] Cost basis tracking (average cost)
- [ ] Realized P&L (when closing positions)
- [ ] Proper P&L calculation (unrealized + realized)
- [ ] Signal validation (tickers in universe)
- [ ] One position analyzer (equity curve plotter)

**Question**: Start with MVP or MVP+?

**Recommendation**: **MVP** first (get green lights), then immediate upgrade to MVP+

**Your Choice**: _____________

---

### Decision 13: Numba Optimization Timing

**Question**: When do we add Numba JIT optimization?

**Context**: Current system uses Numba for 10x speedup, but adds complexity.

**Options**:

**A) From the start** (fast from day 1)
- Write P&L calculation with Numba from beginning
- Faster, matches existing system
- Harder to debug if wrong

**B) Pure Python first, optimize later** (safer)
- Write P&L in pure Python/NumPy
- Get it working and tested
- Add Numba JIT in later iteration
- Easier to debug

**C) Parallel implementation** (both)
- Write both versions
- Test they match
- Use Numba in production

**Recommendation**: **Option B** (Python first) - correctness before performance

**Your Choice**: _____________

---

## Summary of Recommendations

Here are my recommendations for the lean, iterative approach:

| Decision | Recommendation | Rationale |
|----------|----------------|-----------|
| 1. Position access | **Yes** | Enables position-aware strategies |
| 2. Execution price | **Configurable, default next close** | Flexible, consistent with existing |
| 3. Cost basis | **Average cost** | Simpler for MVP |
| 4. Short selling | **Yes** | Not much extra complexity |
| 5. Fractional shares | **Yes** | More flexible, minimal complexity |
| 6. Benchmark | **Keep separate (A) or defer (D)** | Simplest for MVP |
| 7. Initial positions | **No** | Add later if needed |
| 8. Universe exit | **Force liquidation** | Clean, simple |
| 9. File structure | **Parallel** | Clear separation |
| 10. Data requirements | **Close only** | Works with existing data |
| 11. Testing approach | **Hybrid** | As you requested |
| 12. MVP scope | **MVP first** | Get green lights ASAP |
| 13. Numba timing | **Python first** | Correctness before speed |

---

## Next Steps

Once you've made these decisions, I will create:

**`position-backtester-implementation-plan.md`**

This will contain:
1. **Iteration 0**: Project setup (file creation, imports)
2. **Iteration 1**: Minimal position tracking (simplest testable feature)
3. **Iteration 2**: Strategy interface (build on Iteration 1)
4. **Iteration 3**: Backtester loop (build on Iteration 2)
5. **Iteration 4**: P&L calculation - unrealized only (build on Iteration 3)
6. **Iteration 5**: P&L calculation - realized (build on Iteration 4)
7. **Iteration 6**: Cost basis tracking (build on Iteration 5)
8. **Iteration 7**: Results formatting (build on Iteration 6)
9. **Iteration 8**: Validation and edge cases (build on Iteration 7)
10. **Iteration 9**: First analyzer (build on Iteration 8)
11. **Iteration 10+**: Additional features based on testing

Each iteration will:
- Build on previous tested functionality
- Have clear acceptance criteria
- Include test cases
- Be independently verifiable
- Add one new capability

---

## Your Decisions

Please fill in your choices for each decision (1-13), and I'll create the detailed implementation plan.

**Format**: Just reply with the decision numbers and your choice (A, B, C, or D):

```
1: A
2: D
3: A
...
```

Or feel free to explain any modifications to the options!
