# Critical Blockers - Data Refactor

**Status**: Must be resolved before merging `loader` branch to `main`

---

## 1. Missing `earliest()` Method in DataStore Classes

**Priority**: 🚨 HIGH - Runtime crash

### Problem
- `DataInterface` calls `store.earliest(symbol)` at multiple locations:
  - [portwine/data/interface.py:63](../portwine/data/interface.py#L63)
  - [portwine/data/interface.py:211](../portwine/data/interface.py#L211)
- None of the DataStore implementations have this method:
  - [ParquetDataStore](../portwine/data/stores/parquet.py)
  - [CSVDataStore](../portwine/data/stores/csvstore.py)
  - [Base DataStore](../portwine/data/stores/base.py)

### Impact
- **Crash scenario**: When backtester calls `earliest_any_date()` or `earliest_common_date()`
- **AttributeError**: `'ParquetDataStore' object has no attribute 'earliest'`
- Prevents backtester from determining valid start dates for strategies

### Solution
Implement `earliest()` method in all DataStore classes:

```python
def earliest(self, identifier: str) -> Union[datetime, None]:
    """
    Get the earliest date for a given identifier.

    Parameters
    ----------
    identifier : str
        The symbol/ticker to query

    Returns
    -------
    datetime or None
        Earliest available date, or None if identifier not found
    """
    df = self._load_dataframe(identifier)
    if df is None or df.empty:
        return None
    return df.index.min()
```

### Files to Modify
1. [portwine/data/stores/base.py](../portwine/data/stores/base.py) - Add abstract method
2. [portwine/data/stores/csvstore.py](../portwine/data/stores/csvstore.py) - Implement
3. [portwine/data/stores/parquet.py](../portwine/data/stores/parquet.py) - Implement
4. [portwine/data/stores/noisy.py](../portwine/data/stores/noisy.py) - Delegate to wrapped store

### Test Coverage Needed
- Test with existing data
- Test with missing identifier
- Test with empty dataset
- Test with single data point

---

## 2. Missing Dependency: cvxpy

**Priority**: 🚨 HIGH - Blocks all tests

### Problem
- Test suite fails to import due to missing `cvxpy` module
- Error occurs in [portwine/backtester/benchmarks.py:1](../portwine/backtester/benchmarks.py#L1)
- Blocks execution of ALL tests

### Error Output
```
portwine/backtester/benchmarks.py:1: in <module>
    import cvxpy as cp
E   ModuleNotFoundError: No module named 'cvxpy'
```

### Impact
- Cannot run test suite at all
- Cannot verify refactor is working
- Blocks CI/CD pipeline

### Solution
1. **Option A - Add to requirements**:
   - Add `cvxpy>=1.4.0` to `pyproject.toml` dependencies
   - Install: `pip install cvxpy`

2. **Option B - Make it optional** (if benchmarks aren't critical):
   ```python
   try:
       import cvxpy as cp
       CVXPY_AVAILABLE = True
   except ImportError:
       CVXPY_AVAILABLE = False
       # Provide fallback or skip benchmark features
   ```

### Recommendation
Option A - cvxpy is used for Markowitz optimization benchmarks, which are standard features. Should be a required dependency.

### File to Modify
- [pyproject.toml](../pyproject.toml) - Add cvxpy to dependencies

---

## 3. Backtester Data Interface Initialization TODO

**Priority**: 🟡 MEDIUM - Code quality issue

### Problem
- TODO comment at [portwine/backtester/core.py:181](../portwine/backtester/core.py#L181)
- Multiple conditional branches testing different interface types
- Messy logic for handling MultiDataInterface vs DataInterface vs RestrictedDataInterface

### Current Code (Lines 182-194)
```python
def __init__(self, data: DataInterface, calendar: DailyMarketCalendar=DEFAULT_CALENDAR):
    self.data = data
    # Pass all loaders to RestrictedDataInterface if data is MultiDataInterface
    if isinstance(data, MultiDataInterface):
        # If caller already provided a RestrictedDataInterface (or subclass), reuse it
        if isinstance(data, RestrictedDataInterface):
            self.restricted_data = data
        else:
            self.restricted_data = RestrictedDataInterface(data.loaders)
    else:
        # DataInterface case
        self.restricted_data = RestrictedDataInterface({None: data.data_loader})
    self.calendar = calendar
```

### Impact
- Harder to maintain
- Nested conditionals are error-prone
- Unclear initialization path

### Solution Options

**Option 1 - Factory Pattern**:
```python
@classmethod
def _create_restricted_interface(cls, data: DataInterface) -> RestrictedDataInterface:
    """Factory method to create appropriate restricted interface."""
    if isinstance(data, RestrictedDataInterface):
        return data
    elif isinstance(data, MultiDataInterface):
        return RestrictedDataInterface(data.loaders)
    else:
        return RestrictedDataInterface({None: data.data_loader})

def __init__(self, data: DataInterface, calendar: DailyMarketCalendar=DEFAULT_CALENDAR):
    self.data = data
    self.restricted_data = self._create_restricted_interface(data)
    self.calendar = calendar
```

**Option 2 - Interface Protocol**:
Add a `to_restricted()` method to DataInterface base class that each subclass implements.

### Recommendation
Option 1 is cleaner and doesn't require changes to interface classes.

### File to Modify
- [portwine/backtester/core.py](../portwine/backtester/core.py#L181-L194)

---

## 4. MarketDataLoaderAdapter.identifiers() Returns Empty List

**Priority**: 🟢 LOW - Potential future issue

### Problem
- [portwine/data/stores/adapter.py:110](../portwine/data/stores/adapter.py#L110)
- `identifiers()` method returns empty list with comment "rarely used"
- May cause issues if backtester tries to enumerate available tickers

### Current Code
```python
def identifiers(self) -> List[str]:
    """Return available identifiers. Rarely used."""
    return []  # Could implement via self._loader if needed
```

### Impact
- Low priority - current code doesn't rely on this
- Could cause issues if future features need ticker enumeration
- Makes API incomplete

### Solution
```python
def identifiers(self) -> List[str]:
    """Return available identifiers."""
    if hasattr(self._loader, 'identifiers'):
        return self._loader.identifiers()
    return []
```

### File to Modify
- [portwine/data/stores/adapter.py](../portwine/data/stores/adapter.py#L110)

---

## Summary Checklist

Before merging to main:

- [ ] Implement `earliest()` in all DataStore classes
- [ ] Add cvxpy to dependencies and verify tests run
- [ ] Refactor Backtester `__init__` to remove TODO
- [ ] (Optional) Implement `identifiers()` in MarketDataLoaderAdapter
- [ ] Run full test suite and ensure all pass
- [ ] Manual testing of key workflows
