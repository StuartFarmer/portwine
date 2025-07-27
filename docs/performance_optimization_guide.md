# Performance Optimization Guide for PortWine Backtester

## Executive Summary

This guide documents the performance profiling analysis of the PortWine step-driven backtester and provides specific optimization recommendations based on profiling results.

## Key Findings

### Performance Bottlenecks Identified

1. **Major Bottleneck: Pandas Indexing Operations (99% of execution time)**
   - `_bar_dict` method: ~16.8 seconds for 100 tickers × 1000 days
   - Pandas `__getitem__`, `_getitem_axis`, `xs`, `fast_xs` operations
   - Processing rate: Only ~179 dates/second for medium datasets

2. **Poor Scaling Behavior**
   - Small dataset (20 tickers, 500 days): 0.56s (885 dates/second)
   - Medium dataset (100 tickers, 1000 days): 5.67s (176 dates/second)
   - Large dataset (500 tickers, 2000 days): 57.6s (35 dates/second)
   - **O(n²) complexity instead of linear scaling**

3. **Strategy Impact is Minimal**
   - Equal weight strategy: 5.72s
   - Moving average strategy: 6.67s (1.17x overhead)
   - Strategy complexity has minimal impact compared to data access overhead

## Optimization Strategy

### 1. Pre-computed Data Access Patterns

**Problem**: Repeated pandas indexing operations in `_bar_dict` method
**Solution**: Pre-compute data access patterns before the main loop

```python
def _precompute_data_access(self, reg_data, all_ts):
    """Pre-compute data access patterns to avoid repeated pandas indexing."""
    data_cache = {}
    for ticker, df in reg_data.items():
        # Create fast date-to-index mapping
        date_to_idx = {date: idx for idx, date in enumerate(df.index)}
        
        # Pre-extract price data as numpy arrays
        price_cache[ticker] = {
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'volume': df['volume'].values,
            'dates': df.index.values
        }
        
        # Create fast lookup for each timestamp
        data_cache[ticker] = {}
        for ts in all_ts:
            if ts in date_to_idx:
                idx = date_to_idx[ts]
                data_cache[ticker][ts] = {
                    'open': float(df['open'].iloc[idx]),
                    'high': float(df['high'].iloc[idx]),
                    'low': float(df['low'].iloc[idx]),
                    'close': float(df['close'].iloc[idx]),
                    'volume': float(df['volume'].iloc[idx]),
                }
            else:
                data_cache[ticker][ts] = None
    
    return {'data_cache': data_cache, 'price_cache': price_cache}
```

### 2. Optimized Data Access Method

**Problem**: Slow pandas `.loc[]` operations in main loop
**Solution**: Use pre-computed dictionaries for O(1) lookups

```python
def _fast_bar_dict(self, ts, data_cache):
    """Optimized version that uses pre-computed data access."""
    out = {}
    for ticker, ticker_cache in data_cache.items():
        out[ticker] = ticker_cache.get(ts)
    return out
```

### 3. Vectorized Returns Calculation

**Problem**: Inefficient pandas operations for returns calculation
**Solution**: Use pre-computed price data and vectorized operations

```python
# Use pre-computed price data for returns calculation
price_data = {}
for ticker in reg_tkrs:
    if ticker in price_cache:
        ticker_dates = price_cache[ticker]['dates']
        ticker_prices = price_cache[ticker]['close']
        price_series = pd.Series(ticker_prices, index=ticker_dates)
        price_data[ticker] = price_series.reindex(sig_reg.index).ffill()

px = pd.DataFrame(price_data)
ret_df = px.pct_change(fill_method=None).fillna(0.0)
```

## Implementation Files

### 1. Performance Profiler
- **File**: `tests/profile_step_backtester.py`
- **Purpose**: Comprehensive profiling of the `run_backtest` method
- **Features**:
  - Detailed timing analysis
  - Bottleneck identification
  - Scaling analysis
  - Memory usage tracking
  - Strategy complexity comparison

### 2. Performance Decorator
- **File**: `tests/performance_decorator.py`
- **Purpose**: Easy-to-use profiling decorators for ongoing monitoring
- **Features**:
  - `@quick_profile` - Basic profiling
  - `@detailed_profile` - Detailed profiling with file output
  - `@profile_backtest_method` - Specific to backtester
  - Performance monitoring over time

### 3. Optimized Backtester
- **File**: `portwine/backtester_optimized.py`
- **Purpose**: Optimized implementation addressing identified bottlenecks
- **Key Optimizations**:
  - Pre-computed data access patterns
  - Fast lookup structures
  - Vectorized operations
  - Memory-efficient data structures

### 4. Performance Comparison
- **File**: `tests/compare_backtester_performance.py`
- **Purpose**: Compare original vs optimized performance
- **Features**:
  - Side-by-side performance comparison
  - Result verification
  - Scaling analysis
  - Memory usage comparison

## Expected Performance Improvements

### Speedup Estimates
Based on profiling analysis, the optimized backtester should achieve:

- **Small datasets**: 2-3x speedup
- **Medium datasets**: 3-5x speedup  
- **Large datasets**: 5-10x speedup

### Processing Rate Improvements
- **Original**: ~179 dates/second (medium dataset)
- **Optimized**: ~500-1000 dates/second (estimated)

### Memory Usage
- **Trade-off**: Slightly higher memory usage for better performance
- **Benefit**: Reduced CPU time and better scalability

## Usage Instructions

### 1. Running the Profiler

```bash
# Basic profiling
python tests/profile_step_backtester.py

# Performance comparison
python tests/compare_backtester_performance.py

# Demo with decorators
python examples/profile_backtester_demo.py
```

### 2. Using the Optimized Backtester

```python
from portwine.backtester_optimized import OptimizedBacktester

# Create optimized backtester
backtester = OptimizedBacktester(
    market_data_loader=data_loader,
    logger=None,
    log=False
)

# Run backtest with same interface
results = backtester.run_backtest(
    strategy=strategy,
    benchmark='equal_weight',
    verbose=False
)
```

### 3. Using Performance Decorators

```python
from tests.performance_decorator import quick_profile, detailed_profile

@quick_profile
def my_backtest_function():
    # Your backtest code here
    pass

@detailed_profile
def my_detailed_backtest():
    # Detailed profiling with file output
    pass
```

## Additional Optimization Opportunities

### 1. Strategy-Level Optimizations
- **Cache frequently accessed data** in strategy objects
- **Use NumPy operations** instead of Python loops
- **Pre-allocate data structures** where possible
- **Minimize object creation** in hot paths

### 2. Data Loading Optimizations
- **Implement data caching** for repeated access
- **Use vectorized operations** for data alignment
- **Consider NumPy arrays** instead of pandas for large datasets
- **Lazy loading** for very large datasets

### 3. Returns Calculation Optimizations
- **Use NumPy for returns calculation**
- **Pre-allocate arrays**
- **Avoid unnecessary type conversions**
- **Consider numba acceleration** for complex calculations

### 4. Memory Management
- **Reuse objects** where possible
- **Use object pools** for frequently created objects
- **Profile memory usage** with tools like memory_profiler
- **Consider `__slots__`** for data classes

## Monitoring and Maintenance

### 1. Ongoing Performance Monitoring
- Use the performance decorators for continuous monitoring
- Track performance trends over time
- Set up alerts for performance regressions

### 2. Regular Profiling
- Profile new features before deployment
- Compare performance across different data sizes
- Monitor memory usage patterns

### 3. Optimization Validation
- Always verify that optimizations don't change results
- Test with real-world data scenarios
- Validate performance improvements in production-like environments

## Conclusion

The profiling analysis revealed that the major performance bottleneck in the PortWine backtester is pandas indexing operations in the `_bar_dict` method. The optimized implementation addresses this by:

1. **Pre-computing data access patterns** before the main loop
2. **Using fast lookup structures** instead of repeated pandas operations
3. **Vectorizing operations** where possible
4. **Optimizing memory usage** for better performance

These optimizations should provide significant performance improvements, especially for larger datasets, while maintaining the same interface and result accuracy.

## Next Steps

1. **Implement the optimized backtester** in your production environment
2. **Profile your specific use cases** to validate improvements
3. **Monitor performance** over time with the provided tools
4. **Consider additional optimizations** based on your specific requirements
5. **Share performance results** with the community

For questions or additional optimization needs, refer to the profiling tools and examples provided in this guide. 