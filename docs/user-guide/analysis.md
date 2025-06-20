# Performance Analysis

Portwine provides comprehensive tools for analyzing strategy performance, from basic metrics to advanced visualizations.

## Built-in Analyzers

Portwine comes with several built-in analyzers that make it easy to understand your strategy's performance:

### Equity Drawdown Analyzer

Analyzes equity curves and drawdowns:

```python
from portwine.analyzers import EquityDrawdownAnalyzer

# Create analyzer
analyzer = EquityDrawdownAnalyzer()

# Plot results
analyzer.plot(results)
```

This generates:
- Equity curve comparison (strategy vs benchmark)
- Drawdown analysis
- Performance metrics table

### Monte Carlo Analyzer

Performs Monte Carlo simulations to assess strategy robustness:

```python
from portwine.analyzers import MonteCarloAnalyzer

# Run Monte Carlo analysis
analyzer = MonteCarloAnalyzer()
analyzer.plot(results, n_simulations=1000)
```

### Seasonality Analyzer

Analyzes performance patterns across different time periods:

```python
from portwine.analyzers import SeasonalityAnalyzer

# Analyze seasonal patterns
analyzer = SeasonalityAnalyzer()
analyzer.plot(results)
```

## Basic Performance Metrics

### Calculating Key Metrics

```python
import pandas as pd
import numpy as np

def calculate_performance_metrics(strategy_returns, benchmark_returns=None):
    """Calculate comprehensive performance metrics."""
    
    # Basic return metrics
    total_return = (1 + strategy_returns).prod() - 1
    annual_return = strategy_returns.mean() * 252
    volatility = strategy_returns.std() * np.sqrt(252)
    
    # Risk metrics
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    
    # Drawdown analysis
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    sharpe_ratio = (annual_return - risk_free_rate) / volatility
    
    # Sortino ratio
    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    # Win rate
    win_rate = (strategy_returns > 0).mean()
    
    # Average win/loss
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    # Profit factor
    profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf
    
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Annual Return': f"{annual_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Sortino Ratio': f"{sortino_ratio:.2f}",
        'Calmar Ratio': f"{calmar_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Win Rate': f"{win_rate:.2%}",
        'Avg Win': f"{avg_win:.2%}",
        'Avg Loss': f"{avg_loss:.2%}",
        'Profit Factor': f"{profit_factor:.2f}"
    }
    
    return metrics

# Calculate metrics
metrics = calculate_performance_metrics(results['strategy_returns'])
for metric, value in metrics.items():
    print(f"{metric}: {value}")
```

### Benchmark Comparison

```python
def compare_to_benchmark(strategy_returns, benchmark_returns):
    """Compare strategy performance to benchmark."""
    
    # Calculate excess returns
    excess_returns = strategy_returns - benchmark_returns
    
    # Information ratio
    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    # Beta calculation
    covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
    
    # Alpha calculation (annualized)
    benchmark_annual_return = benchmark_returns.mean() * 252
    strategy_annual_return = strategy_returns.mean() * 252
    alpha = strategy_annual_return - (0.02 + beta * (benchmark_annual_return - 0.02))
    
    comparison = {
        'Information Ratio': f"{information_ratio:.2f}",
        'Beta': f"{beta:.2f}",
        'Alpha': f"{alpha:.2%}",
        'Excess Return': f"{excess_returns.mean() * 252:.2%}"
    }
    
    return comparison

# Compare to benchmark
comparison = compare_to_benchmark(
    results['strategy_returns'], 
    results['benchmark_returns']
)
```

## Advanced Analysis

### Rolling Performance

```python
def rolling_performance_analysis(returns, window=252):
    """Analyze rolling performance metrics."""
    
    # Rolling Sharpe ratio
    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling drawdown
    rolling_cumulative = (1 + returns).rolling(window).apply(lambda x: (1 + x).prod() - 1)
    rolling_max = rolling_cumulative.expanding().max()
    rolling_drawdown = (rolling_cumulative - rolling_max) / rolling_max
    
    # Rolling volatility
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    
    return {
        'rolling_sharpe': rolling_sharpe,
        'rolling_drawdown': rolling_drawdown,
        'rolling_vol': rolling_vol
    }

# Calculate rolling metrics
rolling_metrics = rolling_performance_analysis(results['strategy_returns'])
```

### Regime Analysis

```python
def regime_analysis(strategy_returns, benchmark_returns):
    """Analyze performance in different market regimes."""
    
    # Define regimes based on benchmark performance
    benchmark_rolling = benchmark_returns.rolling(60).mean() * 252
    
    # Bull market: benchmark > 10% annualized
    bull_mask = benchmark_rolling > 0.10
    bull_returns = strategy_returns[bull_mask]
    
    # Bear market: benchmark < -10% annualized
    bear_mask = benchmark_rolling < -0.10
    bear_returns = strategy_returns[bear_mask]
    
    # Sideways market: between -10% and 10%
    sideways_mask = (benchmark_rolling >= -0.10) & (benchmark_rolling <= 0.10)
    sideways_returns = strategy_returns[sideways_mask]
    
    regimes = {
        'Bull Market': {
            'Return': f"{bull_returns.mean() * 252:.2%}",
            'Volatility': f"{bull_returns.std() * np.sqrt(252):.2%}",
            'Sharpe': f"{(bull_returns.mean() * 252 - 0.02) / (bull_returns.std() * np.sqrt(252)):.2f}",
            'Days': len(bull_returns)
        },
        'Bear Market': {
            'Return': f"{bear_returns.mean() * 252:.2%}",
            'Volatility': f"{bear_returns.std() * np.sqrt(252):.2%}",
            'Sharpe': f"{(bear_returns.mean() * 252 - 0.02) / (bear_returns.std() * np.sqrt(252)):.2f}",
            'Days': len(bear_returns)
        },
        'Sideways Market': {
            'Return': f"{sideways_returns.mean() * 252:.2%}",
            'Volatility': f"{sideways_returns.std() * np.sqrt(252):.2%}",
            'Sharpe': f"{(sideways_returns.mean() * 252 - 0.02) / (sideways_returns.std() * np.sqrt(252)):.2f}",
            'Days': len(sideways_returns)
        }
    }
    
    return regimes

# Analyze performance by regime
regimes = regime_analysis(results['strategy_returns'], results['benchmark_returns'])
```

## Visualization

### Custom Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_performance_summary(results):
    """Create a comprehensive performance summary plot."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Equity curves
    cumulative_strategy = (1 + results['strategy_returns']).cumprod()
    cumulative_benchmark = (1 + results['benchmark_returns']).cumprod()
    
    axes[0, 0].plot(cumulative_strategy.index, cumulative_strategy, label='Strategy')
    axes[0, 0].plot(cumulative_benchmark.index, cumulative_benchmark, label='Benchmark')
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Drawdown
    running_max = cumulative_strategy.expanding().max()
    drawdown = (cumulative_strategy - running_max) / running_max
    
    axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].grid(True)
    
    # 3. Rolling Sharpe ratio
    rolling_sharpe = results['strategy_returns'].rolling(252).mean() / \
                     results['strategy_returns'].rolling(252).std() * np.sqrt(252)
    
    axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe)
    axes[1, 0].axhline(y=0, color='black', linestyle='--')
    axes[1, 0].set_title('Rolling Sharpe Ratio (252-day)')
    axes[1, 0].grid(True)
    
    # 4. Return distribution
    axes[1, 1].hist(results['strategy_returns'], bins=50, alpha=0.7, label='Strategy')
    axes[1, 1].hist(results['benchmark_returns'], bins=50, alpha=0.7, label='Benchmark')
    axes[1, 1].set_title('Return Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Create performance summary
plot_performance_summary(results)
```

### Correlation Analysis

```python
def correlation_analysis(results):
    """Analyze correlations between strategy and benchmark."""
    
    # Calculate rolling correlation
    rolling_corr = results['strategy_returns'].rolling(60).corr(results['benchmark_returns'])
    
    # Plot correlation over time
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_corr.index, rolling_corr)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Rolling 60-day Correlation with Benchmark')
    plt.grid(True)
    plt.show()
    
    # Overall correlation
    overall_corr = results['strategy_returns'].corr(results['benchmark_returns'])
    print(f"Overall correlation with benchmark: {overall_corr:.3f}")

# Analyze correlations
correlation_analysis(results)
```

## Risk Analysis

### Value at Risk (VaR)

```python
def calculate_var(returns, confidence_level=0.05):
    """Calculate Value at Risk."""
    
    # Historical VaR
    var_historical = np.percentile(returns, confidence_level * 100)
    
    # Parametric VaR (assuming normal distribution)
    mean_return = returns.mean()
    std_return = returns.std()
    var_parametric = mean_return + std_return * norm.ppf(confidence_level)
    
    return {
        'Historical VaR': f"{var_historical:.2%}",
        'Parametric VaR': f"{var_parametric:.2%}"
    }

# Calculate VaR
var_metrics = calculate_var(results['strategy_returns'])
```

### Stress Testing

```python
def stress_test(strategy_returns, stress_scenarios):
    """Perform stress testing on strategy."""
    
    results = {}
    
    for scenario_name, stress_factor in stress_scenarios.items():
        # Apply stress factor to returns
        stressed_returns = strategy_returns * stress_factor
        
        # Calculate stressed metrics
        stressed_total_return = (1 + stressed_returns).prod() - 1
        stressed_max_drawdown = calculate_max_drawdown(stressed_returns)
        
        results[scenario_name] = {
            'Total Return': f"{stressed_total_return:.2%}",
            'Max Drawdown': f"{stressed_max_drawdown:.2%}"
        }
    
    return results

# Define stress scenarios
stress_scenarios = {
    'Market Crash (50% decline)': 0.5,
    'High Volatility (2x)': 2.0,
    'Low Volatility (0.5x)': 0.5
}

# Run stress tests
stress_results = stress_test(results['strategy_returns'], stress_scenarios)
```

## Best Practices

### 1. Use Multiple Time Periods

```python
# Test on different time periods
periods = [
    ('2020-01-01', '2020-12-31'),  # COVID year
    ('2021-01-01', '2021-12-31'),  # Recovery year
    ('2022-01-01', '2022-12-31'),  # Bear market year
]

for start, end in periods:
    period_results = backtester.run_backtest(
        strategy=strategy,
        start_date=start,
        end_date=end
    )
    print(f"\n{start} to {end}:")
    metrics = calculate_performance_metrics(period_results['strategy_returns'])
    print(f"Annual Return: {metrics['Annual Return']}")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']}")
```

### 2. Validate Results

```python
def validate_backtest_results(results):
    """Validate backtest results for common issues."""
    
    issues = []
    
    # Check for unrealistic returns
    if results['strategy_returns'].max() > 0.5:  # 50% daily return
        issues.append("Unrealistic daily returns detected")
    
    # Check for perfect timing
    if results['strategy_returns'].corr(results['benchmark_returns']) > 0.99:
        issues.append("Suspiciously high correlation with benchmark")
    
    # Check for lookahead bias
    signals = results['signals_df']
    if signals.isnull().sum().sum() == 0:
        issues.append("No missing signals - potential lookahead bias")
    
    return issues

# Validate results
issues = validate_backtest_results(results)
if issues:
    print("Validation issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

### 3. Document Assumptions

```python
# Document your analysis assumptions
analysis_assumptions = {
    'Risk-free rate': '2% annual',
    'Transaction costs': '0% (frictionless)',
    'Slippage': '0%',
    'Rebalancing': 'Daily',
    'Data source': 'EODHD',
    'Benchmark': 'SPY'
}

print("Analysis Assumptions:")
for assumption, value in analysis_assumptions.items():
    print(f"  {assumption}: {value}")
```

## Next Steps

- Learn about [building strategies](strategies.md)
- Explore [backtesting](backtesting.md)
- Check out [data management](data-management.md) 