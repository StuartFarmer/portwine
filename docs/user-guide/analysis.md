# Performance Analysis

Portwine provides comprehensive tools for analyzing strategy performance, from basic metrics to advanced visualizations.

## Built-in Analyzers

Portwine comes with a comprehensive suite of analyzers that make it easy to understand your strategy's performance across different dimensions:

### Equity Drawdown Analyzer

The foundational analyzer that provides essential performance visualization and metrics. This analyzer creates clear, professional plots showing equity curves and drawdown analysis.

**What it generates:**
- Equity curve comparison (strategy vs benchmark) with clear visual distinction
- Drawdown analysis showing peak-to-trough declines
- Performance metrics table with key statistics
- Professional formatting with proper legends and grid lines

**Best for:** Initial strategy evaluation, performance overview, and stakeholder presentations.

```python
from portwine.analyzers import EquityDrawdownAnalyzer

# Create analyzer
analyzer = EquityDrawdownAnalyzer()

# Plot results
analyzer.plot(results)
```

### Grid Equity Drawdown Analyzer

A powerful multi-strategy comparison tool that displays multiple strategy results in a grid layout. Each grid cell contains both equity curves and drawdown analysis for easy side-by-side comparison.

**What it generates:**
- Grid layout with customizable columns (default: 2 columns)
- Each cell shows equity curves and drawdowns for one strategy
- Color-coded fill areas showing outperformance/underperformance
- Compact design perfect for comparing multiple strategies or parameter sets

**Best for:** Comparing multiple strategies, parameter optimization results, or basket analysis.

```python
from portwine.analyzers import GridEquityDrawdownAnalyzer

# Create analyzer
analyzer = GridEquityDrawdownAnalyzer()

# Plot multiple strategies
analyzer.plot(results_list, titles, ncols=2)
```

### Monte Carlo Analyzer

Performs Monte Carlo simulations to assess strategy robustness and understand the distribution of possible outcomes. This analyzer helps determine if your strategy's performance is stable or subject to significant randomness.

**What it generates:**
- Distribution of possible outcomes through random sampling
- Confidence intervals for key metrics
- Risk assessment through multiple simulation paths
- Statistical validation of strategy performance

**Best for:** Risk assessment, strategy validation, and understanding performance uncertainty.

```python
from portwine.analyzers import MonteCarloAnalyzer

# Run Monte Carlo analysis
analyzer = MonteCarloAnalyzer()
analyzer.plot(results, n_simulations=1000)
```

### Seasonality Analyzer

Analyzes performance patterns across different time periods to identify seasonal effects, day-of-week patterns, and other temporal dependencies in your strategy.

**What it generates:**
- Monthly performance heatmaps and bar charts
- Day-of-week analysis showing intra-week patterns
- Quarterly and annual seasonal trends
- Statistical significance testing for seasonal effects

**Best for:** Understanding temporal patterns, optimizing rebalancing schedules, and identifying seasonal opportunities.

```python
from portwine.analyzers import SeasonalityAnalyzer

# Analyze seasonal patterns
analyzer = SeasonalityAnalyzer()
analyzer.plot(results, period="monthly")
```

### Correlation Analyzer

Computes and visualizes correlation matrices among the assets in your strategy. This analyzer helps understand the relationships between different positions and identify potential diversification benefits or concentration risks.

**What it generates:**
- Correlation matrix heatmap with color-coded values
- Statistical correlation analysis using multiple methods (Pearson, Spearman, Kendall)
- Asset relationship visualization
- Diversification assessment

**Best for:** Portfolio construction, risk management, and understanding asset relationships.

```python
from portwine.analyzers import CorrelationAnalyzer

# Analyze correlations
analyzer = CorrelationAnalyzer(method='pearson')
analyzer.plot(results)
```

### Strategy Comparison Analyzer

A comprehensive tool for comparing two strategies side-by-side with statistical rigor. This analyzer provides detailed statistical tests and rolling analysis to understand the differences between strategies.

**What it generates:**
- Side-by-side equity curve comparison with fill areas
- Statistical significance tests (t-tests) between strategies
- Rolling correlation, alpha, and beta analysis
- Performance metrics comparison table

**Best for:** Strategy selection, A/B testing, and understanding strategy differences.

```python
from portwine.analyzers import StrategyComparisonAnalyzer

# Compare strategies
analyzer = StrategyComparisonAnalyzer()
analyzer.plot(results, comparison_results, label_main="Strategy A", label_compare="Strategy B")
```

### Train Test Equity Drawdown Analyzer

Evaluates strategy robustness by splitting data into training and testing periods. This analyzer helps identify overfitting and ensures your strategy generalizes well to unseen data.

**What it generates:**
- Equity curves with clear train/test split visualization
- Drawdown analysis for both periods
- Histogram comparison of train vs test returns
- Comprehensive metrics table with overfitting ratios
- Color-coded performance indicators

**Best for:** Model validation, overfitting detection, and ensuring strategy robustness.

```python
from portwine.analyzers import TrainTestEquityDrawdownAnalyzer

# Analyze train/test performance
analyzer = TrainTestEquityDrawdownAnalyzer()
analyzer.plot(results, split=0.7)
```

### Student's T-Test Analyzer

Provides statistical rigor to your strategy evaluation through formal hypothesis testing. This analyzer determines whether your strategy's performance is statistically significant compared to zero or a benchmark.

**What it generates:**
- Statistical significance testing vs zero returns
- Comparison testing vs benchmark returns
- Return distribution histograms
- Color-coded significance indicators
- Plain English interpretation of results

**Best for:** Statistical validation, academic research, and formal strategy evaluation.

```python
from portwine.analyzers import StudentsTTestAnalyzer

# Perform statistical testing
analyzer = StudentsTTestAnalyzer()
analyzer.plot(results, with_equity_curve=True)
```

### Transaction Cost Analyzer

Models the real-world impact of transaction costs on strategy performance. This analyzer helps understand how different cost levels affect your strategy and identifies the breakeven point for profitability.

**What it generates:**
- Performance degradation analysis across cost levels
- Portfolio turnover analysis with rolling metrics
- Breakeven analysis showing cost tolerance
- Equity curves for different cost scenarios
- Comprehensive cost impact report

**Best for:** Real-world implementation planning, cost optimization, and profitability analysis.

```python
from portwine.analyzers import TransactionCostAnalyzer

# Analyze transaction cost impact
analyzer = TransactionCostAnalyzer(cost_levels=[0, 0.0005, 0.001, 0.002, 0.005])
analyzer.plot(results)
```

### Noise Robustness Analyzer

Tests strategy stability by injecting controlled levels of noise into market data. This analyzer helps determine if your strategy is robust to market noise or if it's overfitted to specific data patterns.

**What it generates:**
- Performance stability across noise levels
- Statistical distribution of outcomes
- Robustness metrics and confidence intervals
- Noise tolerance assessment

**Best for:** Strategy validation, robustness testing, and overfitting detection.

```python
from portwine.analyzers import NoiseRobustnessAnalyzer

# Test noise robustness
analyzer = NoiseRobustnessAnalyzer(base_loader, noise_levels=[0.5, 1.0, 1.5, 2.0])
analyzer.plot(results)
```

### Regime Change Analyzer

Analyzes strategy performance across different market regimes (bull, bear, volatile, etc.). This analyzer helps identify how your strategy behaves in different market conditions and potential vulnerabilities.

**What it generates:**
- Market regime identification and classification
- Performance metrics for each regime
- Regime transition analysis
- Strategy behavior across market conditions
- Comprehensive regime performance report

**Best for:** Risk management, strategy optimization, and understanding market condition dependencies.

```python
from portwine.analyzers import RegimeChangeAnalyzer

# Analyze regime performance
analyzer = RegimeChangeAnalyzer()
analyzer.plot(results, method='combined')
```

## Writing Your Own Analyzers

You can also write your own analyzers by following the simple analyzer API. Here's a step-by-step example of creating a custom "Volatility Regime Analyzer" that identifies and analyzes performance in different volatility environments.

### Step 1: Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portwine.analyzers.base import Analyzer
```

**Explanation:**
- `pandas` and `numpy` for data manipulation and calculations
- `matplotlib.pyplot` for creating visualizations
- `Analyzer` base class from portwine that all analyzers must inherit from

### Step 2: Define Your Analyzer Class

```python
class VolatilityRegimeAnalyzer(Analyzer):
    """
    Custom analyzer that identifies different volatility regimes and analyzes
    strategy performance in each regime.
    
    This analyzer helps understand how your strategy performs in:
    - Low volatility periods (calm markets)
    - Medium volatility periods (normal markets) 
    - High volatility periods (stressful markets)
    """
```

**Explanation:**
- Inherit from the `Analyzer` base class
- Add a comprehensive docstring explaining what your analyzer does
- This makes your analyzer compatible with the portwine framework

### Step 3: Initialize Your Analyzer

```python
    def __init__(self, volatility_window=60, regime_thresholds=None):
        """
        Initialize the analyzer with customizable parameters.
        
        Parameters
        ----------
        volatility_window : int, default 60
            Rolling window for volatility calculation (in days)
        regime_thresholds : dict, optional
            Custom thresholds for regime classification
        """
        self.volatility_window = volatility_window
        
        # Default thresholds (30th and 70th percentiles)
        self.regime_thresholds = regime_thresholds or {
            'low': 0.30,    # Below 30th percentile = low volatility
            'high': 0.70    # Above 70th percentile = high volatility
        }
```

**Explanation:**
- `__init__` method sets up your analyzer with configurable parameters
- `volatility_window` determines how many days to use for rolling volatility
- `regime_thresholds` allows users to customize how regimes are defined
- Default thresholds use 30th and 70th percentiles for robust regime classification

### Step 4: Implement the Analysis Logic

```python
    def identify_volatility_regimes(self, returns):
        """
        Identify volatility regimes based on rolling volatility.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns series
            
        Returns
        -------
        pd.Series
            Regime labels for each date
        """
        # Calculate rolling volatility (annualized)
        rolling_vol = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
        
        # Calculate percentile thresholds
        low_threshold = rolling_vol.quantile(self.regime_thresholds['low'])
        high_threshold = rolling_vol.quantile(self.regime_thresholds['high'])
        
        # Classify regimes
        regimes = pd.Series(index=returns.index, dtype='object')
        regimes[rolling_vol <= low_threshold] = 'low_vol'
        regimes[rolling_vol >= high_threshold] = 'high_vol'
        regimes[(rolling_vol > low_threshold) & (rolling_vol < high_threshold)] = 'medium_vol'
        
        return regimes, rolling_vol
```

**Explanation:**
- `identify_volatility_regimes` is a helper method that does the core analysis
- Calculates rolling volatility using the specified window
- Annualizes volatility by multiplying by √252 (trading days)
- Uses quantiles to determine regime thresholds dynamically
- Returns both regime labels and the volatility series for plotting

### Step 5: Calculate Performance Metrics

```python
    def calculate_regime_metrics(self, returns, regimes, ann_factor=252):
        """
        Calculate performance metrics for each volatility regime.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns
        regimes : pd.Series
            Regime labels for each date
        ann_factor : int, default 252
            Annualization factor
            
        Returns
        -------
        dict
            Performance metrics for each regime
        """
        metrics = {}
        
        for regime in ['low_vol', 'medium_vol', 'high_vol']:
            # Filter returns for this regime
            regime_returns = returns[regimes == regime]
            
            if len(regime_returns) == 0:
                metrics[regime] = None
                continue
                
            # Calculate basic metrics
            total_return = (1 + regime_returns).prod() - 1
            annual_return = regime_returns.mean() * ann_factor
            volatility = regime_returns.std() * np.sqrt(ann_factor)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Calculate maximum drawdown
            cumulative = (1 + regime_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate win rate
            win_rate = (regime_returns > 0).mean()
            
            metrics[regime] = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'days': len(regime_returns)
            }
        
        return metrics
```

**Explanation:**
- `calculate_regime_metrics` computes performance statistics for each regime
- Filters returns by regime and calculates comprehensive metrics
- Handles edge cases (empty regimes) gracefully
- Returns a structured dictionary with all metrics organized by regime

### Step 6: Implement the Required Plot Method

```python
    def plot(self, results, figsize=(15, 10), benchmark_label="Benchmark"):
        """
        Create comprehensive visualization of volatility regime analysis.
        
        Parameters
        ----------
        results : dict
            Results dictionary from backtester
        figsize : tuple, default (15, 10)
            Figure size (width, height)
        benchmark_label : str, default "Benchmark"
            Label for benchmark in plots
        """
        # Extract data from results
        strategy_returns = results['strategy_returns']
        benchmark_returns = results.get('benchmark_returns', pd.Series(dtype=float))
        
        # Identify regimes
        regimes, rolling_vol = self.identify_volatility_regimes(benchmark_returns)
        
        # Calculate metrics
        strategy_metrics = self.calculate_regime_metrics(strategy_returns, regimes)
        benchmark_metrics = self.calculate_regime_metrics(benchmark_returns, regimes)
        
        # Create the visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Volatility Regime Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Rolling Volatility with Regime Overlay
        ax1 = axes[0, 0]
        ax1.plot(rolling_vol.index, rolling_vol.values, color='blue', linewidth=1, label='Rolling Volatility')
        
        # Color-code by regime
        for regime, color in [('low_vol', 'green'), ('medium_vol', 'yellow'), ('high_vol', 'red')]:
            regime_mask = regimes == regime
            if regime_mask.any():
                ax1.scatter(rolling_vol[regime_mask].index, rolling_vol[regime_mask].values, 
                           c=color, alpha=0.6, s=20, label=f'{regime.replace("_", " ").title()}')
        
        ax1.set_title('Rolling Volatility with Regime Classification')
        ax1.set_ylabel('Annualized Volatility')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Returns by Regime
        ax2 = axes[0, 1]
        cumulative_strategy = (1 + strategy_returns).cumprod()
        
        for regime, color in [('low_vol', 'green'), ('medium_vol', 'yellow'), ('high_vol', 'red')]:
            regime_mask = regimes == regime
            if regime_mask.any():
                regime_returns = strategy_returns[regime_mask]
                regime_cumulative = (1 + regime_returns).cumprod()
                ax2.plot(regime_cumulative.index, regime_cumulative.values, 
                        color=color, linewidth=2, label=f'{regime.replace("_", " ").title()}')
        
        ax2.set_title('Strategy Performance by Volatility Regime')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance Metrics Comparison
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        # Create metrics table
        metrics_data = []
        for regime in ['low_vol', 'medium_vol', 'high_vol']:
            if strategy_metrics[regime]:
                metrics_data.append([
                    regime.replace('_', ' ').title(),
                    f"{strategy_metrics[regime]['annual_return']:.2%}",
                    f"{strategy_metrics[regime]['sharpe_ratio']:.2f}",
                    f"{strategy_metrics[regime]['max_drawdown']:.2%}",
                    f"{strategy_metrics[regime]['days']}"
                ])
        
        table = ax3.table(cellText=metrics_data,
                         colLabels=['Regime', 'Ann. Return', 'Sharpe', 'Max DD', 'Days'],
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax3.set_title('Strategy Performance by Regime')
        
        # Plot 4: Regime Distribution
        ax4 = axes[1, 1]
        regime_counts = regimes.value_counts()
        colors = ['green', 'yellow', 'red']
        ax4.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%', colors=colors)
        ax4.set_title('Distribution of Volatility Regimes')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        self._print_summary(strategy_metrics, benchmark_metrics)
```

**Explanation:**
- The `plot` method is the main interface that users will call
- Extracts data from the results dictionary
- Calls helper methods to perform analysis
- Creates a comprehensive 2x2 subplot layout
- Each subplot shows different aspects of the analysis
- Includes proper labeling, legends, and formatting
- Calls a helper method to print summary statistics

### Step 7: Add Helper Methods

```python
    def _print_summary(self, strategy_metrics, benchmark_metrics):
        """
        Print a summary of the analysis results.
        
        Parameters
        ----------
        strategy_metrics : dict
            Strategy performance metrics by regime
        benchmark_metrics : dict
            Benchmark performance metrics by regime
        """
        print("\n" + "="*60)
        print("VOLATILITY REGIME ANALYSIS SUMMARY")
        print("="*60)
        
        for regime in ['low_vol', 'medium_vol', 'high_vol']:
            if strategy_metrics[regime]:
                print(f"\n{regime.replace('_', ' ').title()} Regime:")
                print(f"  Days: {strategy_metrics[regime]['days']}")
                print(f"  Annual Return: {strategy_metrics[regime]['annual_return']:.2%}")
                print(f"  Sharpe Ratio: {strategy_metrics[regime]['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {strategy_metrics[regime]['max_drawdown']:.2%}")
                print(f"  Win Rate: {strategy_metrics[regime]['win_rate']:.2%}")
        
        print("\n" + "="*60)
```

**Explanation:**
- Helper method to print formatted summary statistics
- Provides clear, readable output of key findings
- Uses consistent formatting for professional presentation

### Step 8: Usage Example

```python
# Create and use your custom analyzer
from portwine.analyzers import VolatilityRegimeAnalyzer

# Initialize with custom parameters
analyzer = VolatilityRegimeAnalyzer(
    volatility_window=90,  # 90-day rolling window
    regime_thresholds={
        'low': 0.25,    # Bottom 25% = low volatility
        'high': 0.75    # Top 25% = high volatility
    }
)

# Run the analysis
analyzer.plot(results, figsize=(16, 12))
```

**Explanation:**
- Shows how to instantiate your custom analyzer
- Demonstrates parameter customization
- Shows the simple interface for running the analysis

### Key Design Principles

1. **Inherit from Analyzer**: Always inherit from the base `Analyzer` class
2. **Clear Documentation**: Provide comprehensive docstrings for all methods
3. **Flexible Parameters**: Allow users to customize key parameters
4. **Error Handling**: Handle edge cases gracefully (empty data, missing regimes)
5. **Professional Visualization**: Create clear, informative plots with proper formatting
6. **Modular Design**: Break complex logic into smaller, focused methods
7. **Consistent Interface**: Follow the same pattern as built-in analyzers

This example demonstrates how to create a sophisticated custom analyzer that provides valuable insights while maintaining the same professional interface as the built-in analyzers.

## Next Steps

- Learn about [building strategies](strategies.md)
- Explore [backtesting](backtesting.md)
- Check out [data management](data-management.md) 