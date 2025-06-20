# Analyzers API

Portwine provides a comprehensive suite of analyzers to help you understand and visualize strategy performance.

## Base Analyzer

All analyzers inherit from the base `Analyzer` class:

```python
from portwine.analyzers import Analyzer

class Analyzer:
    """
    Base class for all analyzers in portwine.
    """
    
    def plot(self, results: Dict[str, pd.DataFrame], **kwargs):
        """
        Generate plots for the given backtest results.
        
        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Results dictionary from Backtester.run_backtest()
        **kwargs
            Additional plotting parameters
        """
        raise NotImplementedError("Subclasses must implement plot method")
```

## Built-in Analyzers

### EquityDrawdownAnalyzer

Analyzes equity curves and drawdowns:

```python
from portwine.analyzers import EquityDrawdownAnalyzer

class EquityDrawdownAnalyzer(Analyzer):
    """
    Analyzer for equity curves and drawdown analysis.
    
    Generates:
    - Equity curve comparison (strategy vs benchmark)
    - Drawdown analysis
    - Performance metrics table
    """
    
    def plot(self, results: Dict[str, pd.DataFrame], **kwargs):
        """
        Plot equity curves and drawdown analysis.
        
        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Results from Backtester.run_backtest()
        **kwargs
            Additional plotting parameters:
            - figsize : tuple, default (15, 10)
            - title : str, default "Equity and Drawdown Analysis"
        """
        # Implementation details...
```

#### Usage Example

```python
# Create analyzer
analyzer = EquityDrawdownAnalyzer()

# Plot results
analyzer.plot(results)

# With custom parameters
analyzer.plot(
    results,
    figsize=(20, 12),
    title="My Strategy Performance"
)
```

### MonteCarloAnalyzer

Performs Monte Carlo simulations:

```python
from portwine.analyzers import MonteCarloAnalyzer

class MonteCarloAnalyzer(Analyzer):
    """
    Monte Carlo simulation analyzer.
    
    Generates:
    - Distribution of possible outcomes
    - Confidence intervals
    - Risk metrics
    """
    
    def plot(self, results: Dict[str, pd.DataFrame], n_simulations: int = 1000, **kwargs):
        """
        Run Monte Carlo simulation and plot results.
        
        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Results from Backtester.run_backtest()
        n_simulations : int, default 1000
            Number of Monte Carlo simulations to run
        **kwargs
            Additional plotting parameters
        """
        # Implementation details...
```

#### Usage Example

```python
# Create analyzer
analyzer = MonteCarloAnalyzer()

# Run Monte Carlo analysis
analyzer.plot(results, n_simulations=5000)

# With custom parameters
analyzer.plot(
    results,
    n_simulations=10000,
    confidence_levels=[0.05, 0.25, 0.5, 0.75, 0.95]
)
```

### SeasonalityAnalyzer

Analyzes seasonal patterns:

```python
from portwine.analyzers import SeasonalityAnalyzer

class SeasonalityAnalyzer(Analyzer):
    """
    Seasonality analysis analyzer.
    
    Generates:
    - Monthly performance patterns
    - Day-of-week patterns
    - Seasonal trends
    """
    
    def plot(self, results: Dict[str, pd.DataFrame], **kwargs):
        """
        Analyze and plot seasonal patterns.
        
        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Results from Backtester.run_backtest()
        **kwargs
            Additional plotting parameters:
            - period : str, default "monthly"
                Analysis period ("monthly", "weekly", "quarterly")
        """
        # Implementation details...
```

#### Usage Example

```python
# Create analyzer
analyzer = SeasonalityAnalyzer()

# Analyze monthly patterns
analyzer.plot(results, period="monthly")

# Analyze weekly patterns
analyzer.plot(results, period="weekly")
```

## Creating Custom Analyzers

### Basic Analyzer Template

```python
from portwine.analyzers import Analyzer
import matplotlib.pyplot as plt
import pandas as pd

class MyCustomAnalyzer(Analyzer):
    """
    Custom analyzer for specific analysis needs.
    """
    
    def __init__(self, **parameters):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        **parameters
            Analyzer-specific parameters
        """
        self.parameters = parameters
    
    def plot(self, results: Dict[str, pd.DataFrame], **kwargs):
        """
        Generate custom analysis plots.
        
        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Results from Backtester.run_backtest()
        **kwargs
            Additional plotting parameters
        """
        # Extract data from results
        strategy_returns = results['strategy_returns']
        benchmark_returns = results['benchmark_returns']
        
        # Your custom analysis logic here
        self._analyze_data(strategy_returns, benchmark_returns)
        
        # Generate plots
        self._create_plots(**kwargs)
    
    def _analyze_data(self, strategy_returns: pd.Series, benchmark_returns: pd.Series):
        """Perform data analysis."""
        # Your analysis logic here
        pass
    
    def _create_plots(self, **kwargs):
        """Create visualization plots."""
        # Your plotting logic here
        pass
```

### Advanced Analyzer Example

```python
class RiskMetricsAnalyzer(Analyzer):
    """
    Comprehensive risk metrics analyzer.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def plot(self, results: Dict[str, pd.DataFrame], **kwargs):
        """Generate comprehensive risk analysis."""
        
        strategy_returns = results['strategy_returns']
        benchmark_returns = results['benchmark_returns']
        
        # Calculate risk metrics
        metrics = self._calculate_risk_metrics(strategy_returns, benchmark_returns)
        
        # Create visualization
        self._create_risk_plots(metrics, **kwargs)
        
        # Print summary
        self._print_summary(metrics)
    
    def _calculate_risk_metrics(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """Calculate comprehensive risk metrics."""
        
        # Basic metrics
        annual_return = strategy_returns.mean() * 252
        volatility = strategy_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility
        
        # Drawdown analysis
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR and CVaR
        var_95 = np.percentile(strategy_returns, 5)
        cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
        
        # Beta and Alpha
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        benchmark_annual_return = benchmark_returns.mean() * 252
        alpha = annual_return - (self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate))
        
        return {
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'beta': beta,
            'alpha': alpha
        }
    
    def _create_risk_plots(self, metrics: Dict, **kwargs):
        """Create risk visualization plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=kwargs.get('figsize', (15, 10)))
        
        # 1. Risk-return scatter
        axes[0, 0].scatter(metrics['volatility'], metrics['annual_return'], s=100, alpha=0.7)
        axes[0, 0].set_xlabel('Volatility')
        axes[0, 0].set_ylabel('Annual Return')
        axes[0, 0].set_title('Risk-Return Profile')
        axes[0, 0].grid(True)
        
        # 2. Sharpe ratio
        axes[0, 1].bar(['Strategy'], [metrics['sharpe_ratio']])
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_title('Risk-Adjusted Return')
        axes[0, 1].grid(True)
        
        # 3. Drawdown
        axes[1, 0].bar(['Max Drawdown'], [abs(metrics['max_drawdown'])])
        axes[1, 0].set_ylabel('Drawdown')
        axes[1, 0].set_title('Maximum Drawdown')
        axes[1, 0].grid(True)
        
        # 4. Beta
        axes[1, 1].bar(['Beta'], [metrics['beta']])
        axes[1, 1].axhline(y=1, color='red', linestyle='--', label='Market Beta')
        axes[1, 1].set_ylabel('Beta')
        axes[1, 1].set_title('Market Beta')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _print_summary(self, metrics: Dict):
        """Print risk metrics summary."""
        
        print("=== Risk Metrics Summary ===")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"VaR (95%): {metrics['var_95']:.2%}")
        print(f"CVaR (95%): {metrics['cvar_95']:.2%}")
        print(f"Beta: {metrics['beta']:.2f}")
        print(f"Alpha: {metrics['alpha']:.2%}")
```

## Analyzer Best Practices

### 1. Consistent Interface

```python
class ConsistentAnalyzer(Analyzer):
    def plot(self, results: Dict[str, pd.DataFrame], **kwargs):
        """
        Standard plot method with consistent parameters.
        
        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Results from Backtester.run_backtest()
        figsize : tuple, optional
            Figure size (width, height)
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        # Extract common parameters
        figsize = kwargs.get('figsize', (12, 8))
        title = kwargs.get('title', 'Analysis Results')
        save_path = kwargs.get('save_path', None)
        
        # Your analysis logic here
        # ...
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

### 2. Error Handling

```python
class RobustAnalyzer(Analyzer):
    def plot(self, results: Dict[str, pd.DataFrame], **kwargs):
        """Robust analyzer with error handling."""
        
        try:
            # Validate input
            self._validate_results(results)
            
            # Perform analysis
            analysis_data = self._perform_analysis(results)
            
            # Create plots
            self._create_plots(analysis_data, **kwargs)
            
        except KeyError as e:
            print(f"Missing required data: {e}")
            print("Available keys:", list(results.keys()))
        except Exception as e:
            print(f"Analysis failed: {e}")
    
    def _validate_results(self, results: Dict[str, pd.DataFrame]):
        """Validate that results contain required data."""
        required_keys = ['strategy_returns', 'benchmark_returns']
        
        for key in required_keys:
            if key not in results:
                raise KeyError(f"Missing required key: {key}")
            
            if not isinstance(results[key], pd.Series):
                raise TypeError(f"{key} must be a pandas Series")
```

### 3. Performance Optimization

```python
class EfficientAnalyzer(Analyzer):
    def __init__(self):
        self.cache = {}
    
    def plot(self, results: Dict[str, pd.DataFrame], **kwargs):
        """Efficient analyzer with caching."""
        
        # Check cache first
        cache_key = self._get_cache_key(results, kwargs)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Perform analysis
        analysis_result = self._perform_analysis(results, **kwargs)
        
        # Cache result
        self.cache[cache_key] = analysis_result
        
        return analysis_result
    
    def _get_cache_key(self, results: Dict[str, pd.DataFrame], kwargs: Dict) -> str:
        """Generate cache key for results and parameters."""
        # Create a hash of the results and parameters
        import hashlib
        
        # Simple hash of results length and kwargs
        data_str = f"{len(results)}-{sorted(kwargs.items())}"
        return hashlib.md5(data_str.encode()).hexdigest()
```

## Combining Analyzers

### Multi-Analyzer

```python
class MultiAnalyzer(Analyzer):
    """
    Combines multiple analyzers into a single comprehensive analysis.
    """
    
    def __init__(self, analyzers: List[Analyzer]):
        self.analyzers = analyzers
    
    def plot(self, results: Dict[str, pd.DataFrame], **kwargs):
        """Run all analyzers and display results."""
        
        for i, analyzer in enumerate(self.analyzers):
            print(f"\n=== Analysis {i+1}: {analyzer.__class__.__name__} ===")
            analyzer.plot(results, **kwargs)
```

### Usage Example

```python
# Create multiple analyzers
equity_analyzer = EquityDrawdownAnalyzer()
monte_carlo_analyzer = MonteCarloAnalyzer()
risk_analyzer = RiskMetricsAnalyzer()

# Combine them
multi_analyzer = MultiAnalyzer([
    equity_analyzer,
    monte_carlo_analyzer,
    risk_analyzer
])

# Run comprehensive analysis
multi_analyzer.plot(results)
```

## Testing Analyzers

### Unit Testing

```python
import pytest
import pandas as pd
import numpy as np

def test_analyzer_initialization():
    """Test analyzer initialization."""
    analyzer = MyCustomAnalyzer(param1=10, param2="test")
    assert analyzer.parameters['param1'] == 10
    assert analyzer.parameters['param2'] == "test"

def test_analyzer_plot():
    """Test analyzer plot method."""
    analyzer = MyCustomAnalyzer()
    
    # Mock results
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    strategy_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 100), index=dates)
    
    results = {
        'strategy_returns': strategy_returns,
        'benchmark_returns': benchmark_returns
    }
    
    # Should not raise an exception
    analyzer.plot(results)

def test_analyzer_error_handling():
    """Test analyzer error handling."""
    analyzer = MyCustomAnalyzer()
    
    # Invalid results
    invalid_results = {'invalid_key': pd.Series()}
    
    # Should handle gracefully
    with pytest.raises(KeyError):
        analyzer.plot(invalid_results)
```

## Next Steps

- Learn about [backtesting](backtester.md)
- Explore [strategies](strategies.md)
- Check out [performance analysis](analysis.md) 