"""
Walk Forward Optimization analysis tools.

This module provides visualization and analytical tools for examining
walk-forward optimization results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class WalkForwardAnalyzer:
    """
    Analyzer for walk-forward optimization results.

    This analyzer provides tools to visualize and analyze the performance of
    walk-forward optimization results, including equity curves, drawdowns,
    parameter evolution, and performance metrics.
    """

    def analyze(self, results):
        """
        Analyze walk-forward optimization results.

        Parameters
        ----------
        results : dict
            Results from a WalkForwardOptimizer.optimize() call.

        Returns
        -------
        dict
            Analysis metrics
        """
        # Extract test periods data
        test_periods = results.get('test_periods', {})
        if not test_periods:
            return {"error": "No test periods found in results"}

        # Combine all test period returns for overall analysis
        all_returns = []
        for period_key, period_data in test_periods.items():
            if ('results' in period_data and period_data['results'] and
                    'strategy_returns' in period_data['results']):
                period_returns = period_data['results']['strategy_returns']
                if not period_returns.empty:
                    all_returns.append(period_returns)

        # Only concatenate if we have returns to work with
        combined_returns = pd.Series(dtype=float)
        if all_returns:
            combined_returns = pd.concat(all_returns)

        # Sort combined returns by date
        if not combined_returns.empty:
            combined_returns = combined_returns.sort_index()

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(combined_returns)

        # Extract parameter evolution data
        param_evolution = self._extract_parameter_evolution(test_periods)

        # Extract period transitions
        period_transitions = self._extract_period_transitions(test_periods)

        # Store analysis results
        analysis_results = {
            'combined_returns': combined_returns,
            'metrics': metrics,
            'param_evolution': param_evolution,
            'period_transitions': period_transitions
        }

        return analysis_results

    def _extract_period_transitions(self, test_periods):
        """Extract transition dates between test periods."""
        transitions = []

        # Convert to list and sort by start date
        periods_list = [(period_key, period_data) for period_key, period_data in test_periods.items()]
        sorted_periods = sorted(periods_list, key=lambda x: x[1]['start_date'])

        # Extract start and end dates
        for _, period_data in sorted_periods:
            transitions.append(period_data['start_date'])
            transitions.append(period_data['end_date'])

        # Remove duplicates and sort
        return sorted(list(set(transitions)))

    def _calculate_performance_metrics(self, returns):
        """Calculate comprehensive performance metrics."""
        if returns.empty or len(returns) < 10:
            return {}

        # Annual factor for daily returns
        ann_factor = 252.0

        # Basic metrics
        mean_return = returns.mean() * ann_factor
        std_return = returns.std() * np.sqrt(ann_factor)
        sharpe = mean_return / std_return if std_return > 1e-8 else 0.0
        total_return = (1 + returns).prod() - 1
        win_rate = (returns > 0).mean()

        # Drawdown calculations
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1)
        max_drawdown = drawdown.min()

        # Calmar ratio (annualized return / max drawdown)
        calmar = (mean_return / abs(max_drawdown)) if max_drawdown < 0 else float('inf')

        # Recovery metrics
        underwater_periods = (drawdown < 0).sum()
        underwater_pct = underwater_periods / len(returns) if len(returns) > 0 else 0

        # Monthly returns
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_win_rate = (monthly_returns > 0).mean()

        # Return all metrics
        return {
            'ann_return': mean_return,
            'ann_volatility': std_return,
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'underwater_pct': underwater_pct,
            'monthly_win_rate': monthly_win_rate
        }

    def _extract_parameter_evolution(self, test_periods):
        """Extract parameter evolution data across test periods."""
        if not test_periods:
            return {}

        # Convert test periods to a DataFrame for easier analysis
        periods_data = []
        for period_key, period_data in test_periods.items():
            row = {
                'period': period_key,
                'start_date': period_data['start_date'],
                'end_date': period_data['end_date'],
                'step': period_data.get('step', None)  # Add step information
            }

            # Extract parameters (excluding tickers which would be lists)
            if 'best_params' in period_data:
                for param, value in period_data['best_params'].items():
                    if param != 'tickers':
                        row[f'param_{param}'] = value

            # Add basic metrics if available
            if ('results' in period_data and period_data['results'] and
                    'strategy_returns' in period_data['results']):
                returns = period_data['results']['strategy_returns']
                if len(returns) >= 5:  # Minimum for meaningful metrics
                    row['sharpe'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
                    row['total_return'] = (1 + returns).prod() - 1

            periods_data.append(row)

        if periods_data:
            df = pd.DataFrame(periods_data)
            # Sort by step if available, otherwise by period start date
            if 'step' in df.columns and df['step'].notna().all():
                df = df.sort_values('step')
            else:
                df = df.sort_values('start_date')
            return df.set_index('period')
        else:
            return {}

    def plot(self, results, figsize=(15, 12)):
        """
        Plot walk-forward optimization results.

        Parameters
        ----------
        results : dict
            Results from WalkForwardOptimizer.optimize()
        figsize : tuple
            Figure size (width, height)
        """
        # First analyze the results
        analysis = self.analyze(results)

        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return

        combined_returns = analysis['combined_returns']
        param_evolution = analysis['param_evolution']
        metrics = analysis['metrics']
        period_transitions = analysis['period_transitions']

        if combined_returns.empty:
            print("No returns data to plot.")
            return

        # Extract configuration
        optimizer_type = results.get('optimizer_type', 'walk_forward')
        anchored = results.get('anchored', False)

        # Create the figure with 3 rows
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

        # Add vertical lines for period transitions (pastel yellow)
        transition_color = 'lightgrey'  # Pastel yellow

        ax1 = fig.add_subplot(gs[0])
        equity_curve = (1 + combined_returns).cumprod()
        for date in period_transitions:
            if date in equity_curve.index:
                ax1.axvline(date, color=transition_color, linestyle='--', alpha=0.7)

        # Equity curve with transition markers
        ax1.plot(equity_curve.index, equity_curve.values, label='Strategy')

        # Add benchmark if available in any test period
        all_benchmark_returns = []
        for period_key, period_data in results['test_periods'].items():
            if ('results' in period_data and period_data['results'] and
                    'benchmark_returns' in period_data['results']):
                period_benchmark = period_data['results']['benchmark_returns']
                if not period_benchmark.empty:
                    all_benchmark_returns.append(period_benchmark)

        benchmark_returns = pd.Series(dtype=float)
        if all_benchmark_returns:
            benchmark_returns = pd.concat(all_benchmark_returns)

        if not benchmark_returns.empty:
            benchmark_returns = benchmark_returns.sort_index()
            benchmark_equity = (1 + benchmark_returns).cumprod()
            ax1.plot(benchmark_equity.index, benchmark_equity.values, label='Benchmark', alpha=0.7)

        title_type = f"{'Anchored' if anchored else 'Sliding'} Walk-Forward" if optimizer_type == 'walk_forward' else optimizer_type.replace(
            '_', ' ').title()
        ax1.set_title(f'Out-of-Sample Equity Curve ({title_type})')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()

        ax2 = fig.add_subplot(gs[1])
        drawdown = (equity_curve / equity_curve.cummax() - 1)

        # Add vertical lines for period transitions in drawdown chart
        for date in period_transitions:
            if date in drawdown.index:
                ax2.axvline(date, color=transition_color, linestyle='--', alpha=0.7)

        # Drawdowns (full width)
        ax2.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color='red')
        ax2.set_title('Out-of-Sample Drawdown')
        ax2.set_ylabel('Drawdown (%)')

        # Parameter evolution at the bottom (full width) with simple numbered x-axis
        if isinstance(param_evolution, pd.DataFrame) and not param_evolution.empty:
            ax3 = fig.add_subplot(gs[2])

            # Find parameter columns
            param_cols = [col for col in param_evolution.columns if col.startswith('param_')]

            # Create x-axis positions (just sequential numbers)
            x_values = range(len(param_evolution))

            # Add vertical lines for parameter transitions
            # We'll place them evenly across the parameter chart where possible
            for i in range(len(param_evolution) - 1):  # Don't add a line after the last period
                ax3.axvline(i + 0.5, color=transition_color, linestyle='--', alpha=0.7)

            # Plot up to 4 parameters for readability
            for col in param_cols[:4]:
                param_name = col.replace('param_', '')
                ax3.plot(x_values, param_evolution[col], 'o-', label=param_name)

            ax3.set_title('Parameter Evolution')
            ax3.set_xlabel('Walk-Forward Step')
            ax3.set_ylabel('Parameter Value')

            # Simple numbered x-axis
            ax3.set_xticks(x_values)
            ax3.set_xticklabels([str(i + 1) for i in x_values])

            ax3.legend()

        plt.tight_layout()
        plt.show()

    def _print_summary(self, metrics):
        """Print a summary of performance metrics."""
        if not metrics:
            print("No performance metrics available.")
            return

        print("\n=== Performance Summary ===")
        formatted_metrics = {
            'Annual Return': f"{metrics.get('ann_return', 0):.2%}",
            'Annual Volatility': f"{metrics.get('ann_volatility', 0):.2%}",
            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'Total Return': f"{metrics.get('total_return', 0):.2%}",
            'Win Rate': f"{metrics.get('win_rate', 0):.2%}",
            'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
            'Calmar Ratio': f"{metrics.get('calmar_ratio', 0):.2f}",
            'Time Underwater': f"{metrics.get('underwater_pct', 0):.2%}",
            'Monthly Win Rate': f"{metrics.get('monthly_win_rate', 0):.2%}"
        }

        # Print in a nice format
        max_key_len = max(len(k) for k in formatted_metrics.keys())
        for key, value in formatted_metrics.items():
            print(f"{key:{max_key_len}}: {value}")

    def plot_parameter_surface(self, results, param1, param2, metric='sharpe', figsize=(12, 8)):
        """
        Plot a 3D surface showing the relationship between two parameters and a performance metric.

        Parameters
        ----------
        results : dict
            Results from WalkForwardOptimizer.optimize()
        param1 : str
            First parameter name
        param2 : str
            Second parameter name
        metric : str
            Performance metric to visualize
        figsize : tuple
            Figure size (width, height)
        """
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("3D plotting requires mpl_toolkits.mplot3d")
            return

        # Get parameter data
        param_evolution = self._extract_parameter_evolution(results.get('test_periods', {}))

        if isinstance(param_evolution, pd.DataFrame) and not param_evolution.empty:
            # Check if we have the required columns
            param1_col = f"param_{param1}"
            param2_col = f"param_{param2}"

            if param1_col not in param_evolution.columns:
                print(f"Parameter {param1} not found in results")
                return

            if param2_col not in param_evolution.columns:
                print(f"Parameter {param2} not found in results")
                return

            if metric not in param_evolution.columns:
                print(f"Metric {metric} not found in results")
                return

            # Create the 3D plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            # Plot the surface
            ax.scatter(param_evolution[param1_col],
                       param_evolution[param2_col],
                       param_evolution[metric],
                       c=param_evolution[metric],
                       cmap='viridis',
                       s=100,
                       alpha=0.8)

            # Connect points in sequence to show evolution
            if 'step' in param_evolution.columns:
                sorted_df = param_evolution.sort_values('step')
                ax.plot(sorted_df[param1_col], sorted_df[param2_col], sorted_df[metric], 'r-', alpha=0.6)

            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_zlabel(metric)
            ax.set_title(f'Parameter Surface: {param1} vs {param2} vs {metric}')

            plt.tight_layout()
            plt.show()
        else:
            print("No parameter evolution data available")