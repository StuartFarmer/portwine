import numpy as np
import matplotlib.pyplot as plt
import time
from portwine.analyzers.base import Analyzer
from portwine.backtester import Backtester
from portwine.backtester.core import DailyMarketCalendar
from portwine.data.stores.noisy import NoisyDataStore
from portwine.data.interface import DataInterface
from portwine.data.stores.adapter import MarketDataLoaderAdapter


class NoiseRobustnessAnalyzer(Analyzer):
    """
    Analyzes a strategy's robustness to market noise by running multiple backtest iterations
    with different levels of noise injected into the price data.

    This analyzer fits into the portwine framework and allows testing whether a strategy's
    performance is stable across variations in the market data or if it's overfitted.

    Parameters
    ----------
    base_loader : MarketDataLoader
        A base loader with the original market data
    noise_levels : list of float, optional
        List of noise multipliers to test (default: [0.5, 1.0, 1.5, 2.0])
    iterations_per_level : int, optional
        Number of backtest iterations to run per noise level (default: 20)
    volatility_window : int, optional
        Window for rolling volatility calculation (default: 21)
    calendar_name : str, optional
        Name of the market calendar to use (default: "NYSE")
    """

    def __init__(
            self,
            base_loader,
            noise_levels=None,
            iterations_per_level=20,
            volatility_window=21,
            calendar_name="NYSE"
    ):
        self.base_loader = base_loader
        self.noise_levels = noise_levels if noise_levels is not None else [0.5, 1.0, 1.5, 2.0]
        self.iterations_per_level = iterations_per_level
        self.volatility_window = volatility_window
        self.calendar_name = calendar_name
        self.results = None

    def analyze(self, strategy, benchmark=None, start_date=None,
                shift_signals=True, require_all_history=False, verbose=True, n_jobs=-1):
        """
        Run multiple backtest iterations at each noise level to analyze strategy robustness.
        Uses joblib for parallel processing to speed up execution.

        Parameters
        ----------
        strategy : StrategyBase
            The strategy to test for robustness
        benchmark : None, str, or callable, optional
            Benchmark specification (see Backtester documentation)
        start_date : None, str, or datetime, optional
            The earliest date to start the backtest
        shift_signals : bool, optional
            Whether to shift signals by 1 day to avoid lookahead bias (default: True)
        require_all_history : bool, optional
            Whether to wait for all tickers to have data before starting (default: False)
        verbose : bool, optional
            Whether to show progress information (default: True)
        n_jobs : int, optional
            Number of parallel jobs to run. -1 means using all processors (default: -1)

        Returns
        -------
        dict
            Dictionary containing robust analysis results for each noise level
        """
        from joblib import Parallel, delayed

        # Try to import joblib_progress, fall back to tqdm if not available
        try:
            from joblib_progress import joblib_progress
            has_joblib_progress = True
        except ImportError:
            has_joblib_progress = False
            if verbose:
                print("Note: Install joblib_progress package for better progress tracking")
                from tqdm.auto import tqdm

        # Function to run a single backtest iteration
        def run_single_backtest(noise_level, iteration, seed=None):
            # Create a unique seed if not provided
            if seed is None:
                seed = int(time.time() * 1000) % 10000 + iteration

            # Create an adapter to make MarketDataLoader compatible with DataStore
            loader_adapter = MarketDataLoaderAdapter(self.base_loader)

            # Create noise store with current noise level and seed
            noisy_store = NoisyDataStore(
                base_store=loader_adapter,
                noise_multiplier=noise_level,
                volatility_window=self.volatility_window,
                seed=seed
            )

            # Create a data interface from the noisy store
            noisy_data_interface = DataInterface(noisy_store)

            # Create a market calendar
            calendar = DailyMarketCalendar(self.calendar_name)

            # Create a new backtester with the noisy data interface
            backtester = Backtester(data=noisy_data_interface, calendar=calendar)

            # Run backtest with noisy data
            backtest_result = backtester.run_backtest(
                strategy=strategy,
                benchmark=benchmark,
                start_date=start_date
            )

            return backtest_result

        # Prepare all backtest tasks
        all_tasks = []

        for noise_level in self.noise_levels:
            for iteration in range(self.iterations_per_level):
                all_tasks.append((noise_level, iteration))

        # Total number of tasks
        total_tasks = len(all_tasks)

        if verbose:
            print(
                f"Running {self.iterations_per_level} iterations for each of {len(self.noise_levels)} noise levels...")
            print(f"Total backtest runs: {total_tasks}")

        # Run tasks using joblib
        if has_joblib_progress and verbose:
            # Use joblib_progress for tracking progress
            with joblib_progress("Running noise robustness backtests...", total=total_tasks):
                results = Parallel(n_jobs=n_jobs)(
                    delayed(run_single_backtest)(noise_level, iteration)
                    for noise_level, iteration in all_tasks
                )
        else:
            # Use tqdm or no progress tracking
            if verbose and not has_joblib_progress:
                tasks_iter = tqdm(all_tasks, desc="Running backtests")
            else:
                tasks_iter = all_tasks

            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_backtest)(noise_level, iteration)
                for noise_level, iteration in tasks_iter
            )

        # Organize results by noise level
        results_per_level = {noise_level: [] for noise_level in self.noise_levels}

        for task, result in zip(all_tasks, results):
            noise_level, _ = task
            if result is not None:
                results_per_level[noise_level].append(result)

        # Count results per level
        if verbose:
            print("\nResults summary:")
            for noise_level in self.noise_levels:
                count = len(results_per_level[noise_level])
                print(f"  Noise level {noise_level}: {count}/{self.iterations_per_level} successful runs")

        self.results = results_per_level
        return self.compute_analysis(results_per_level)

    def compute_analysis(self, results_per_level):
        """
        Compute statistical analysis of backtest results across noise levels.

        Parameters
        ----------
        results_per_level : dict
            Dictionary mapping noise levels to lists of backtest results

        Returns
        -------
        dict
            Dictionary with analysis statistics for each noise level
        """
        analysis = {}

        for noise_level, level_results in results_per_level.items():
            if not level_results:
                analysis[noise_level] = {"error": "No valid backtest results"}
                continue

            # Extract performance metrics from each backtest result
            metrics = []

            for result in level_results:
                # Extract strategy returns
                strategy_returns = result.get('strategy_returns')
                if strategy_returns is None or strategy_returns.empty:
                    continue

                # Calculate key metrics
                total_return = (1 + strategy_returns).prod() - 1
                annual_vol = strategy_returns.std() * np.sqrt(252)  # Annualize with 252 trading days
                mean_return = strategy_returns.mean()
                sharpe = (mean_return / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() > 0 else 0

                # Calculate max drawdown
                equity_curve = (1 + strategy_returns).cumprod()
                rolling_max = equity_curve.cummax()
                drawdown = (equity_curve - rolling_max) / rolling_max
                max_drawdown = drawdown.min()

                metrics.append({
                    'total_return': total_return,
                    'mean_return': mean_return,
                    'annual_vol': annual_vol,
                    'sharpe': sharpe,
                    'max_drawdown': max_drawdown
                })

            # Compute statistics across all iterations for this noise level
            if not metrics:
                analysis[noise_level] = {"error": "Failed to calculate metrics"}
                continue

            analysis[noise_level] = {
                'iterations': len(metrics),
                'total_return': {
                    'mean': np.mean([m['total_return'] for m in metrics]),
                    'std': np.std([m['total_return'] for m in metrics]),
                    'min': np.min([m['total_return'] for m in metrics]),
                    'max': np.max([m['total_return'] for m in metrics]),
                    'median': np.median([m['total_return'] for m in metrics]),
                    'values': [m['total_return'] for m in metrics]
                },
                'mean_return': {
                    'mean': np.mean([m['mean_return'] for m in metrics]),
                    'std': np.std([m['mean_return'] for m in metrics]),
                    'min': np.min([m['mean_return'] for m in metrics]),
                    'max': np.max([m['mean_return'] for m in metrics]),
                    'median': np.median([m['mean_return'] for m in metrics]),
                    'values': [m['mean_return'] for m in metrics]
                },
                'sharpe': {
                    'mean': np.mean([m['sharpe'] for m in metrics]),
                    'std': np.std([m['sharpe'] for m in metrics]),
                    'min': np.min([m['sharpe'] for m in metrics]),
                    'max': np.max([m['sharpe'] for m in metrics]),
                    'median': np.median([m['sharpe'] for m in metrics]),
                    'values': [m['sharpe'] for m in metrics]
                },
                'max_drawdown': {
                    'mean': np.mean([m['max_drawdown'] for m in metrics]),
                    'std': np.std([m['max_drawdown'] for m in metrics]),
                    'min': np.min([m['max_drawdown'] for m in metrics]),
                    'max': np.max([m['max_drawdown'] for m in metrics]),
                    'median': np.median([m['max_drawdown'] for m in metrics]),
                    'values': [m['max_drawdown'] for m in metrics]
                }
            }

        return analysis

    def plot(self, analysis=None, figsize=(15, 16)):
        """
        Plot a combined figure with three rows:
        1. Averaged equity curves for all noise levels
        2. Three metrics (total return, Sharpe ratio, max drawdown) vs noise
        3. Exponential decay curve fit (if enough data points)

        Parameters
        ----------
        analysis : dict, optional
            Analysis results from compute_analysis (if None, uses self.results)
        figsize : tuple, optional
            Figure size for the combined plot
        """
        if analysis is None:
            if self.results is None:
                raise ValueError("No analysis results. Run analyze() first.")
            analysis = self.compute_analysis(self.results)

        if self.results is None:
            raise ValueError("Raw backtest results not available. Cannot plot equity curves.")

        # Set up the combined figure with subplots
        fig = plt.figure(figsize=figsize)

        # Define subplot grid: 3 rows, with the middle row having 3 columns
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1])

        # Row 1: Equity Curves
        ax_equity = fig.add_subplot(gs[0, :])

        # Line colors based on noise levels (from blue to red)
        noise_levels = sorted(self.results.keys())
        num_levels = len(noise_levels)

        # Generate a color map from blue to red
        cmap = plt.cm.get_cmap('coolwarm', num_levels)
        colors = [cmap(i) for i in range(num_levels)]

        # Plot averaged equity curves for each noise level
        for i, noise_level in enumerate(noise_levels):
            backtest_results = self.results[noise_level]
            if not backtest_results:
                continue

            # Calculate average returns across all iterations
            avg_returns = None
            common_index = None

            # First find a common date index (intersection of all backtest periods)
            for result in backtest_results:
                strategy_returns = result.get('strategy_returns')
                if strategy_returns is None or strategy_returns.empty:
                    continue

                if common_index is None:
                    common_index = strategy_returns.index
                else:
                    common_index = common_index.intersection(strategy_returns.index)

            if common_index is None or len(common_index) == 0:
                continue

            # Sum returns for the common period, then average
            for result in backtest_results:
                strategy_returns = result.get('strategy_returns')
                if strategy_returns is None or strategy_returns.empty:
                    continue

                # Align returns to common index
                aligned_returns = strategy_returns.reindex(common_index)

                if avg_returns is None:
                    avg_returns = aligned_returns
                else:
                    avg_returns = avg_returns.add(aligned_returns)

            if avg_returns is not None and len(backtest_results) > 0:
                # Calculate average
                avg_returns = avg_returns / len(backtest_results)

                # Create equity curve
                equity_curve = (1 + avg_returns).cumprod()

                # Plot with appropriate color and label
                ax_equity.plot(equity_curve.index, equity_curve.values,
                               color=colors[i], label=f"Noise {noise_level}")

        ax_equity.set_title("Average Equity Curves by Noise Level")
        ax_equity.set_ylabel("Equity (Starting at 1.0)")
        ax_equity.grid(True, alpha=0.3)
        ax_equity.legend()

        # Row 2: Individual Metrics
        metrics = ['mean_return', 'sharpe', 'max_drawdown']
        metric_labels = {
            'mean_return': 'Mean Daily Return',
            'sharpe': 'Sharpe Ratio',
            'max_drawdown': 'Maximum Drawdown'
        }

        # Create subplots for each metric
        axes_metrics = [fig.add_subplot(gs[1, i]) for i in range(3)]

        # For each metric, create a separate line plot
        for i, metric in enumerate(metrics):
            ax = axes_metrics[i]

            # Collect data for this metric
            means = []
            noise_levels_metric = []

            for noise_level in sorted(analysis.keys()):
                if "error" in analysis[noise_level]:
                    continue

                if metric in analysis[noise_level]:
                    means.append(analysis[noise_level][metric]['mean'])
                    noise_levels_metric.append(noise_level)

            # Plot the line
            ax.plot(noise_levels_metric, means, 'o-', linewidth=2, markersize=8)

            # Set labels and title
            ax.set_xlabel('Noise Level')

            # Format y-axis based on metric type
            if metric == 'mean_return':
                ax.set_ylabel('Daily Return')
                # For mean returns, use scientific notation for very small values
                min_val = min(means) if means else 0
                if abs(min_val) < 0.0001:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(
                        lambda y, _: f'{y:.2e}' if abs(y) < 0.0001 else f'{y:.2%}'
                    ))
                else:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
            elif metric == 'max_drawdown':
                ax.set_ylabel('Drawdown')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
            else:  # sharpe
                ax.set_ylabel('Ratio')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

            ax.set_title(metric_labels.get(metric, metric))
            ax.grid(True, alpha=0.3)

            # # Add value labels above each point
            # for j, (x, y) in enumerate(zip(noise_levels_metric, means)):
            #     if metric == 'mean_return':
            #         if abs(y) < 0.0001:
            #             label = f'{y:.2e}'
            #         else:
            #             label = f'{y:.2%}'
            #     elif metric == 'max_drawdown':
            #         label = f'{y:.2%}'
            #     else:  # sharpe
            #         label = f'{y:.2f}'

            #     ax.annotate(label,
            #                (x, y),
            #                textcoords="offset points",
            #                xytext=(0, 10),
            #                ha='center')

        # Row 3: Curve Fit for Sharpe Ratio
        ax_fit = fig.add_subplot(gs[2, :])

        # Plot curve fit if there are enough data points
        if len(analysis) >= 3:
            from scipy.optimize import curve_fit
            import numpy as np

            # Prepare data for curve fitting using Sharpe ratio instead of mean returns
            noise_levels_array = np.array(sorted(analysis.keys()))
            sharpes = np.array([analysis[level]['sharpe']['mean'] for level in noise_levels_array])

            # Define exponential decay function: f(x) = a * exp(-b * x) + c
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c

            try:
                # Try to fit an exponential decay curve
                params, _ = curve_fit(exp_decay, noise_levels_array, sharpes,
                                      p0=[sharpes[0], 0.5, 0],
                                      bounds=([0, 0, -1], [np.inf, np.inf, 1]))

                a, b, c = params

                # Create a smooth curve for plotting
                x_smooth = np.linspace(min(noise_levels_array), max(noise_levels_array), 100)
                y_smooth = exp_decay(x_smooth, a, b, c)

                # Calculate R-squared
                residuals = sharpes - exp_decay(noise_levels_array, a, b, c)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((sharpes - np.mean(sharpes)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # Format equation for display
                if a < 0.001:
                    a_fmt = f"{a:.2e}"
                else:
                    a_fmt = f"{a:.4f}"

                if c < 0.001 and c > -0.001:
                    c_fmt = f"{c:.2e}"
                else:
                    c_fmt = f"{c:.4f}"

                # Plot the data points and curve fit
                ax_fit.scatter(noise_levels_array, sharpes, color='blue', label='Actual Sharpe Ratios')
                ax_fit.plot(x_smooth, y_smooth, 'r-',
                            label=f'Exponential Fit: {a_fmt}*exp(-{b:.3f}*x) + {c_fmt}')

                ax_fit.set_xlabel('Noise Level')
                ax_fit.set_ylabel('Sharpe Ratio')
                ax_fit.set_title(f'Exponential Decay Fit of Sharpe Ratio vs Noise (R² = {r_squared:.3f})')
                ax_fit.grid(True, alpha=0.3)
                ax_fit.legend()

                # Calculate and mark critical noise level (where Sharpe drops to 50% of baseline)
                if a > 0:
                    baseline_sharpe = sharpes[0]
                    half_sharpe = baseline_sharpe * 0.5

                    try:
                        if half_sharpe > c:  # Standard case - target is above asymptote
                            critical_noise = -np.log((half_sharpe - c) / a) / b

                            # Add critical level line if within reasonable range
                            if 0 < critical_noise < max(noise_levels_array) * 2:
                                ax_fit.axvline(x=critical_noise, color='g', linestyle='--',
                                               label=f'50% Sharpe Level: {critical_noise:.2f}')
                                ax_fit.axhline(y=half_sharpe, color='g', linestyle=':')
                                ax_fit.legend()
                            else:
                                # Critical level exists but is outside our plot range
                                ax_fit.text(0.05, 0.95, f"50% Sharpe at noise = {critical_noise:.2f} (outside range)",
                                            transform=ax_fit.transAxes, fontsize=9, va='top', color='darkred')
                        else:  # Special case - half sharpe is below asymptote
                            ax_fit.text(0.05, 0.95, "Note: 50% Sharpe level unreachable (below asymptote)",
                                        transform=ax_fit.transAxes, fontsize=9, va='top', color='darkred')
                    except:
                        # Handle any calculation errors
                        ax_fit.text(0.05, 0.95, "Could not calculate 50% Sharpe noise level",
                                    transform=ax_fit.transAxes, fontsize=9, va='top', color='darkred')

            except (RuntimeError, ValueError):
                # Just show scatter plot of sharpe vs noise if curve fitting fails
                ax_fit.scatter(noise_levels_array, sharpes, color='blue')
                ax_fit.set_xlabel('Noise Level')
                ax_fit.set_ylabel('Sharpe Ratio')
                ax_fit.set_title('Sharpe Ratio vs Noise Level')
                ax_fit.grid(True, alpha=0.3)

        else:
            # Not enough data points for curve fitting
            ax_fit.text(0.5, 0.5, "Not enough data points for curve fitting\n(need at least 3 noise levels)",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax_fit.transAxes)

        plt.tight_layout()
        plt.show()

    def generate_report(self, analysis=None):
        """
        Generate a text report summarizing the noise robustness analysis.

        Parameters
        ----------
        analysis : dict, optional
            Analysis results from compute_analysis (if None, uses self.results)

        Returns
        -------
        str
            Text report of the analysis
        """
        if analysis is None:
            if self.results is None:
                raise ValueError("No analysis results. Run analyze() first.")
            analysis = self.compute_analysis(self.results)

        lines = []
        lines.append("\n==== NOISE ROBUSTNESS ANALYSIS ====\n")

        for noise_level in sorted(analysis.keys()):
            level_data = analysis[noise_level]

            if "error" in level_data:
                lines.append(f"Noise Level {noise_level}: {level_data['error']}")
                continue

            lines.append(f"\n--- Noise Level: {noise_level} ({level_data['iterations']} iterations) ---")

            # Format mean return
            mr_mean = level_data['mean_return']['mean']
            mr_std = level_data['mean_return']['std']
            lines.append(f"Mean Daily Return: {mr_mean:.4%} ± {mr_std:.4%}")
            lines.append(f"  Range: [{level_data['mean_return']['min']:.4%}, "
                         f"{level_data['mean_return']['max']:.4%}]")

            # Format Sharpe ratio
            sh_mean = level_data['sharpe']['mean']
            sh_std = level_data['sharpe']['std']
            lines.append(f"Sharpe Ratio: {sh_mean:.2f} ± {sh_std:.2f}")
            lines.append(f"  Range: [{level_data['sharpe']['min']:.2f}, "
                         f"{level_data['sharpe']['max']:.2f}]")

            # Format maximum drawdown
            dd_mean = level_data['max_drawdown']['mean']
            dd_std = level_data['max_drawdown']['std']
            lines.append(f"Max Drawdown: {dd_mean:.2%} ± {dd_std:.2%}")
            lines.append(f"  Range: [{level_data['max_drawdown']['min']:.2%}, "
                         f"{level_data['max_drawdown']['max']:.2%}]")

        # Add robustness assessment with curve fitting
        if len(analysis) >= 3:  # Need at least 3 points for meaningful curve fitting
            from scipy.optimize import curve_fit
            import numpy as np

            # Collect x (noise levels) and y (mean returns) for curve fitting
            noise_levels = sorted(analysis.keys())
            returns = [analysis[level]['mean_return']['mean'] for level in noise_levels]

            # Define exponential decay function: f(x) = a * exp(-b * x) + c
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c

            try:
                # Try to fit an exponential decay curve
                params, _ = curve_fit(exp_decay, noise_levels, returns,
                                      p0=[returns[0], 0.5, 0],
                                      bounds=([0, 0, -1], [np.inf, np.inf, 1]))

                a, b, c = params

                # Calculate R-squared to measure goodness of fit
                residuals = np.array(returns) - exp_decay(np.array(noise_levels), a, b, c)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((np.array(returns) - np.mean(returns)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # Assess based on rate of decay (b parameter)
                lines.append("\n--- Robustness Assessment (Curve Fit) ---")
                lines.append(f"Model: Mean Return = {a:.6f} * exp(-{b:.4f} * noise) + {c:.6f}")
                lines.append(f"R-squared: {r_squared:.4f}")

                # Interpret decay rate
                if b < 0.2:
                    lines.append("Verdict: Highly Robust - Very slow decay rate with increasing noise")
                elif b < 0.5:
                    lines.append("Verdict: Robust - Moderate decay rate with increasing noise")
                elif b < 1.0:
                    lines.append("Verdict: Moderately Robust - Noticeable decay with increasing noise")
                elif b < 2.0:
                    lines.append("Verdict: Somewhat Fragile - Rapid decay with increasing noise")
                else:
                    lines.append("Verdict: Very Fragile - Extremely rapid decay with increasing noise")

                # Calculate noise level at which performance drops below 50% of baseline
                if a > 0 and b > 0:
                    half_performance = returns[0] * 0.5
                    critical_noise = -np.log((half_performance - c) / a) / b if a != 0 else float('inf')

                    # Only report if the critical noise level is within a reasonable range
                    if 0 < critical_noise < max(noise_levels) * 2:
                        lines.append(
                            f"Critical Noise Level: {critical_noise:.2f} (where mean return drops to 50% of baseline)")

            except (RuntimeError, ValueError) as e:
                # Fallback if curve fitting fails
                lines.append("\n--- Robustness Assessment (Linear) ---")

                # Calculate linear slope between first and last point
                baseline = returns[0]
                highest_noise = returns[-1]
                slope = (highest_noise - baseline) / (noise_levels[-1] - noise_levels[0])

                # Normalize slope to express as percent change per unit of noise
                norm_slope = slope / baseline if baseline != 0 else 0

                lines.append(
                    f"Normalized Slope: {norm_slope:.4f} (fraction of baseline performance lost per unit of noise)")

                if norm_slope > -0.05:
                    lines.append("Verdict: Highly Robust - Minimal degradation with increasing noise")
                elif norm_slope > -0.15:
                    lines.append("Verdict: Robust - Gradual degradation with increasing noise")
                elif norm_slope > -0.3:
                    lines.append("Verdict: Moderately Robust - Noticeable degradation with increasing noise")
                elif norm_slope > -0.5:
                    lines.append("Verdict: Somewhat Fragile - Significant degradation with increasing noise")
                else:
                    lines.append("Verdict: Very Fragile - Severe degradation with increasing noise")

        return "\n".join(lines)
