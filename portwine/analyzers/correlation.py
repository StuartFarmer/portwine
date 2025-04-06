from portwine.analyzers import Analyzer
import matplotlib.pyplot as plt

class CorrelationAnalyzer(Analyzer):
    """
    Computes and plots correlation among the tickers' daily returns.

    Usage:
      1) correlation_dict = analyzer.analyze(results)
      2) analyzer.plot(results)

    'results' should be the dictionary from the backtester, containing:
        'tickers_returns': DataFrame of daily returns for each ticker
                           (columns = ticker symbols, index = dates)
    """

    def __init__(self, method='pearson'):
        """
        Parameters
        ----------
        method : str
            Correlation method (e.g. 'pearson', 'spearman', 'kendall').
        """
        self.method = method

    def analyze(self, results):
        """
        Generates a correlation matrix of the daily returns among all tickers.

        Parameters
        ----------
        results : dict
            {
              'tickers_returns': DataFrame of daily returns per ticker
              ...
            }

        Returns
        -------
        analysis_dict : dict
            {
              'correlation_matrix': DataFrame (square) of correlations
            }
        """
        tickers_returns = results.get('tickers_returns')
        if tickers_returns is None or tickers_returns.empty:
            print("Error: 'tickers_returns' missing or empty in results.")
            return None

        # Compute correlation
        corr_matrix = tickers_returns.corr(method=self.method)

        return {
            'correlation_matrix': corr_matrix
        }

    def plot(self, results):
        """
        Plots a heatmap of the correlation matrix.

        Parameters
        ----------
        results : dict
            The same dictionary used in 'analyze', containing 'tickers_returns'.
        """
        analysis_dict = self.analyze(results)
        if analysis_dict is None:
            print("No correlation data to plot.")
            return

        corr_matrix = analysis_dict['correlation_matrix']
        if corr_matrix.empty:
            print("Correlation matrix is empty. Nothing to plot.")
            return

        # Plot the correlation matrix as a heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr_matrix, aspect='auto')
        fig.colorbar(cax)

        # Set tick marks for each ticker
        tickers = corr_matrix.columns
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=45, ha='left')
        ax.set_yticklabels(tickers)

        ax.set_title("Correlation Matrix", pad=20)
        plt.tight_layout()
        plt.show()
