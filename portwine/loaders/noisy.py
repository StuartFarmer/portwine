import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple


class NoisyMarketDataLoader:
    """
    A market data loader that injects noise scaled by rolling volatility.

    This implementation adds noise to price returns, with the magnitude of noise
    proportional to the local volatility (measured as rolling standard deviation
    of returns). This ensures the noise adapts to different market regimes.

    Parameters
    ----------
    base_loader : object
        A base loader with load_ticker(ticker) and fetch_data(tickers) methods
    noise_multiplier : float, optional
        Base multiplier for the noise magnitude (default: 1.0)
    volatility_window : int, optional
        Window size in days for rolling volatility calculation (default: 21)
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
            self,
            base_loader: Any,
            noise_multiplier: float = 1.0,
            volatility_window: int = 21,
            seed: Optional[int] = None
    ):
        self.base_loader = base_loader
        self.noise_multiplier = noise_multiplier
        self.volatility_window = volatility_window
        self._original_data: Dict[str, pd.DataFrame] = {}  # Cache for original data

        if seed is not None:
            np.random.seed(seed)

    def get_original_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get original data for a ticker from cache or load it.

        Parameters
        ----------
        ticker : str
            Ticker symbol

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with OHLCV data or None if not available
        """
        if ticker in self._original_data:
            return self._original_data[ticker]

        df = self.base_loader.load_ticker(ticker)
        if df is not None:
            self._original_data[ticker] = df.sort_index()
        return df

    def calculate_rolling_volatility(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate rolling standard deviation of returns.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with OHLCV data

        Returns
        -------
        numpy.ndarray
            Array of rolling volatility values
        """
        # Calculate returns
        returns = df['close'].pct_change().values

        # Initialize volatility array
        n = len(returns)
        volatility = np.ones(n)  # Default to 1 for scaling

        # Calculate overall volatility for initial values
        overall_std = np.std(returns[1:])  # Skip the first NaN value
        if not np.isfinite(overall_std) or overall_std <= 0:
            overall_std = 0.01  # Fallback if volatility is zero or invalid

        # Fill initial window with overall volatility to avoid NaNs
        volatility[:self.volatility_window] = overall_std

        # Calculate rolling standard deviation
        for i in range(self.volatility_window, n):
            window_returns = returns[i - self.volatility_window + 1:i + 1]
            # Remove any NaN values
            window_returns = window_returns[~np.isnan(window_returns)]
            if len(window_returns) > 0:
                vol = np.std(window_returns)
                # Ensure we have positive volatility
                volatility[i] = vol if np.isfinite(vol) and vol > 0 else overall_std
            else:
                volatility[i] = overall_std

        return volatility

    def generate_noise(self, size: int, volatility_scaling: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate volatility-scaled noise for OHLC data.

        Parameters
        ----------
        size : int
            Number of days to generate noise for
        volatility_scaling : numpy.ndarray
            Array of volatility scaling factors

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing noise for open, high, low, and close
        """

        # Generate base zero-mean noise
        def generate_zero_mean_noise(size):
            raw_noise = np.random.normal(0, 1, size=size)
            if size > 1:
                # Ensure perfect zero mean to avoid drift
                raw_noise = raw_noise - np.mean(raw_noise)
            return raw_noise

        # Generate noise for each price component and scale by volatility
        noise_open = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier
        noise_high = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier
        noise_low = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier
        noise_close = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier

        return noise_open, noise_high, noise_low, noise_close

    def inject_noise(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-scaled noise to OHLC data while preserving high/low consistency.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        df : pandas.DataFrame
            DataFrame with OHLCV data

        Returns
        -------
        pandas.DataFrame
            Copy of input DataFrame with adaptive noise added to OHLC values
        """
        if df is None or len(df) < 2:
            return df.copy() if df is not None else None

        df_result = df.copy()

        # Compute volatility scaling factors
        volatility_scaling = self.calculate_rolling_volatility(df)

        # Extract original values for vectorized operations
        orig_open = df['open'].values
        orig_high = df['high'].values
        orig_low = df['low'].values
        orig_close = df['close'].values

        # Create arrays for new values
        new_open = np.empty_like(orig_open)
        new_high = np.empty_like(orig_high)
        new_low = np.empty_like(orig_low)
        new_close = np.empty_like(orig_close)

        # First day remains unchanged
        new_open[0] = orig_open[0]
        new_high[0] = orig_high[0]
        new_low[0] = orig_low[0]
        new_close[0] = orig_close[0]

        # Pre-generate all noise at once with volatility scaling
        n_days = len(df) - 1
        noise_open, noise_high, noise_low, noise_close = self.generate_noise(n_days, volatility_scaling[1:])

        # Process each day
        for i in range(1, len(df)):
            prev_orig_close = orig_close[i - 1]
            prev_new_close = new_close[i - 1]

            # Calculate returns relative to previous close
            r_open = (orig_open[i] / prev_orig_close) - 1
            r_high = (orig_high[i] / prev_orig_close) - 1
            r_low = (orig_low[i] / prev_orig_close) - 1
            r_close = (orig_close[i] / prev_orig_close) - 1

            # Add noise to returns and calculate new prices
            tentative_open = prev_new_close * (1 + r_open + noise_open[i - 1])
            tentative_high = prev_new_close * (1 + r_high + noise_high[i - 1])
            tentative_low = prev_new_close * (1 + r_low + noise_low[i - 1])
            tentative_close = prev_new_close * (1 + r_close + noise_close[i - 1])

            # Ensure high is the maximum and low is the minimum
            new_open[i] = tentative_open
            new_close[i] = tentative_close
            new_high[i] = max(tentative_open, tentative_high, tentative_low, tentative_close)
            new_low[i] = min(tentative_open, tentative_high, tentative_low, tentative_close)

        # Update the result DataFrame
        df_result['open'] = new_open
        df_result['high'] = new_high
        df_result['low'] = new_low
        df_result['close'] = new_close

        return df_result

    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load ticker data and inject volatility-scaled noise.

        Parameters
        ----------
        ticker : str
            Ticker symbol

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with noisy OHLCV data or None if not available
        """
        df = self.get_original_ticker_data(ticker)
        if df is None:
            return None
        return self.inject_noise(ticker, df)

    def fetch_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers and inject volatility-scaled noise.

        Parameters
        ----------
        tickers : list of str
            List of ticker symbols

        Returns
        -------
        dict
            Dictionary mapping ticker symbols to DataFrames with noisy data
        """
        result = {}
        for ticker in tickers:
            df_noisy = self.load_ticker(ticker)
            if df_noisy is not None:
                result[ticker] = df_noisy
        return result
