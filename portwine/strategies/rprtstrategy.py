
import numpy as np
from portwine.strategies.base import StrategyBase

class RPRTStrategy(StrategyBase):
    """
    Reweighted Price Relative Tracking (RPRT) Strategy

    This strategy implements the RPRT algorithm from the paper:
    "Reweighted Price Relative Tracking System for Automatic Portfolio Optimization"

    The strategy:
    1. Calculates price relatives for each asset
    2. Uses a reweighted price relative prediction system that adapts to each asset's performance
    3. Optimizes portfolio weights using a tracking system with a generalized increasing factor
    """

    def __init__(self, tickers, window_size=5, theta=0.8, epsilon=1.01):
        """
        Parameters
        ----------
        tickers : list
            List of ticker symbols to consider for investment
        window_size : int
            Window size for SMA calculations
        theta : float
            Mixing parameter for reweighted price relative prediction
        epsilon : float
            Expected profiting level for portfolio optimization
        """
        super().__init__(tickers)
        self.window_size = window_size
        self.theta = theta
        self.epsilon = epsilon

        # Initialize internal state for RPRT
        self.price_relatives_history = []
        self.phi_hat = None  # Reweighted price relative prediction
        self.current_portfolio = None
        self.price_history = {ticker: [] for ticker in tickers}
        self.has_initialized = False

    def step(self, current_date, daily_data):
        """
        Process daily data and determine allocations using RPRT

        Parameters
        ----------
        current_date : datetime
            Current backtesting date
        daily_data : dict
            Dictionary with price data for each ticker

        Returns
        -------
        dict
            Portfolio weights for each ticker
        """
        # Get today's close prices
        today_prices = {}
        for ticker in self.tickers:
            if daily_data.get(ticker) is not None:
                price = daily_data[ticker].get('close')
                if price is not None:
                    today_prices[ticker] = price
                    # Update price history
                    self.price_history[ticker].append(price)
                elif len(self.price_history[ticker]) > 0:
                    # Forward fill missing prices
                    price = self.price_history[ticker][-1]
                    today_prices[ticker] = price
                    self.price_history[ticker].append(price)
            elif len(self.price_history[ticker]) > 0:
                # Forward fill missing prices
                price = self.price_history[ticker][-1]
                today_prices[ticker] = price
                self.price_history[ticker].append(price)

        # Need at least two days of data to calculate price relatives
        if not self.has_initialized:
            # On first day, initialize with equal weights
            if len(today_prices) > 0:
                weights = {ticker: 1.0 / len(today_prices) if ticker in today_prices else 0.0
                           for ticker in self.tickers}
                self.current_portfolio = np.array([weights.get(ticker, 0.0) for ticker in self.tickers])
                self.has_initialized = True
                return weights
            else:
                return {ticker: 0.0 for ticker in self.tickers}

        # Calculate price relatives (today's price / yesterday's price)
        yesterday_prices = {}
        for ticker in self.tickers:
            if len(self.price_history[ticker]) >= 2:
                yesterday_prices[ticker] = self.price_history[ticker][-2]

        price_relatives = []
        for ticker in self.tickers:
            if ticker in today_prices and ticker in yesterday_prices and yesterday_prices[ticker] > 0:
                price_relative = today_prices[ticker] / yesterday_prices[ticker]
            else:
                price_relative = 1.0  # No change for missing data
            price_relatives.append(price_relative)

        # Convert to numpy array
        price_relatives = np.array(price_relatives)

        # Update portfolio using RPRT algorithm
        new_portfolio = self.update_rprt(price_relatives)

        # Convert portfolio weights to dictionary
        weights = {ticker: weight for ticker, weight in zip(self.tickers, new_portfolio)}

        return weights

    def update_rprt(self, price_relatives):
        """
        Core RPRT algorithm for portfolio optimization

        Parameters
        ----------
        price_relatives : numpy.ndarray
            Array of price relatives for each ticker

        Returns
        -------
        numpy.ndarray
            Updated portfolio weights
        """
        # Store price relatives history
        self.price_relatives_history.append(price_relatives)

        # Only keep the recent window_size price relatives
        if len(self.price_relatives_history) > self.window_size:
            self.price_relatives_history.pop(0)

        # Step 1: Calculate SMA price relative prediction and diagonal matrix D
        xhat_sma = self._calculate_sma_prediction()
        D = np.diag(xhat_sma)

        # Step 2: Calculate the reweighted price relative prediction
        self._update_price_relative_prediction(price_relatives)

        # Step 3 & 4: Calculate lambda (step size)
        phi_hat_mean = np.mean(self.phi_hat)
        phi_hat_normalized = self.phi_hat - phi_hat_mean

        norm_squared = np.sum(phi_hat_normalized ** 2)

        if norm_squared == 0:
            lambda_hat = 0
        else:
            expected_profit = self.current_portfolio.dot(self.phi_hat)
            if expected_profit < self.epsilon:
                lambda_hat = (self.epsilon - expected_profit) / norm_squared
            else:
                lambda_hat = 0

        # Step 5: Optimization step
        b_next = self.current_portfolio + lambda_hat * D.dot(phi_hat_normalized)

        # Step 6: Projection onto simplex
        b_next = self._project_to_simplex(b_next)

        # Update the current portfolio
        self.current_portfolio = b_next

        return self.current_portfolio

    def _calculate_sma_prediction(self):
        """Calculate the SMA price relative prediction."""
        if len(self.price_relatives_history) < self.window_size:
            # If we don't have enough history, use the latest price relative
            return self.price_relatives_history[-1]

        # Calculate the SMA prediction using Equation (7) from the paper
        recent_prices = np.array(self.price_relatives_history)

        # Calculate the cumulative product of price relatives to get relative prices
        # This is equivalent to p_{t-k} / p_t in the paper
        relative_prices = np.cumprod(1.0 / recent_prices[::-1], axis=0)

        # Calculate SMA prediction according to Equation (7)
        xhat_sma = (1.0 / self.window_size) * (1 + np.sum(relative_prices[:-1], axis=0))

        return xhat_sma

    def _update_price_relative_prediction(self, price_relatives):
        """Update the reweighted price relative prediction."""
        if self.phi_hat is None:
            # Initialize phi_hat with the first price relative
            self.phi_hat = price_relatives
            return

        # Calculate gamma using Equation (13)
        gamma = (self.theta * price_relatives) / (self.theta * price_relatives + self.phi_hat)

        # Update phi_hat using Equation (12)
        self.phi_hat = gamma + (1 - gamma) * (self.phi_hat / price_relatives)

    def _project_to_simplex(self, b):
        """
        Project b onto the simplex.
        Ensures that all values in b are non-negative and sum to 1.
        """
        # Handle the case where b is already a valid distribution
        if np.all(b >= 0) and np.isclose(np.sum(b), 1.0):
            return b

        # Sort b in descending order
        b_sorted = np.sort(b)[::-1]

        # Calculate the cumulative sum
        cum_sum = np.cumsum(b_sorted)

        # Find the index where the projection condition is met
        indices = np.arange(1, len(b) + 1)
        is_greater = (b_sorted * indices) > (cum_sum - 1)

        if not np.any(is_greater):
            # If no element satisfies the condition, set rho to the last element
            rho = len(b) - 1
        else:
            rho = np.max(np.where(is_greater)[0])

        # Calculate the threshold value
        theta = (cum_sum[rho] - 1) / (rho + 1)

        # Project b onto the simplex
        return np.maximum(b - theta, 0)