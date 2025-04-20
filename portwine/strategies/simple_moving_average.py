"""
Simple Moving Average Strategy Implementation

This module provides a simple moving average crossover strategy that:
1. Calculates short and long moving averages for each ticker
2. Generates buy signals when short MA crosses above long MA
3. Generates sell signals when short MA crosses below long MA
"""

import logging
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

from portwine.strategies.base import StrategyBase

# Configure logging
logger = logging.getLogger(__name__)


class SimpleMovingAverageStrategy(StrategyBase):
    """
    Simple Moving Average Crossover Strategy
    
    This strategy:
    1. Calculates short and long moving averages for each ticker
    2. Generates buy signals when short MA crosses above long MA
    3. Generates sell signals when short MA crosses below long MA
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to trade
    short_window : int, default 20
        Short moving average window in days
    long_window : int, default 50
        Long moving average window in days
    position_size : float, default 0.1
        Position size as a fraction of portfolio (e.g., 0.1 = 10%)
    """
    
    def __init__(
        self, 
        tickers: List[str], 
        short_window: int = 20, 
        long_window: int = 50,
        position_size: float = 0.1,
        **kwargs
    ):
        """Initialize the strategy with parameters."""
        super().__init__(tickers)
        
        # Store parameters
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        
        # Validate parameters
        if short_window >= long_window:
            logger.warning("Short window should be smaller than long window")
        
        # Initialize price history
        self.price_history = {ticker: [] for ticker in tickers}
        self.dates = []
        
        # Current signals (allocations)
        self.current_signals = {ticker: 0.0 for ticker in tickers}
        
        logger.info(f"Initialized SimpleMovingAverageStrategy with {len(tickers)} tickers")
        logger.info(f"Parameters: short_window={short_window}, long_window={long_window}, position_size={position_size}")
    
    def calculate_moving_averages(self, prices: List[float]) -> Dict[str, Optional[float]]:
        """
        Calculate short and long moving averages from price history.
        
        Parameters
        ----------
        prices : List[float]
            List of historical prices
            
        Returns
        -------
        Dict[str, Optional[float]]
            Dictionary with short_ma and long_ma values, or None if not enough data
        """
        if len(prices) < self.long_window:
            return {"short_ma": None, "long_ma": None}
        
        # Calculate moving averages
        short_ma = sum(prices[-self.short_window:]) / self.short_window
        long_ma = sum(prices[-self.long_window:]) / self.long_window
        
        return {"short_ma": short_ma, "long_ma": long_ma}
    
    def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Process daily data and generate trading signals.
        
        Parameters
        ----------
        current_date : pd.Timestamp
            Current trading date
        daily_data : Dict[str, Dict[str, Any]]
            Dictionary of ticker data for the current date
            
        Returns
        -------
        Dict[str, float]
            Dictionary of target weights for each ticker
        """
        # Track dates
        self.dates.append(current_date)
        
        # Update price history
        for ticker in self.tickers:
            price = None
            if ticker in daily_data and daily_data[ticker] is not None:
                price = daily_data[ticker].get('close')
            
            # Forward fill missing data
            if price is None and len(self.price_history[ticker]) > 0:
                price = self.price_history[ticker][-1]
            
            self.price_history[ticker].append(price)
        
        # Calculate signals for each ticker
        signals = {}
        for ticker in self.tickers:
            prices = self.price_history[ticker]
            
            # Skip tickers with None values
            if None in prices:
                signals[ticker] = 0.0
                continue
            
            # Calculate moving averages
            mas = self.calculate_moving_averages(prices)
            
            # Not enough data yet
            if mas["short_ma"] is None or mas["long_ma"] is None:
                signals[ticker] = 0.0
                continue
            
            # Get previous signal
            prev_signal = self.current_signals.get(ticker, 0.0)
            
            # Generate signal based on moving average crossover
            if mas["short_ma"] > mas["long_ma"]:
                # Bullish signal - short MA above long MA
                signals[ticker] = self.position_size
            else:
                # Bearish signal - short MA below long MA
                signals[ticker] = 0.0
            
            # Log signal changes
            if signals[ticker] != prev_signal:
                direction = "BUY" if signals[ticker] > 0 else "SELL"
                logger.info(f"{current_date}: {direction} signal for {ticker} - Short MA: {mas['short_ma']:.2f}, Long MA: {mas['long_ma']:.2f}")
        
        # Update current signals
        self.current_signals = signals.copy()
        
        return signals
    
    def generate_signals(self) -> Dict[str, float]:
        """
        Generate current trading signals.
        
        This method is used by the DailyExecutor to get the current signals.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of target weights for each ticker
        """
        return self.current_signals.copy()
    
    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down SimpleMovingAverageStrategy") 