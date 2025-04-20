#!/usr/bin/env python
"""
Example script demonstrating the use of the portwine execution system.

This script shows how to set up and run both mock and Alpaca execution
systems with a simple momentum strategy. The Alpaca implementation uses
direct REST API calls instead of a Python SDK.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader
from portwine.execution import MockExecution
from portwine.execution import AlpacaExecution


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMomentumStrategy(StrategyBase):
    """
    A simple momentum strategy that invests in assets based on recent performance.
    
    This strategy:
    1. Calculates N-day momentum for each ticker
    2. Invests in the top performing tickers
    3. Rebalances according to momentum
    """
    
    def __init__(self, tickers, lookback_days=10, top_n=2):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        tickers : list
            List of tickers to trade
        lookback_days : int, default 10
            Number of days to look back for momentum calculation
        top_n : int, default 2
            Number of top performing tickers to invest in
        """
        super().__init__(tickers)
        self.lookback_days = lookback_days
        self.top_n = min(top_n, len(tickers))
        self.price_history = {ticker: [] for ticker in tickers}
        self.dates = []
    
    def step(self, current_date, daily_data):
        """
        Process data and generate signals.
        
        Parameters
        ----------
        current_date : pd.Timestamp
            Current date
        daily_data : dict
            Dictionary of ticker data
            
        Returns
        -------
        dict
            Dictionary of target weights
        """
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
        
        # Calculate momentum if we have enough history
        if len(self.dates) > self.lookback_days:
            momentums = {}
            for ticker in self.tickers:
                prices = self.price_history[ticker]
                if len(prices) > self.lookback_days:
                    start_price = prices[-self.lookback_days - 1]
                    end_price = prices[-1]
                    
                    if start_price is not None and end_price is not None and start_price > 0:
                        momentum = end_price / start_price - 1
                        momentums[ticker] = momentum
            
            # Sort by momentum and get top N
            if momentums:
                top_tickers = sorted(momentums.items(), key=lambda x: x[1], reverse=True)[:self.top_n]
                
                # Equal weight the top performers
                weight = 1.0 / self.top_n
                weights = {ticker: weight if ticker in [t[0] for t in top_tickers] else 0.0 
                          for ticker in self.tickers}
                
                return weights
        
        # Default to equal weight if not enough history
        weight = 1.0 / len(self.tickers)
        return {ticker: weight for ticker in self.tickers}


def run_mock_execution(tickers, days=10):
    """
    Run a mock execution for testing.
    
    Parameters
    ----------
    tickers : list
        List of tickers to trade
    days : int, default 10
        Number of days to simulate
    """
    logger.info("Setting up mock execution")
    
    # Create a strategy
    strategy = SimpleMomentumStrategy(tickers, lookback_days=5, top_n=2)
    
    # Create mock data
    from tests.test_execution2 import create_mock_price_data, MockMarketDataLoader
    
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - timedelta(days=30)
    
    # Generate mock data
    mock_data = {}
    for ticker in tickers:
        mock_data[ticker] = create_mock_price_data(
            ticker, start_date, end_date, 
            start_price=100.0 * (1 + tickers.index(ticker) * 0.1),  # Different starting prices
            volatility=0.02
        )
    
    # Create market data loader
    data_loader = MockMarketDataLoader(mock_data)
    
    # Create execution
    execution = MockExecution(
        strategy=strategy,
        market_data_loader=data_loader,
        initial_cash=100000.0
    )
    
    # Run for a number of days
    logger.info(f"Running mock execution for {days} days")
    
    simulation_dates = [end_date - timedelta(days=days-i) for i in range(days)]
    
    for i, date in enumerate(simulation_dates):
        logger.info(f"Day {i+1}: {date.strftime('%Y-%m-%d')}")
        
        # Execute step
        results = execution.step(date)
        
        # Get account info
        account_info = execution.get_account_info()
        
        # Log results
        logger.info(f"Portfolio value: ${account_info['portfolio_value']:.2f}")
        logger.info(f"Cash: ${account_info['cash']:.2f}")
        
        # Log positions
        for symbol, position in account_info['positions'].items():
            logger.info(f"Position: {symbol} - {position['qty']} shares, ${position['market_value']:.2f}")
        
        logger.info("-" * 50)
    
    logger.info("Mock execution completed")


def run_alpaca_execution(tickers, paper_trading=True):
    """
    Run a live execution with Alpaca using direct REST API calls.
    
    Parameters
    ----------
    tickers : list
        List of tickers to trade
    paper_trading : bool, default True
        Whether to use paper trading
    """
    logger.info(f"Setting up Alpaca execution (paper_trading={paper_trading})")
    
    # Check for API keys
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("Alpaca API credentials not found in environment variables")
        logger.error("Please set ALPACA_API_KEY and ALPACA_API_SECRET")
        return
    
    # Create a strategy
    strategy = SimpleMomentumStrategy(tickers, lookback_days=5, top_n=2)
    
    # Create market data loader
    from portwine.loaders.alpaca import AlpacaMarketDataLoader
    
    data_loader = AlpacaMarketDataLoader(
        api_key=api_key,
        api_secret=api_secret,
        cache_dir="./data/alpaca_cache",
        paper_trading=paper_trading
    )
    
    # Create execution
    execution = AlpacaExecution(
        strategy=strategy,
        market_data_loader=data_loader,
        api_key=api_key,
        api_secret=api_secret,
        paper_trading=paper_trading
    )
    
    # Check if market is open
    if not execution.check_market_status():
        logger.warning("Market is currently closed. Demo will still run but no trades will be executed.")
    
    # Get initial account info
    account_info = execution.get_account_info()
    logger.info(f"Initial portfolio value: ${account_info['portfolio_value']:.2f}")
    logger.info(f"Initial cash: ${account_info['cash']:.2f}")
    
    # Run a single step
    logger.info("Executing trading step...")
    results = execution.step()
    
    # Get updated account info
    account_info = execution.get_account_info()
    logger.info(f"Updated portfolio value: ${account_info['portfolio_value']:.2f}")
    logger.info(f"Updated cash: ${account_info['cash']:.2f}")
    
    # Log positions
    logger.info("Current positions:")
    for symbol, position in account_info['positions'].items():
        logger.info(f"Position: {symbol} - {position['qty']} shares, ${position['market_value']:.2f}")
    
    logger.info("Alpaca execution completed")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Execution example for portwine')
    parser.add_argument('--mode', choices=['mock', 'alpaca'], default='mock',
                      help='Execution mode (mock or alpaca)')
    parser.add_argument('--paper', action='store_true', default=True,
                      help='Use paper trading for Alpaca (default: True)')
    parser.add_argument('--days', type=int, default=10,
                      help='Number of days to simulate in mock mode (default: 10)')
    
    args = parser.parse_args()
    
    # Example tickers (tech stocks)
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
    
    if args.mode == 'mock':
        run_mock_execution(tickers, days=args.days)
    else:
        run_alpaca_execution(tickers, paper_trading=args.paper)


if __name__ == '__main__':
    main() 