"""
Unit tests for the execution_complex module.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from portwine.execution_complex import Order, Position
from portwine.execution_complex import MockExecution
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader


class MockMarketDataLoader(MarketDataLoader):
    """Mock market data loader for testing."""
    
    def __init__(self, mock_data=None):
        super().__init__()
        self.mock_data = mock_data or {}
    
    def load_ticker(self, ticker):
        """Return pre-defined mock data for a ticker."""
        return self.mock_data.get(ticker)
    
    def fetch_data(self, tickers):
        """Fetch data for multiple tickers."""
        result = {}
        for ticker in tickers:
            data = self.load_ticker(ticker)
            if data is not None:
                result[ticker] = data
        return result
    
    def next(self, tickers, timestamp):
        """Get data for tickers at the given timestamp."""
        result = {}
        for ticker in tickers:
            df = self.fetch_data([ticker]).get(ticker)
            if df is not None:
                bar = self._get_bar_at_or_before(df, timestamp)
                if bar is not None:
                    result[ticker] = {
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "volume": float(bar["volume"]),
                    }
                else:
                    result[ticker] = None
            else:
                result[ticker] = None
        return result


class SimpleTestStrategy(StrategyBase):
    """Simple strategy for testing."""
    
    def __init__(self, tickers):
        super().__init__(tickers)
        # Default to equal weight
        self.weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
    
    def step(self, current_date, daily_data):
        """Return fixed allocations for testing."""
        return self.weights.copy()
    
    def set_weights(self, new_weights):
        """Update weights for testing."""
        self.weights = new_weights


def create_mock_price_data(ticker, start_date, end_date, start_price=100.0, volatility=0.01):
    """Create mock price data for testing."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(date_range)
    
    # Generate random returns
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0, volatility, n)
    
    # Calculate prices
    log_returns = np.log(1 + returns)
    log_prices = np.cumsum(log_returns) + np.log(start_price)
    prices = np.exp(log_prices)
    
    # Create DataFrame
    data = {
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.005, n)),
        'low': prices * (1 - np.random.uniform(0, 0.005, n)),
        'close': prices,
        'volume': np.random.randint(1000, 100000, n)
    }
    
    return pd.DataFrame(data, index=date_range)


class TestMockExecution(unittest.TestCase):
    """Test the MockExecution class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create date range
        self.start_date = pd.Timestamp('2023-01-01')
        self.end_date = pd.Timestamp('2023-01-10')
        
        # Create mock data
        self.tickers = ['AAPL', 'MSFT', 'GOOG']
        mock_data = {}
        for ticker in self.tickers:
            mock_data[ticker] = create_mock_price_data(ticker, self.start_date, self.end_date)
        
        # Create market data loader
        self.data_loader = MockMarketDataLoader(mock_data)
        
        # Create strategy
        self.strategy = SimpleTestStrategy(self.tickers)
        
        # Create execution_complex
        self.execution = MockExecution(
            strategy=self.strategy,
            market_data_loader=self.data_loader,
            initial_cash=100000.0
        )
    
    def test_initialization(self):
        """Test initialization of MockExecution."""
        self.assertEqual(self.execution.cash, 100000.0)
        self.assertEqual(len(self.execution.positions), 0)
        self.assertEqual(len(self.execution.executed_orders), 0)
    
    def test_get_account_info(self):
        """Test getting account info."""
        account_info = self.execution.get_account_info()
        
        self.assertEqual(account_info['cash'], 100000.0)
        self.assertEqual(account_info['portfolio_value'], 100000.0)
        self.assertEqual(len(account_info['positions']), 0)
    
    def test_execute_order_buy(self):
        """Test executing a buy order."""
        # Create a buy order
        order: Order = {
            'symbol': 'AAPL',
            'qty': 10,
            'order_type': 'market',
            'time_in_force': 'day',
            'limit_price': None,
        }
        
        # Execute order
        success = self.execution.execute_order(order)
        
        # Check result
        self.assertTrue(success)
        self.assertEqual(len(self.execution.executed_orders), 1)
        
        # Check portfolio
        account_info = self.execution.get_account_info()
        self.assertLess(account_info['cash'], 100000.0)
        self.assertIn('AAPL', account_info['positions'])
        self.assertEqual(account_info['positions']['AAPL']['qty'], 10)
    
    def test_execute_order_sell(self):
        """Test executing a sell order."""
        # First buy some shares
        buy_order: Order = {
            'symbol': 'AAPL',
            'qty': 10,
            'order_type': 'market',
            'time_in_force': 'day',
            'limit_price': None,
        }
        self.execution.execute_order(buy_order)
        
        # Then sell some
        sell_order: Order = {
            'symbol': 'AAPL',
            'qty': -5,
            'order_type': 'market',
            'time_in_force': 'day',
            'limit_price': None,
        }
        success = self.execution.execute_order(sell_order)
        
        # Check result
        self.assertTrue(success)
        self.assertEqual(len(self.execution.executed_orders), 2)
        
        # Check portfolio
        account_info = self.execution.get_account_info()
        self.assertIn('AAPL', account_info['positions'])
        self.assertEqual(account_info['positions']['AAPL']['qty'], 5)
    
    def test_execute_order_fail(self):
        """Test executing an order that fails."""
        # Create an execution_complex with failing symbols
        execution = MockExecution(
            strategy=self.strategy,
            market_data_loader=self.data_loader,
            initial_cash=100000.0,
            fail_symbols={'AAPL'}
        )
        
        # Create an order for a failing symbol
        order: Order = {
            'symbol': 'AAPL',
            'qty': 10,
            'order_type': 'market',
            'time_in_force': 'day',
            'limit_price': None,
        }
        
        # Execute order
        success = execution.execute_order(order)
        
        # Check result
        self.assertFalse(success)
        self.assertEqual(len(execution.executed_orders), 0)
        
        # Check portfolio (should be unchanged)
        account_info = execution.get_account_info()
        self.assertEqual(account_info['cash'], 100000.0)
        self.assertEqual(len(account_info['positions']), 0)
    
    def test_step(self):
        """Test executing a step."""
        # Set strategy to allocate 30% to each stock
        weights = {ticker: 0.3 for ticker in self.tickers}  # 90% total allocation
        self.strategy.set_weights(weights)
        
        # Execute step
        results = self.execution.step(self.end_date)
        
        # Check results
        self.assertEqual(len(results), 3)
        self.assertTrue(all(results.values()))
        
        # Check portfolio
        account_info = self.execution.get_account_info()
        self.assertLess(account_info['cash'], 100000.0)
        self.assertEqual(len(account_info['positions']), 3)
        
        # Check position weights
        portfolio_value = account_info['portfolio_value']
        for ticker in self.tickers:
            self.assertIn(ticker, account_info['positions'])
            position = account_info['positions'][ticker]
            weight = position['market_value'] / portfolio_value
            self.assertAlmostEqual(weight, 0.3, delta=0.05)
    
    def test_calculate_position_changes(self):
        """Test calculation of position changes."""
        # Set up account with some positions
        self.execution.positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 10,
                'market_value': 1500.0,
                'avg_entry_price': 150.0,
                'unrealized_pl': 0.0,
            },
            'MSFT': {
                'symbol': 'MSFT',
                'qty': 5,
                'market_value': 1000.0,
                'avg_entry_price': 200.0,
                'unrealized_pl': 0.0,
            }
        }
        
        # Create target weights
        target_weights = {
            'AAPL': 0.1,  # Decrease
            'MSFT': 0.3,  # Increase
            'GOOG': 0.2,  # New position
        }
        
        # Calculate position changes
        account_info = {
            'cash': 7500.0,
            'portfolio_value': 10000.0,
            'positions': self.execution.positions,
        }
        changes = self.execution.calculate_position_changes(target_weights, account_info)
        
        # Check changes
        self.assertAlmostEqual(changes['AAPL'], -500.0)  # From 1500 to 1000
        self.assertAlmostEqual(changes['MSFT'], 2000.0)  # From 1000 to 3000
        self.assertAlmostEqual(changes['GOOG'], 2000.0)  # From 0 to 2000
    
    def test_generate_orders(self):
        """Test generating orders from position changes."""
        # Set up position changes
        position_changes = {
            'AAPL': -500.0,
            'MSFT': 2000.0,
            'GOOG': 2000.0,
        }
        
        # Set up prices
        prices = {
            'AAPL': 150.0,
            'MSFT': 200.0,
            'GOOG': 100.0,
        }
        
        # Generate orders
        orders = self.execution.generate_orders(position_changes, prices)
        
        # Check orders
        self.assertEqual(len(orders), 3)
        
        # Check AAPL order
        aapl_order = next((o for o in orders if o['symbol'] == 'AAPL'), None)
        self.assertIsNotNone(aapl_order)
        self.assertEqual(aapl_order['qty'], -3)  # -500 / 150 = -3.33, rounded to -3
        
        # Check MSFT order
        msft_order = next((o for o in orders if o['symbol'] == 'MSFT'), None)
        self.assertIsNotNone(msft_order)
        self.assertEqual(msft_order['qty'], 10)  # 2000 / 200 = 10
        
        # Check GOOG order
        goog_order = next((o for o in orders if o['symbol'] == 'GOOG'), None)
        self.assertIsNotNone(goog_order)
        self.assertEqual(goog_order['qty'], 20)  # 2000 / 100 = 20


if __name__ == '__main__':
    unittest.main() 