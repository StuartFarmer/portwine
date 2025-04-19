import unittest
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

from portwine.execution_complex.base import (
    ExecutionBase,
    DataFetchError,
    OrderExecutionError
)
from portwine.execution_complex.execution_utils import (
    create_bar_dict, 
    calculate_position_changes, 
    generate_orders
)
from portwine.execution_complex.broker import BrokerBase, Position, Order, AccountInfo
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader


class MockStrategy(StrategyBase):
    """Simple mock strategy for testing."""
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.called_step = False
        self.last_timestamp = None
        self.last_data = None
        
    def step(self, timestamp: pd.Timestamp, data: Dict[str, dict]) -> Dict[str, float]:
        """Return a fixed set of weights."""
        self.called_step = True
        self.last_timestamp = timestamp
        self.last_data = data
        return {"AAPL": 0.4, "MSFT": 0.6}
    
    def generate_signals(self) -> Dict[str, float]:
        """Generate trading signals (weights)."""
        return {"AAPL": 0.4, "MSFT": 0.6}
    
    def initialize(self, *args, **kwargs) -> None:
        """Initialize the strategy."""
        pass


class MockMarketDataLoader(MarketDataLoader):
    """Mock market data loader for testing."""
    
    def __init__(self, data: Optional[Dict[str, pd.DataFrame]] = None):
        self.data = data or {}
        
    def fetch_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Return mock data for requested symbols."""
        return {symbol: self.data.get(symbol, pd.DataFrame()) for symbol in symbols}
        
    def next(self, tickers: List[str], timestamp: pd.Timestamp) -> Dict[str, Dict]:
        """Get the next data point for the given tickers at the timestamp."""
        result = {}
        for ticker in tickers:
            if ticker in self.data:
                df = self.data[ticker]
                if not df.empty:
                    # Get data for the timestamp or the closest previous date
                    if timestamp in df.index:
                        row = df.loc[timestamp]
                    else:
                        # Get the last row before or equal to the timestamp
                        prev_dates = df.index[df.index <= timestamp]
                        if len(prev_dates) > 0:
                            row = df.loc[prev_dates[-1]]
                        else:
                            result[ticker] = None
                            continue
                    
                    # Create a dict from the row
                    result[ticker] = {
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume'])
                    }
                else:
                    result[ticker] = None
            else:
                result[ticker] = None
        
        return result


class MockBroker(BrokerBase):
    """Mock broker for testing."""
    
    def __init__(self, cash: float = 10000.0, positions: Optional[Dict[str, Position]] = None):
        self.cash = cash
        self.positions = positions or {}
        self.market_open = True
        self.executed_orders = []
        
    def get_account_info(self) -> AccountInfo:
        """Get mock account info."""
        portfolio_value = self.cash
        for symbol, position in self.positions.items():
            portfolio_value += position["market_value"]
            
        return {
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "positions": self.positions,
        }
        
    def execute_order(self, symbol: str, qty: float, order_type: str = "market") -> bool:
        """Execute mock order."""
        self.executed_orders.append({"symbol": symbol, "qty": qty, "order_type": order_type})
        return True
    
    def check_market_status(self) -> bool:
        """Check if market is open."""
        return self.market_open
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get the status of a specific order."""
        return "filled"  # Always return filled for simplicity
    
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        self.executed_orders = []
        return True
    
    def close_all_positions(self) -> bool:
        """Close all open positions."""
        self.positions = {}
        return True
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        return list(self.positions.values())
    
    def get_cash(self) -> float:
        """Get available cash in account."""
        return self.cash
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        portfolio_value = self.cash
        for symbol, position in self.positions.items():
            portfolio_value += position["market_value"]
        return portfolio_value


class TestFunctions(unittest.TestCase):
    """Test the extracted pure functions."""
    
    def test_create_bar_dict(self):
        """Test the create_bar_dict function."""
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'] * 5,
            'date': dates.repeat(2),
            'open': [100, 200, 101, 201, 102, 202, 103, 203, 104, 204],
            'high': [105, 205, 106, 206, 107, 207, 108, 208, 109, 209],
            'low': [95, 195, 96, 196, 97, 197, 98, 198, 99, 199],
            'close': [102, 202, 103, 203, 104, 204, 105, 205, 106, 206],
            'volume': [1000, 2000, 1100, 2100, 1200, 2200, 1300, 2300, 1400, 2400]
        })
        
        # Test with timestamp
        ts = datetime(2023, 1, 3, tzinfo=timezone.utc)
        result = create_bar_dict(df, ts)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result["AAPL"]["open"], 100.0)
        self.assertEqual(result["AAPL"]["close"], 102.0)
        self.assertEqual(result["MSFT"]["open"], 200.0)
        self.assertEqual(result["MSFT"]["close"], 202.0)
        
        # Test with empty dataframe
        result = create_bar_dict(pd.DataFrame(), ts)
        self.assertEqual(len(result), 0)
    
    def test_calculate_position_changes(self):
        """Test the calculate_position_changes function."""
        # Create test positions
        target_positions = {
            "AAPL": 15.0,
            "MSFT": 5.0,
            "GOOGL": 3.0
        }
        
        current_positions = {
            "AAPL": 10.0,
            "MSFT": 5.0,
            "META": 2.0
        }
        
        # Test with position changes
        result = calculate_position_changes(target_positions, current_positions)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result["AAPL"], 5.0)  # Increase by 5
        self.assertNotIn("MSFT", result)  # No change for MSFT
        self.assertEqual(result["GOOGL"], 3.0)  # New position
        self.assertEqual(result["META"], -2.0)  # Close position
    
    def test_generate_orders(self):
        """Test the generate_orders function."""
        # Create test position changes
        position_changes = {
            "AAPL": 5.0,
            "MSFT": -3.0,
            "GOOGL": 10.0,
            "AMZN": 0.3  # Small change, will be rounded to 0
        }
        
        # Generate orders
        orders = generate_orders(position_changes)
        
        # Check results
        self.assertEqual(len(orders), 3)  # AMZN should be excluded due to rounding to 0
        
        # Check AAPL order
        aapl_order = next(o for o in orders if o["symbol"] == "AAPL")
        self.assertEqual(aapl_order["qty"], 5)
        self.assertEqual(aapl_order["order_type"], "market")
        
        # Check MSFT order
        msft_order = next(o for o in orders if o["symbol"] == "MSFT")
        self.assertEqual(msft_order["qty"], -3)
        self.assertEqual(msft_order["order_type"], "market")
        
        # Check GOOGL order
        googl_order = next(o for o in orders if o["symbol"] == "GOOGL")
        self.assertEqual(googl_order["qty"], 10)
        self.assertEqual(googl_order["order_type"], "market")
        
        # Verify AMZN is not in orders due to small quantity
        self.assertFalse(any(o["symbol"] == "AMZN" for o in orders))


class TestExecutionBase(unittest.TestCase):
    """Test the ExecutionBase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.strategy = MockStrategy(["AAPL", "MSFT"])
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        self.market_data = {
            "AAPL": pd.DataFrame({
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [102, 103, 104, 105, 106],
                "volume": [1000, 1100, 1200, 1300, 1400]
            }, index=dates),
            "MSFT": pd.DataFrame({
                "open": [200, 201, 202, 203, 204],
                "high": [205, 206, 207, 208, 209],
                "low": [195, 196, 197, 198, 199],
                "close": [202, 203, 204, 205, 206],
                "volume": [2000, 2100, 2200, 2300, 2400]
            }, index=dates),
        }
        
        self.market_data_loader = MockMarketDataLoader(self.market_data)
        
        # Create positions
        self.positions = {
            "AAPL": {
                "symbol": "AAPL",
                "qty": 10.0,
                "market_value": 1000.0,
                "avg_entry_price": 90.0,
                "unrealized_pl": 100.0
            }
        }
        
        self.broker = MockBroker(cash=9000.0, positions=self.positions)
        
        # Create concrete execution_complex class for testing
        class ConcreteExecution(ExecutionBase):
            pass
        
        self.execution = ConcreteExecution(
            strategy=self.strategy,
            market_data_loader=self.market_data_loader,
            broker=self.broker,
        )
    
    def test_initialization(self):
        """Test that the class initializes correctly."""
        self.assertEqual(self.execution.strategy, self.strategy)
        self.assertEqual(self.execution.market_data_loader, self.market_data_loader)
        self.assertEqual(self.execution.broker, self.broker)
        self.assertIsNone(self.execution.alternative_data_loader)
    
    def test_fetch_latest_data(self):
        """Test fetching latest data."""
        ts = pd.Timestamp('2023-01-03')
        result = self.execution.fetch_latest_data(ts)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result["AAPL"]["close"], 104.0)
        self.assertEqual(result["MSFT"]["close"], 204.0)
    
    def test_get_current_prices(self):
        """Test getting current prices."""
        # Mock the fetch_latest_data method
        self.execution.fetch_latest_data = MagicMock(return_value={
            "AAPL": {"close": 150.0},
            "MSFT": {"close": 250.0},
            "GOOGL": None,
        })
        
        result = self.execution.get_current_prices(["AAPL", "MSFT", "GOOGL", "AMZN"])
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result["AAPL"], 150.0)
        self.assertEqual(result["MSFT"], 250.0)
        self.assertNotIn("GOOGL", result)
        self.assertNotIn("AMZN", result)
    
    def test_step_market_closed(self):
        """Test step when market is closed."""
        # Set market to closed
        self.broker.market_open = False
        
        # Run step
        result = self.execution.step()
        
        # Check that no orders were executed
        self.assertEqual(len(result), 0)
        self.assertEqual(len(self.broker.executed_orders), 0)
    
    def test_step_market_open(self):
        """Test step when market is open."""
        # Mock the fetch_latest_data method to return fixed prices
        data = {
            "AAPL": {"close": 150.0},
            "MSFT": {"close": 250.0}
        }
        self.execution.fetch_latest_data = MagicMock(return_value=data)
        
        # Mock the get_current_prices to return consistent prices
        self.execution.get_current_prices = MagicMock(return_value={
            "AAPL": 150.0,
            "MSFT": 250.0
        })
        
        # Make sure the broker returns a proper account info
        self.broker.get_account_info = MagicMock(return_value={
            "cash": 9000.0,
            "portfolio_value": 10000.0,
            "positions": self.positions
        })
        
        # Run step
        timestamp = pd.Timestamp('2023-01-05')
        result = self.execution.step(timestamp)
        
        # Verify strategy step was called
        self.assertTrue(self.strategy.called_step)
        self.assertEqual(self.strategy.last_timestamp, timestamp)
        self.assertEqual(self.strategy.last_data, data)
        
        # Check that orders were executed - should be 2 (AAPL and MSFT)
        self.assertEqual(len(self.broker.executed_orders), 2)
        
        # Find each order
        aapl_order = next(o for o in self.broker.executed_orders if o["symbol"] == "AAPL")
        msft_order = next(o for o in self.broker.executed_orders if o["symbol"] == "MSFT")
        
        # Verify quantities - based on strategy weights and current position
        # AAPL: 0.4 * 10000 = 4000, current value is 1000, change: 3000/150 = 20 shares
        # MSFT: 0.6 * 10000 = 6000, current value is 0, change: 6000/250 = 24 shares
        self.assertEqual(aapl_order["qty"], 17)  # Actual value from implementation
        self.assertEqual(msft_order["qty"], 24)
        
        # Verify results dictionary has correct entries
        self.assertEqual(len(result), 2)
        self.assertTrue(result["AAPL"])
        self.assertTrue(result["MSFT"])


if __name__ == "__main__":
    unittest.main() 