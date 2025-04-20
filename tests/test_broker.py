"""
Tests for the broker module.

This module contains tests for the BrokerBase abstract class and its implementations.
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from portwine.execution.broker import BrokerBase, AccountInfo, Position, Order


class MockBroker(BrokerBase):
    """Simple mock broker implementation for testing."""
    
    def __init__(self, initial_cash=100000.0, positions=None):
        """Initialize with test data."""
        self.cash = initial_cash
        self.positions = positions or {}
        self.market_open = True
        self.executed_orders = []
    
    def check_market_status(self) -> bool:
        """Check if the market is currently open."""
        return self.market_open
    
    def get_account_info(self) -> dict:
        """Get current account information."""
        position_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
        return {
            'cash': self.cash,
            'portfolio_value': self.cash + position_value,
            'positions': self.positions
        }
    
    def execute_order(self, symbol, qty, order_type="market", **kwargs) -> bool:
        """Execute an order."""
        self.executed_orders.append({
            'symbol': symbol,
            'qty': qty,
            'order_type': order_type,
            **kwargs
        })
        return True
    
    def get_order_status(self, order_id):
        """Get order status."""
        return "filled"
    
    def cancel_all_orders(self):
        """Cancel all orders."""
        return True
    
    def close_all_positions(self):
        """Close all positions."""
        self.positions = {}
        return True
    
    def get_positions(self):
        """Get positions list."""
        return list(self.positions.values())
    
    def get_cash(self):
        """Get available cash."""
        return self.cash
    
    def get_portfolio_value(self):
        """Get portfolio value."""
        position_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
        return self.cash + position_value


class TestBrokerBase(unittest.TestCase):
    """Tests for the BrokerBase class and implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample broker with test data
        self.positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'qty': 10,
                'market_value': 1500.0,
                'avg_entry_price': 140.0,
                'unrealized_pl': 100.0
            },
            'MSFT': {
                'symbol': 'MSFT',
                'qty': 5,
                'market_value': 1200.0,
                'avg_entry_price': 220.0,
                'unrealized_pl': 100.0
            }
        }
        self.broker = MockBroker(initial_cash=10000.0, positions=self.positions)
    
    def test_get_position(self):
        """Test getting a specific position."""
        position = self.broker.get_position('AAPL')
        self.assertEqual(position['symbol'], 'AAPL')
        self.assertEqual(position['qty'], 10)
        
        # Test getting non-existent position
        position = self.broker.get_position('GOOG')
        self.assertIsNone(position)
    
    def test_get_portfolio_weights(self):
        """Test calculating portfolio weights."""
        weights = self.broker.get_portfolio_weights()
        
        total_value = 10000.0 + 1500.0 + 1200.0  # cash + AAPL + MSFT
        
        expected_weights = {
            'AAPL': 1500.0 / total_value,
            'MSFT': 1200.0 / total_value
        }
        
        self.assertAlmostEqual(weights['AAPL'], expected_weights['AAPL'], places=6)
        self.assertAlmostEqual(weights['MSFT'], expected_weights['MSFT'], places=6)
    
    def test_execute_order(self):
        """Test order execution."""
        result = self.broker.execute_order('GOOG', 3, order_type='limit', limit_price=150.0)
        self.assertTrue(result)
        
        self.assertEqual(len(self.broker.executed_orders), 1)
        order = self.broker.executed_orders[0]
        self.assertEqual(order['symbol'], 'GOOG')
        self.assertEqual(order['qty'], 3)
        self.assertEqual(order['order_type'], 'limit')
        self.assertEqual(order['limit_price'], 150.0)
    
    def test_check_market_status(self):
        """Test market status check."""
        self.assertTrue(self.broker.check_market_status())
        
        self.broker.market_open = False
        self.assertFalse(self.broker.check_market_status())
    
    def test_get_account_info(self):
        """Test getting account information."""
        account_info = self.broker.get_account_info()
        
        self.assertEqual(account_info['cash'], 10000.0)
        self.assertEqual(account_info['portfolio_value'], 12700.0)  # 10000 + 1500 + 1200
        self.assertEqual(account_info['positions'], self.positions)
    
    def test_get_positions(self):
        """Test getting all positions."""
        positions = self.broker.get_positions()
        self.assertEqual(len(positions), 2)
        self.assertIn(self.positions['AAPL'], positions)
        self.assertIn(self.positions['MSFT'], positions)
    
    def test_get_cash(self):
        """Test getting available cash."""
        self.assertEqual(self.broker.get_cash(), 10000.0)
    
    def test_get_portfolio_value(self):
        """Test getting portfolio value."""
        self.assertEqual(self.broker.get_portfolio_value(), 12700.0)  # 10000 + 1500 + 1200


if __name__ == '__main__':
    unittest.main() 