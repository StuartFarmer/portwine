#!/usr/bin/env python
"""
Comprehensive test suite for the MockBroker class.

This module tests all methods of the MockBroker class, covering both
positive and negative cases, as well as different branches (e.g., market
open vs. closed).
"""

import unittest
from decimal import Decimal
from datetime import datetime

from portwine.execution_complex.broker_mock import MockBroker


class TestMockBroker(unittest.TestCase):
    """Test suite for the MockBroker class."""

    def setUp(self):
        """Set up a fresh MockBroker instance before each test."""
        self.broker = MockBroker(initial_cash=100000.0, market_hours=True)

    def test_initialization(self):
        """Test broker initialization with different parameters."""
        # Test default initialization
        broker1 = MockBroker()
        self.assertEqual(broker1.cash, 100000.0)
        self.assertTrue(broker1.market_open)

        # Test custom initialization
        broker2 = MockBroker(initial_cash=50000.0, market_hours=False)
        self.assertEqual(broker2.cash, 50000.0)
        self.assertFalse(broker2.market_open)

    def test_get_account_info_empty(self):
        """Test getting account info with no positions."""
        account_info = self.broker.get_account_info()

        self.assertEqual(account_info['cash'], 100000.0)
        self.assertEqual(account_info['portfolio_value'], 100000.0)
        self.assertEqual(account_info['buying_power'], 200000.0)  # 2x margin
        self.assertEqual(account_info['equity'], 100000.0)
        self.assertEqual(account_info['status'], 'ACTIVE')

    def test_get_account_info_with_positions(self):
        """Test getting account info with positions."""
        # Add a position
        self.broker.execute_order('AAPL', 10)  # Buy 10 shares

        account_info = self.broker.get_account_info()

        # Cash should be reduced
        self.assertEqual(account_info['cash'], 99000.0)  # 100000 - (10 * 100)

        # Portfolio value should include positions
        self.assertEqual(account_info['portfolio_value'], 100000.0)  # Cash + position value

        # Buying power should be 2x cash
        self.assertEqual(account_info['buying_power'], 198000.0)  # 2 * 99000

    def test_execute_order_buy_success(self):
        """Test executing a buy order successfully."""
        # Execute a buy order
        result = self.broker.execute_order('AAPL', 10)  # Buy 10 shares

        # Assert order was successful
        self.assertTrue(result)

        # Check cash was reduced
        self.assertEqual(self.broker.cash, 99000.0)  # 100000 - (10 * 100)

        # Check position was created
        self.assertIn('AAPL', self.broker.positions)
        self.assertEqual(self.broker.positions['AAPL']['qty'], 10)
        self.assertEqual(self.broker.positions['AAPL']['cost_basis'], 100.0)

    def test_execute_order_buy_insufficient_cash(self):
        """Test executing a buy order with insufficient cash."""
        # Try to buy more than we can afford
        result = self.broker.execute_order('AAPL', 1001)  # Buy 1001 shares (100100 > 100000)

        # Assert order failed
        self.assertFalse(result)

        # Check cash wasn't changed
        self.assertEqual(self.broker.cash, 100000.0)

        # Check no position was created
        self.assertNotIn('AAPL', self.broker.positions)

    def test_execute_order_buy_additional(self):
        """Test buying more of an existing position."""
        # Buy initial position
        self.broker.execute_order('AAPL', 10)  # Buy 10 shares

        # Buy more
        result = self.broker.execute_order('AAPL', 5)  # Buy 5 more shares

        # Assert order was successful
        self.assertTrue(result)

        # Check cash was reduced properly
        self.assertEqual(self.broker.cash, 98500.0)  # 100000 - (10 * 100) - (5 * 100)

        # Check position was updated correctly
        self.assertEqual(self.broker.positions['AAPL']['qty'], 15)
        self.assertEqual(self.broker.positions['AAPL']['cost_basis'], 100.0)

    def test_execute_order_sell_success(self):
        """Test executing a sell order successfully."""
        # First buy some shares
        self.broker.execute_order('AAPL', 10)  # Buy 10 shares

        # Then sell some shares
        result = self.broker.execute_order('AAPL', -5)  # Sell 5 shares

        # Assert order was successful
        self.assertTrue(result)

        # Check cash was increased
        self.assertEqual(self.broker.cash, 99500.0)  # 100000 - (10 * 100) + (5 * 100)

        # Check position was updated
        self.assertEqual(self.broker.positions['AAPL']['qty'], 5)

    def test_execute_order_sell_all(self):
        """Test selling an entire position."""
        # First buy some shares
        self.broker.execute_order('AAPL', 10)  # Buy 10 shares

        # Then sell all shares
        result = self.broker.execute_order('AAPL', -10)  # Sell 10 shares

        # Assert order was successful
        self.assertTrue(result)

        # Check cash was increased
        self.assertEqual(self.broker.cash, 100000.0)  # Back to original

        # Check position was removed
        self.assertNotIn('AAPL', self.broker.positions)

    def test_execute_order_sell_no_position(self):
        """Test selling a position that doesn't exist."""
        # Try to sell shares we don't have
        result = self.broker.execute_order('AAPL', -5)  # Sell 5 shares

        # Assert order failed
        self.assertFalse(result)

        # Check cash wasn't changed
        self.assertEqual(self.broker.cash, 100000.0)

    def test_execute_order_sell_too_many(self):
        """Test selling more shares than owned."""
        # First buy some shares
        self.broker.execute_order('AAPL', 10)  # Buy 10 shares

        # Try to sell more than we have
        result = self.broker.execute_order('AAPL', -15)  # Sell 15 shares

        # Assert order failed
        self.assertFalse(result)

        # Check cash wasn't changed further
        self.assertEqual(self.broker.cash, 99000.0)  # 100000 - (10 * 100)

        # Check position wasn't changed
        self.assertEqual(self.broker.positions['AAPL']['qty'], 10)

    def test_execute_order_market_closed(self):
        """Test executing an order when market is closed."""
        # Close the market
        self.broker.set_market_status(False)

        # Try to execute an order
        result = self.broker.execute_order('AAPL', 10)

        # Assert order failed
        self.assertFalse(result)

        # Check cash wasn't changed
        self.assertEqual(self.broker.cash, 100000.0)

        # Check no position was created
        self.assertNotIn('AAPL', self.broker.positions)

    def test_check_market_status(self):
        """Test checking market status."""
        # Default is market open
        self.assertTrue(self.broker.check_market_status())

        # Change to market closed
        self.broker.set_market_status(False)
        self.assertFalse(self.broker.check_market_status())

        # Change back to market open
        self.broker.set_market_status(True)
        self.assertTrue(self.broker.check_market_status())

    def test_get_order_status(self):
        """Test getting order status."""
        # Execute an order to create an order
        self.broker.execute_order('AAPL', 10)

        # Get the order ID
        order_id = '1'  # The first order ID is always '1'

        # Check order status
        status = self.broker.get_order_status(order_id)
        self.assertEqual(status, 'filled')

        # Check non-existent order
        status = self.broker.get_order_status('999')
        self.assertIsNone(status)

    def test_cancel_all_orders(self):
        """Test canceling all orders."""
        # First execute some orders
        self.broker.execute_order('AAPL', 10)
        self.broker.execute_order('MSFT', 10)

        # Cancel all orders
        result = self.broker.cancel_all_orders()

        # Assert cancellation was successful
        self.assertTrue(result)

        # Check all orders are marked as canceled or filled
        for order_id, order in self.broker.orders.items():
            self.assertIn(order['status'], ['filled', 'canceled'])

    def test_close_all_positions_success(self):
        """Test closing all positions successfully."""
        # First create some positions
        self.broker.execute_order('AAPL', 10)
        self.broker.execute_order('MSFT', 15)

        # Close all positions
        result = self.broker.close_all_positions()

        # Assert close was successful
        self.assertTrue(result)

        # Check all positions are gone
        self.assertEqual(len(self.broker.positions), 0)

        # Check cash is back to original (plus any gain/loss)
        self.assertEqual(self.broker.cash, 100000.0)

    def test_close_all_positions_market_closed(self):
        """Test closing all positions when market is closed."""
        # First create some positions
        self.broker.execute_order('AAPL', 10)
        self.broker.execute_order('MSFT', 15)

        # Close the market
        self.broker.set_market_status(False)

        # Try to close all positions
        result = self.broker.close_all_positions()

        # Assert close failed
        self.assertFalse(result)

        # Check positions still exist
        self.assertEqual(len(self.broker.positions), 2)

        # Check cash wasn't changed
        self.assertEqual(self.broker.cash, 97500.0)  # 100000 - (10 * 100) - (15 * 100)

    def test_get_positions_empty(self):
        """Test getting positions when there are none."""
        positions = self.broker.get_positions()

        # Should be an empty list
        self.assertEqual(len(positions), 0)

    def test_get_positions_with_data(self):
        """Test getting positions with existing positions."""
        # Create some positions
        self.broker.execute_order('AAPL', 10)
        self.broker.execute_order('MSFT', 15)

        positions = self.broker.get_positions()

        # Should have 2 positions
        self.assertEqual(len(positions), 2)

        # Check the first position data
        aapl_pos = next((p for p in positions if p['symbol'] == 'AAPL'), None)
        self.assertIsNotNone(aapl_pos)
        self.assertEqual(aapl_pos['qty'], 10)
        self.assertEqual(aapl_pos['cost_basis'], 100.0)
        self.assertEqual(aapl_pos['market_value'], 1000.0)

        # Check the second position data
        msft_pos = next((p for p in positions if p['symbol'] == 'MSFT'), None)
        self.assertIsNotNone(msft_pos)
        self.assertEqual(msft_pos['qty'], 15)
        self.assertEqual(msft_pos['cost_basis'], 100.0)
        self.assertEqual(msft_pos['market_value'], 1500.0)

    def test_get_cash(self):
        """Test getting cash amount."""
        # Initial cash
        self.assertEqual(self.broker.get_cash(), 100000.0)

        # After buying
        self.broker.execute_order('AAPL', 10)
        self.assertEqual(self.broker.get_cash(), 99000.0)

        # After selling
        self.broker.execute_order('AAPL', -5)
        self.assertEqual(self.broker.get_cash(), 99500.0)

    def test_get_portfolio_value_no_positions(self):
        """Test getting portfolio value with no positions."""
        # With no positions, portfolio value should equal cash
        self.assertEqual(self.broker.get_portfolio_value(), 100000.0)

    def test_get_portfolio_value_with_positions(self):
        """Test getting portfolio value with positions."""
        # Create some positions
        self.broker.execute_order('AAPL', 10)  # $1000 value
        self.broker.execute_order('MSFT', 15)  # $1500 value

        # Portfolio value should be cash + positions
        expected_value = 99000.0 - 1500.0 + 1000.0 + 1500.0  # Cash + position values
        self.assertEqual(self.broker.get_portfolio_value(), expected_value)

    def test_simulate_price_update(self):
        """Test simulating price updates."""
        # Create a position
        self.broker.execute_order('AAPL', 10)

        # Initial position market value
        initial_market_value = self.broker.positions['AAPL']['market_value']
        self.assertEqual(initial_market_value, 1000.0)  # 10 shares * $100

        # Update the price
        self.broker.simulate_price_update('AAPL', 110.0)

        # Check updated position market value
        updated_market_value = self.broker.positions['AAPL']['market_value']
        self.assertEqual(updated_market_value, 1100.0)  # 10 shares * $110

        # Check updated position current price
        self.assertEqual(self.broker.positions['AAPL']['current_price'], 110.0)

    def test_simulate_price_update_nonexistent_position(self):
        """Test updating price for a non-existent position."""
        # Should not raise an error
        self.broker.simulate_price_update('NONEXISTENT', 150.0)

    def test_reset(self):
        """Test resetting the broker."""
        # Create some activity
        self.broker.execute_order('AAPL', 10)
        self.broker.execute_order('MSFT', 15)

        # Check we have positions
        self.assertEqual(len(self.broker.positions), 2)

        # Check we have orders
        self.assertEqual(len(self.broker.orders), 2)

        # Check cash is reduced
        self.assertEqual(self.broker.cash, 97500.0)

        # Reset the broker
        self.broker.reset()

        # Check cash is back to default
        self.assertEqual(self.broker.cash, 100000.0)

        # Check positions are cleared
        self.assertEqual(len(self.broker.positions), 0)

        # Check orders are cleared
        self.assertEqual(len(self.broker.orders), 0)

        # Check order ID counter is reset
        self.assertEqual(self.broker.next_order_id, 1)

    def test_reset_custom_cash(self):
        """Test resetting the broker with custom cash."""
        # Reset with custom cash
        self.broker.reset(initial_cash=50000.0)

        # Check cash is updated
        self.assertEqual(self.broker.cash, 50000.0)

    # Additional Edge Cases and Complex Scenarios

    def test_buy_sell_cycle_with_price_changes(self):
        """Test a complete buy-sell cycle with price changes in between."""
        # Buy shares
        self.broker.execute_order('AAPL', 10)
        self.assertEqual(self.broker.cash, 99000.0)

        # Simulate price increase
        self.broker.simulate_price_update('AAPL', 120.0)

        # Sell shares at new price - NOTE: In the actual implementation,
        # simulate_price_update only affects portfolio value calculation
        # but not the execution price, which seems to be fixed at 100.0
        self.broker.execute_order('AAPL', -10)

        # Cash should reflect original sale price (not the updated price)
        self.assertEqual(self.broker.cash, 100000.0)  # 99000 + (10 * 100)

        # Position should be gone
        self.assertNotIn('AAPL', self.broker.positions)

    def test_multiple_price_updates(self):
        """Test multiple price updates on the same position."""
        # Buy shares
        self.broker.execute_order('AAPL', 10)

        # Initial market value
        self.assertEqual(self.broker.positions['AAPL']['market_value'], 1000.0)

        # First price update
        self.broker.simulate_price_update('AAPL', 110.0)
        self.assertEqual(self.broker.positions['AAPL']['market_value'], 1100.0)

        # Second price update
        self.broker.simulate_price_update('AAPL', 90.0)
        self.assertEqual(self.broker.positions['AAPL']['market_value'], 900.0)

        # Third price update
        self.broker.simulate_price_update('AAPL', 105.0)
        self.assertEqual(self.broker.positions['AAPL']['market_value'], 1050.0)

    def test_market_open_close_cycle(self):
        """Test transitioning between market open and closed states."""
        # Market starts open
        self.assertTrue(self.broker.check_market_status())

        # Buy position
        self.broker.execute_order('AAPL', 10)
        self.assertIn('AAPL', self.broker.positions)

        # Close market
        self.broker.set_market_status(False)
        self.assertFalse(self.broker.check_market_status())

        # Try to buy more (should fail)
        result = self.broker.execute_order('MSFT', 10)
        self.assertFalse(result)
        self.assertNotIn('MSFT', self.broker.positions)

        # Try to sell existing (should fail)
        result = self.broker.execute_order('AAPL', -5)
        self.assertFalse(result)
        self.assertEqual(self.broker.positions['AAPL']['qty'], 10)

        # Re-open market
        self.broker.set_market_status(True)
        self.assertTrue(self.broker.check_market_status())

        # Now we can sell
        result = self.broker.execute_order('AAPL', -5)
        self.assertTrue(result)
        self.assertEqual(self.broker.positions['AAPL']['qty'], 5)

    def test_zero_quantity_order(self):
        """Test handling of zero quantity orders."""
        # Try to execute a zero quantity order
        result = self.broker.execute_order('AAPL', 0)

        # In the actual implementation, zero quantity orders are rejected
        self.assertFalse(result)

        # Nothing should change
        self.assertEqual(self.broker.cash, 100000.0)
        self.assertNotIn('AAPL', self.broker.positions)

    def test_buy_multiple_symbols(self):
        """Test buying positions in multiple symbols."""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

        # Buy positions in all symbols
        for symbol in symbols:
            self.broker.execute_order(symbol, 10)

        # Check all positions were created
        for symbol in symbols:
            self.assertIn(symbol, self.broker.positions)
            self.assertEqual(self.broker.positions[symbol]['qty'], 10)

        # Check cash was reduced correctly
        self.assertEqual(self.broker.cash, 95000.0)  # 100000 - (5 * 10 * 100)

        # Check portfolio value
        self.assertEqual(self.broker.get_portfolio_value(), 100000.0)

    def test_multi_order_scenario(self):
        """Test a complex multi-order scenario with buys and sells."""
        # Buy initial positions
        self.broker.execute_order('AAPL', 20)
        self.broker.execute_order('MSFT', 15)

        # Simulate market movements
        self.broker.simulate_price_update('AAPL', 110.0)
        self.broker.simulate_price_update('MSFT', 95.0)

        # Rebalance positions - NOTE: In the actual implementation,
        # execute_order uses a fixed price of 100.0 regardless of price updates
        self.broker.execute_order('AAPL', -5)  # Reduce Apple
        self.broker.execute_order('MSFT', 5)  # Increase Microsoft

        # Check positions
        self.assertEqual(self.broker.positions['AAPL']['qty'], 15)
        self.assertEqual(self.broker.positions['MSFT']['qty'], 20)

        # Check cash - using fixed execution price of 100.0 for all trades
        # Initial: 100000
        # Buy AAPL 20 @ 100: -2000 = 98000
        # Buy MSFT 15 @ 100: -1500 = 96500
        # Sell AAPL 5 @ 100: +500 = 97000
        # Buy MSFT 5 @ 100: -500 = 96500
        self.assertEqual(self.broker.cash, 96500.0)

        # Simulate market crash
        self.broker.simulate_price_update('AAPL', 60.0)
        self.broker.simulate_price_update('MSFT', 50.0)

        # Check portfolio value - Updated prices ARE used for portfolio value
        # Cash: 96500
        # AAPL: 15 * 60 = 900
        # MSFT: 20 * 50 = 1000
        # Total: 96500 + 900 + 1000 = 98400
        self.assertEqual(self.broker.get_portfolio_value(), 98400.0)

        # Close all positions - will use the updated prices for closing
        self.broker.close_all_positions()

        # Check all positions are closed
        self.assertEqual(len(self.broker.positions), 0)

        # In the actual implementation, closing positions adds their current
        # market value based on the updated prices to cash
        # Previous cash: 96500
        # AAPL: 15 * 60 = 900
        # MSFT: 20 * 50 = 1000
        # New cash: 96500 + 900 + 1000 = 98400
        self.assertEqual(self.broker.cash, 98400.0)


if __name__ == "__main__":
    unittest.main() 