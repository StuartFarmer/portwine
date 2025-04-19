"""
Mock broker implementation.

This module provides a simulated broker implementation for testing and development purposes.
It allows for testing trading strategies without connecting to an actual brokerage API.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from portwine.execution_complex.broker_base import BrokerBase

class MockBroker(BrokerBase):
    """
    Mock broker implementation for testing and simulation.
    
    This class provides a simulated broker environment for testing strategies
    and execution logic without connecting to a real trading platform.
    """
    
    def __init__(self, initial_cash: float = 100000.0, market_hours: bool = True):
        """
        Initialize the mock broker with simulated account and market information.
        
        Parameters
        ----------
        initial_cash : float, default 100000.0
            Initial cash balance for the simulated account
        market_hours : bool, default True
            Whether to simulate market as open (True) or closed (False)
        """
        self.logger = logging.getLogger(__name__)
        self.cash = initial_cash
        self.market_open = market_hours
        self.positions = {}  # symbol -> {'qty': float, 'cost_basis': float}
        self.orders = {}  # order_id -> order_details
        self.next_order_id = 1
        
        self.logger.info(f"Initialized MockBroker with ${initial_cash:.2f} initial cash")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information from simulated account.
        
        Returns
        -------
        Dict[str, Any]
            Simulated account information
        """
        portfolio_value = self.cash
        
        # Add value of positions
        for symbol, pos in self.positions.items():
            portfolio_value += pos.get('market_value', 0)
        
        return {
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'buying_power': self.cash * 2,  # Simple 2x margin simulation
            'equity': portfolio_value,
            'status': 'ACTIVE'
        }
    
    def execute_order(self, symbol: str, qty: float, order_type: str = "market") -> bool:
        """
        Execute a simulated trade.
        
        Parameters
        ----------
        symbol : str
            The ticker symbol of the asset to trade
        qty : float
            The quantity to trade (positive for buy, negative for sell)
        order_type : str, default "market"
            The type of order (market, limit, etc.)
            
        Returns
        -------
        bool
            True if order was executed successfully, False otherwise
        """
        if not self.market_open:
            self.logger.warning("Market is closed, order not executed")
            return False
            
        side = "buy" if qty > 0 else "sell"
        abs_qty = abs(qty)
        
        # Simulate a simple price for the symbol (could be enhanced later)
        price = 100.0  # Placeholder price
        order_value = price * abs_qty
        
        # Process buy order
        if side == "buy":
            if self.cash < order_value:
                self.logger.warning(f"Insufficient cash (${self.cash:.2f}) to execute buy order for ${order_value:.2f}")
                return False
                
            # Update cash and positions
            self.cash -= order_value
            
            if symbol in self.positions:
                # Update existing position
                current_pos = self.positions[symbol]
                total_qty = current_pos['qty'] + abs_qty
                new_cost_basis = ((current_pos['cost_basis'] * current_pos['qty']) + order_value) / total_qty
                
                self.positions[symbol] = {
                    'qty': total_qty,
                    'cost_basis': new_cost_basis,
                    'market_value': total_qty * price,
                    'current_price': price
                }
            else:
                # Create new position
                self.positions[symbol] = {
                    'qty': abs_qty,
                    'cost_basis': price,
                    'market_value': abs_qty * price,
                    'current_price': price
                }
        
        # Process sell order
        else:
            if symbol not in self.positions:
                self.logger.warning(f"No position in {symbol} to sell")
                return False
                
            current_pos = self.positions[symbol]
            if current_pos['qty'] < abs_qty:
                self.logger.warning(f"Insufficient shares ({current_pos['qty']}) to sell {abs_qty} of {symbol}")
                return False
                
            # Update cash and positions
            self.cash += order_value
            
            new_qty = current_pos['qty'] - abs_qty
            if new_qty > 0:
                # Update existing position
                self.positions[symbol]['qty'] = new_qty
                self.positions[symbol]['market_value'] = new_qty * price
            else:
                # Remove position completely
                del self.positions[symbol]
        
        # Record the order
        order_id = str(self.next_order_id)
        self.next_order_id += 1
        
        self.orders[order_id] = {
            'id': order_id,
            'symbol': symbol,
            'qty': abs_qty,
            'side': side,
            'type': order_type,
            'status': 'filled',
            'filled_at': datetime.now().isoformat(),
            'filled_price': price,
            'filled_qty': abs_qty
        }
        
        self.logger.info(f"Executed {side} order for {abs_qty} shares of {symbol} at ${price:.2f}")
        return True
    
    def check_market_status(self) -> bool:
        """
        Check simulated market status.
        
        Returns
        -------
        bool
            True if simulated market is open, False otherwise
        """
        return self.market_open
    
    def set_market_status(self, is_open: bool) -> None:
        """
        Set the simulated market status.
        
        Parameters
        ----------
        is_open : bool
            Whether to set the market as open (True) or closed (False)
        """
        self.market_open = is_open
        status = "open" if is_open else "closed"
        self.logger.info(f"Market status set to {status}")
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """
        Get the status of a simulated order.
        
        Parameters
        ----------
        order_id : str
            ID of the order to check
            
        Returns
        -------
        Optional[str]
            Status of the order, or None if the order doesn't exist
        """
        if order_id in self.orders:
            return self.orders[order_id]['status']
        return None
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open simulated orders.
        
        Returns
        -------
        bool
            Always returns True as this is a simulation
        """
        # Find orders that aren't already filled or canceled
        for order_id, order in self.orders.items():
            if order['status'] not in ['filled', 'canceled']:
                order['status'] = 'canceled'
        
        self.logger.info("All orders canceled")
        return True
    
    def close_all_positions(self) -> bool:
        """
        Close all simulated positions.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not self.market_open:
            self.logger.warning("Market is closed, cannot close positions")
            return False
            
        symbols = list(self.positions.keys())
        
        for symbol in symbols:
            position = self.positions[symbol]
            # Execute a sell order for each position
            self.execute_order(symbol, -position['qty'])
        
        self.logger.info("All positions closed")
        return True
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current simulated positions.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of current positions
        """
        result = []
        for symbol, pos in self.positions.items():
            position_data = {
                'symbol': symbol,
                'qty': pos['qty'],
                'market_value': pos['market_value'],
                'cost_basis': pos['cost_basis'],
                'current_price': pos.get('current_price', 100.0),
                'unrealized_pl': pos['market_value'] - (pos['cost_basis'] * pos['qty']),
                'side': 'long'
            }
            result.append(position_data)
        return result
    
    def get_cash(self) -> float:
        """
        Get available cash in simulated account.
        
        Returns
        -------
        float
            Available cash for trading
        """
        return self.cash
    
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value from simulated account.
        
        Returns
        -------
        float
            Total value of the portfolio
        """
        portfolio_value = self.cash
        
        # Add value of positions
        for symbol, pos in self.positions.items():
            portfolio_value += pos.get('market_value', 0)
            
        return portfolio_value
        
    def simulate_price_update(self, symbol: str, new_price: float) -> None:
        """
        Update the simulated price of a symbol.
        
        Parameters
        ----------
        symbol : str
            The ticker symbol to update
        new_price : float
            The new price to set
        """
        if symbol in self.positions:
            qty = self.positions[symbol]['qty']
            self.positions[symbol]['current_price'] = new_price
            self.positions[symbol]['market_value'] = qty * new_price
            self.logger.info(f"Updated {symbol} price to ${new_price:.2f}")
    
    def reset(self, initial_cash: float = 100000.0) -> None:
        """
        Reset the mock broker to initial state.
        
        Parameters
        ----------
        initial_cash : float, default 100000.0
            Initial cash balance to reset to
        """
        self.cash = initial_cash
        self.positions = {}
        self.orders = {}
        self.next_order_id = 1
        self.logger.info(f"Reset MockBroker with ${initial_cash:.2f} initial cash") 