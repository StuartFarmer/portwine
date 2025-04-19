"""
Base broker interface.

This module defines the abstract base class that all broker implementations must follow.
A broker is responsible for executing orders, managing positions, and providing account information.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BrokerBase(ABC):
    """
    Abstract base class defining the interface for all broker implementations.
    
    A broker is responsible for:
    - Executing trading orders
    - Managing positions
    - Providing account information
    - Interfacing with trading platforms or exchanges
    
    All broker implementations must implement these methods to provide
    a consistent interface for the execution system.
    """
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information from the broker.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing account information such as cash, equity, etc.
        """
        pass
    
    @abstractmethod
    def execute_order(self, symbol: str, qty: float, order_type: str = "market") -> bool:
        """
        Execute a trade order.
        
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
        pass
    
    @abstractmethod
    def check_market_status(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns
        -------
        bool
            True if market is open, False otherwise
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[str]:
        """
        Get the status of a specific order.
        
        Parameters
        ----------
        order_id : str
            ID of the order to check
            
        Returns
        -------
        Optional[str]
            Status of the order, or None if the order doesn't exist
        """
        pass
    
    @abstractmethod
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def close_all_positions(self) -> bool:
        """
        Close all open positions.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of current positions
        """
        pass
    
    @abstractmethod
    def get_cash(self) -> float:
        """
        Get available cash in account.
        
        Returns
        -------
        float
            Available cash for trading
        """
        pass
    
    @abstractmethod
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value.
        
        Returns
        -------
        float
            Total value of the portfolio
        """
        pass 