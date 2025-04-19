"""Utility functions for execution systems.

This module contains functions that help with execution-related tasks 
like creating bar dictionaries, calculating position changes, and generating orders.
"""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone


def create_bar_dict(df: pd.DataFrame, timestamp: datetime) -> Dict[str, Dict[str, Any]]:
    """Create a dictionary of bar data from a DataFrame.
    
    Args:
        df: DataFrame containing OHLCV data for multiple symbols
        timestamp: The timestamp to use for the bars
        
    Returns:
        Dict mapping symbol to bar data
    """
    if df is None or df.empty:
        return {}
        
    # Ensure timestamp is timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
        
    # Create bar dictionary by symbol
    bar_dict = {}
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].iloc[0]
        
        bar_dict[symbol] = {
            'timestamp': timestamp,
            'open': float(symbol_data.get('open', 0)),
            'high': float(symbol_data.get('high', 0)),
            'low': float(symbol_data.get('low', 0)),
            'close': float(symbol_data.get('close', 0)),
            'volume': float(symbol_data.get('volume', 0))
        }
    
    return bar_dict


def calculate_position_changes(
    target_positions: Dict[str, float], 
    current_positions: Dict[str, float]
) -> Dict[str, float]:
    """Calculate the position changes needed to reach target positions.
    
    Args:
        target_positions: Dictionary mapping symbol to target position quantity
        current_positions: Dictionary mapping symbol to current position quantity
        
    Returns:
        Dictionary mapping symbol to position change (positive for buy, negative for sell)
    """
    if not target_positions:
        # Close all positions if no target positions provided
        return {symbol: -qty for symbol, qty in current_positions.items() if qty != 0}
        
    # Get all unique symbols from both dictionaries
    all_symbols = set(target_positions.keys()).union(set(current_positions.keys()))
    
    # Calculate position changes
    position_changes = {}
    for symbol in all_symbols:
        target = target_positions.get(symbol, 0)
        current = current_positions.get(symbol, 0)
        change = target - current
        
        # Only include non-zero changes
        if abs(change) > 1e-6:  # Using small epsilon for float comparison
            position_changes[symbol] = change
            
    return position_changes


def generate_orders(
    position_changes: Dict[str, float],
    order_type: str = "market",
    limit_prices: Optional[Dict[str, float]] = None,
    time_in_force: str = "day"
) -> List[Dict[str, Any]]:
    """Generate orders from position changes.
    
    Args:
        position_changes: Dictionary mapping symbol to position change
        order_type: Order type to use (market, limit, etc.)
        limit_prices: Dictionary mapping symbol to limit price (for limit orders)
        time_in_force: Time in force parameter (day, gtc, etc.)
        
    Returns:
        List of order dictionaries
    """
    if limit_prices is None:
        limit_prices = {}
        
    orders = []
    for symbol, qty in position_changes.items():
        if abs(qty) < 1e-6:  # Skip very small orders
            continue
            
        # Round quantity to nearest whole number
        # (Fractional shares could be implemented differently depending on broker)
        qty_rounded = round(qty)
        if qty_rounded == 0:
            continue
            
        order = {
            "symbol": symbol,
            "qty": qty_rounded,
            "order_type": order_type,
            "time_in_force": time_in_force
        }
        
        # Add limit price if available and order type is limit
        if order_type in ["limit", "stop_limit"] and symbol in limit_prices:
            order["limit_price"] = limit_prices[symbol]
            
        orders.append(order)
        
    return orders 