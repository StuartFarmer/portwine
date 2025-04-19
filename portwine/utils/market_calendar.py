"""
Market Calendar Utilities

This module provides classes and utilities for working with market calendar
data, including market status information.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MarketStatus:
    """
    Class representing the current market status.
    
    Attributes:
        is_open (bool): Whether the market is currently open
        next_open (Optional[datetime]): The next market open time
        next_close (Optional[datetime]): The next market close time
    """
    is_open: bool
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None 