"""
Simple universe management for historical constituents.
"""

import pandas as pd
from typing import List
from datetime import date


class Universe:
    """
    Simple universe class for managing historical constituents.
    
    Loads a CSV file with date and basket columns, then provides
    efficient lookup of constituents at any given date.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize universe from CSV file.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with columns: date, basket
            basket should be comma-separated tickers
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Store as dict: date -> list of tickers
        self.constituents = {}
        for _, row in df.iterrows():
            self.constituents[row['date']] = row['basket'].split(',')
        
        # Pre-sort dates for binary search
        self.sorted_dates = sorted(self.constituents.keys())
    
    def get_constituents(self, dt) -> List[str]:
        """
        Get the basket for a given date.
        
        Parameters
        ----------
        dt : datetime-like
            Date to get constituents for
            
        Returns
        -------
        List[str]
            List of tickers in the basket at the given date
        """
        target_date = pd.to_datetime(dt).date()
        
        # Binary search to find the most recent date <= target_date
        left, right = 0, len(self.sorted_dates) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if self.sorted_dates[mid] <= target_date:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        if result == -1:
            return []
            
        return self.constituents[self.sorted_dates[result]]