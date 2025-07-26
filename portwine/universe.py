"""
Simple universe management for historical constituents.
"""

from typing import List
from datetime import date


class Universe:
    """
    Simple universe class for managing historical constituents.
    
    Loads a CSV file with format: date,ticker1,ticker2,ticker3,...
    then provides efficient lookup of constituents at any given date.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize universe from CSV file.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with format: date,ticker1,ticker2,ticker3,...
        """
        # Load CSV with raw file I/O
        self.constituents = {}
        
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                    
                # Parse date
                date_str = parts[0].strip()
                try:
                    year, month, day = map(int, date_str.split('-'))
                    current_date = date(year, month, day)
                except ValueError:
                    continue  # Skip invalid dates
                
                # Parse tickers (skip empty ones)
                tickers = [ticker.strip() for ticker in parts[1:] if ticker.strip()]
                self.constituents[current_date] = tickers
        
        # Pre-sort dates for binary search
        self.sorted_dates = sorted(self.constituents.keys())

        self.all_tickers = self._all_tickers()
    
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
        # Convert to date object
        if hasattr(dt, 'date'):
            target_date = dt.date()
        else:
            target_date = date.fromisoformat(str(dt).split()[0])
        
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
    
    def _all_tickers(self) -> set:
        """
        Get all unique tickers that have ever been in the universe.
        
        Returns
        -------
        set
            Set of all ticker symbols
        """
        all_tickers = set()
        for tickers in self.constituents.values():
            all_tickers.update(tickers)
        return all_tickers