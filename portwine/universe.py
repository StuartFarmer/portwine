"""
Simple universe management for historical constituents.
"""

from typing import List, Dict, Set
from datetime import date
import numpy as np


class Universe:
    """
    Base universe class for managing historical constituents.
    
    This class provides efficient lookup of constituents at any given date
    using binary search on pre-sorted dates.
    """

    def __init__(self, constituents: Dict[np.datetime64, Set[str]]):
        """
        Initialize universe with constituent mapping.
        
        Parameters
        ----------
        constituents : Dict[np.datetime64, Set[str]]
            Dictionary mapping numpy datetime64 dates to sets of ticker symbols
        """
        self.constituents = constituents
        
        # Pre-sort dates for binary search
        self.sorted_dates = sorted(self.constituents.keys())
        
        # Pre-compute all tickers
        self._all_tickers = self._compute_all_tickers()
        # Add attribute to track current date for dynamic tickers
        self._current_date = None
    
    def get_constituents(self, dt: np.datetime64) -> Set[str]:
        """
        Get the basket for a given date.
        
        Parameters
        ----------
        dt : np.datetime64
            Numpy datetime64 object representing the date to get constituents for
            
        Returns
        -------
        Set[str]
            Set of tickers in the basket at the given date
        """
        # Binary search to find the most recent date <= target_date
        left, right = 0, len(self.sorted_dates) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if self.sorted_dates[mid] <= dt:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        if result == -1:
            return set()
            
        return self.constituents[self.sorted_dates[result]]
    
    def _compute_all_tickers(self) -> set:
        """
        Compute all unique tickers that have ever been in the universe.
        
        Returns
        -------
        set
            Set of all ticker symbols
        """
        all_tickers = set()
        for tickers in self.constituents.values():
            all_tickers.update(tickers)
        return all_tickers
    
    @property
    def all_tickers(self) -> set:
        """
        Get all unique tickers that have ever been in the universe.
        
        Returns
        -------
        set
            Set of all ticker symbols
        """
        return self._all_tickers

    def set_datetime(self, dt: np.datetime64):
        """Set the current datetime for dynamic constituents lookup.
        
        Parameters
        ----------
        dt : np.datetime64
            Numpy datetime64 object representing the current date
        """
        self._current_date = dt

    @property
    def tickers(self) -> List[str]:
        """Return list of tickers for the current date."""
        if self._current_date is None:
            # No date set yet; return complete list of tickers, sorted lexicographically
            return sorted(self._all_tickers)
        # Return tickers for the currently set date, sorted lexicographically
        return sorted(self.get_constituents(self._current_date))

    def __iter__(self):
        return iter(self.tickers)

    def __len__(self):
        return len(self.tickers)


class CSVUniverse(Universe):
    """
    Universe class that loads constituent data from CSV files.
    
    Expected CSV format:
    date,ticker1,ticker2,ticker3,...
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize universe from CSV file.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with format: date,ticker1,ticker2,ticker3,...
        """
        constituents = self._load_from_csv(csv_path)
        super().__init__(constituents)
    
    def _load_from_csv(self, csv_path: str) -> Dict[np.datetime64, Set[str]]:
        """
        Load constituent data from CSV file.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file
            
        Returns
        -------
        Dict[np.datetime64, Set[str]]
            Dictionary mapping numpy datetime64 dates to sets of tickers
        """
        constituents = {}
        
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
                    current_date = np.datetime64(f'{year:04d}-{month:02d}-{day:02d}')
                except ValueError:
                    continue  # Skip invalid dates
                
                # Parse tickers (skip empty ones) and convert to set
                tickers = {ticker.strip() for ticker in parts[1:] if ticker.strip()}
                constituents[current_date] = tickers
        
        return constituents
