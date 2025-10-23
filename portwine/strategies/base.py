from typing import Union, List, Set
from portwine.universe import Universe
from datetime import date
import abc
import numpy as np

class StrategyBase(abc.ABC):
    """
    Base class for a trading strategy. Subclass this to implement a custom strategy.

    A 'step' method is called each day with that day's data. The method should return
    a dictionary of signals/weights for each ticker on that day.

    The strategy always uses a universe object internally. If you pass a list of tickers,
    it creates a static universe with those tickers from 1970-01-01 onwards.
    """

    def __init__(self, tickers: Union[List[str], Universe]):
        """
        Parameters
        ----------
        tickers : Union[List[str], Universe]
            Either a list of ticker symbols or a Universe object.
            If a list is provided, it creates a static universe with those tickers.
        """
        # Initialize universe: if list, treat as static universe; if Universe, use directly
        if isinstance(tickers, Universe):
            self.universe = tickers
        else:
            self.universe = self._create_static_universe(tickers)

    @property
    def tickers(self) -> List[str]:
        """
        The tickers that the strategy is currently using.
        """
        # Always return a sorted list of current universe tickers
        return self.universe.tickers

    def _create_static_universe(self, tickers: List[str]) -> Universe:
        """
        Create a static universe from a list of tickers.
        
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols
            
        Returns
        -------
        Universe
            Static universe with tickers from 1970-01-01 onwards
        """
        # Remove duplicates and convert to set
        unique_tickers = set(tickers)
        
        # Create static universe mapping with numpy datetime64
        constituents = {np.datetime64("1970-01-01"): unique_tickers}
        
        return Universe(constituents)

    def step(self, current_date, daily_data):
        """
        Called each day with that day's data for each ticker.

        Parameters
        ----------
        current_date : pd.Timestamp
        daily_data : dict
            daily_data[ticker] = {
                'open': ..., 'high': ..., 'low': ...,
                'close': ..., 'volume': ...
            }
            or None if no data for that ticker on this date.

            The backtester ensures that daily_data only contains tickers
            that are currently in the universe.

        Returns
        -------
        signals : dict
            { ticker -> float weight }, where the weights are the fraction
            of capital allocated to each ticker (long/short).
        """
        ...
