"""
Abstract live data loader – same public interface as any other
MarketDataLoader: implement `.next(tickers, ts)` and (optionally) `.load_ticker`.

`next` must return::

    {
        "AAPL": {"open": ..., "high": ..., "low": ..., "close": ..., "volume": ...},
        "MSFT": None,     # if no fresh bar available
    }
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
from portwine.loaders.base import MarketDataLoader


class LiveMarketDataLoader(MarketDataLoader, ABC):
    """
    Sub‑class `MarketDataLoader` so it can be passed anywhere a historical loader
    is expected.  Historical helper methods (fetch_data, get_all_dates, …) still
    work, but for live execution only `.next` is required.
    """

    # If a subclass does NOT support historical `load_ticker`, leave it unimplemented.
    def load_ticker(self, ticker: str):  # noqa: D401, pylint: disable=unused-argument
        """Live loaders have no local cache by default."""
        return None

    # ------------------------------------------------------------------ #
    # Mandatory live method
    # ------------------------------------------------------------------ #
    @abstractmethod
    def next(
        self,
        tickers: List[str],
        ts: pd.Timestamp,
    ) -> Dict[str, dict | None]:
        """
        Return a bar *at or before* ``ts`` for each requested ticker, exactly
        like the historical implementation – but normally `ts` will be “now”.
        """
