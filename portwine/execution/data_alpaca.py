"""
Live Alpaca data loader implementing the *standard* `.next` API.
"""
from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
import alpaca_trade_api as tradeapi

from .data_live_base import LiveMarketDataLoader


class AlpacaLiveDataLoader(LiveMarketDataLoader):
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = "https://data.alpaca.markets",
    ):
        self._api = tradeapi.REST(
            api_key or os.getenv("APCA_API_KEY_ID"),
            api_secret or os.getenv("APCA_API_SECRET_KEY"),
            base_url,
            api_version="v2",
        )
        super().__init__()

    # ------------------------------------------------------------------ #
    # Live method
    # ------------------------------------------------------------------ #
    def next(self, tickers: List[str], ts: pd.Timestamp) -> Dict[str, dict | None]:
        # Alpaca snapshot endpoint (bulk)
        snaps = self._api.get_snapshots(",".join(tickers))
        bar_dict: Dict[str, dict | None] = {}

        for sym in tickers:
            snap = snaps.get(sym)
            if snap and snap.daily_bar:
                dbar = snap.daily_bar
                bar_dict[sym] = {
                    "open": dbar.o,
                    "high": dbar.h,
                    "low": dbar.l,
                    "close": dbar.c,
                    "volume": dbar.v,
                }
            else:
                bar_dict[sym] = None              # Missing
        return bar_dict
