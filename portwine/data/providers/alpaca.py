from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from .base import DataProvider


class AlpacaProvider(DataProvider):
    data_url = "https://data.alpaca.markets"

    def __init__(self, api_key: str, api_secret: str, *, client: Optional[httpx.Client] = None, data_url: Optional[str] = None):
        if not api_key or not api_secret:
            raise ValueError("Alpaca API key and secret are required")
        self.api_key = api_key
        self.api_secret = api_secret
        if data_url is not None:
            self.data_url = data_url
        self._client = client or httpx.Client(
            base_url=self.data_url,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
        )

    def get_data(self, identifier: str, start_date: datetime, end_date: Optional[datetime] = None):
        if end_date is None:
            end_date = start_date

        params = {
            "symbols": identifier,
            "timeframe": "1Day",
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "limit": 10000,
            "adjustment": "raw",
        }
        resp = self._client.get("/v2/stocks/bars", params=params)
        resp.raise_for_status()
        payload = resp.json()

        bars = payload.get("bars", {}).get(identifier, [])
        normalized: Dict[datetime, Dict[str, Any]] = {}
        for row in bars:
            dt = datetime.fromisoformat(row["t"])
            normalized[dt] = {
                "open": float(row.get("o", 0.0)),
                "high": float(row.get("h", 0.0)),
                "low": float(row.get("l", 0.0)),
                "close": float(row.get("c", 0.0)),
                "volume": float(row.get("v", 0.0)),
            }
        return normalized


