from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from .base import DataProvider


class EODHDProvider(DataProvider):
    base_url = "https://eodhd.com/api"

    def __init__(self, api_key: str, exchange_code: str = "US", *, client: Optional[httpx.Client] = None, base_url: Optional[str] = None):
        if not api_key:
            raise ValueError("EODHD API key is required")
        self.api_key = api_key
        self.exchange_code = exchange_code
        if base_url is not None:
            self.base_url = base_url
        self._client = client or httpx.Client(base_url=self.base_url)

    def _build_url(self, identifier: str, start_date: datetime, end_date: Optional[datetime]):
        url = f"/eod/{identifier}.{self.exchange_code}?api_token={self.api_key}&fmt=json"
        if start_date is not None:
            url += f"&from={start_date.strftime('%Y-%m-%d')}"
        if end_date is not None:
            url += f"&to={end_date.strftime('%Y-%m-%d')}"
        return url

    def get_data(self, identifier: str, start_date: datetime, end_date: Optional[datetime] = None):
        url = self._build_url(identifier, start_date, end_date)
        resp = self._client.get(url)
        resp.raise_for_status()
        data = resp.json() or []

        normalized: Dict[datetime, Dict[str, Any]] = {}
        for row in data:
            dt = datetime.strptime(row["date"], "%Y-%m-%d")
            normalized[dt] = {
                "open": float(row.get("open", 0.0)),
                "high": float(row.get("high", 0.0)),
                "low": float(row.get("low", 0.0)),
                "close": float(row.get("adjusted_close", row.get("close", 0.0))),
                "volume": float(row.get("volume", 0.0)),
            }
        return normalized


