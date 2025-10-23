from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from .base import DataProvider


class PolygonProvider(DataProvider):
    base_url = "https://api.polygon.io"

    def __init__(self, api_key: str, *, client: Optional[httpx.Client] = None, base_url: Optional[str] = None):
        if not api_key:
            raise ValueError("Polygon API key is required")
        self.api_key = api_key
        if base_url is not None:
            self.base_url = base_url
        self._client = client or httpx.Client(
            base_url=self.base_url, headers={"Authorization": f"Bearer {api_key}"}
        )

    def _range_url(self, ticker: str, start_ms: int, end_ms: int) -> str:
        return f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_ms}/{end_ms}"

    def _to_utc_ms(self, dt: datetime) -> int:
        # Robust UTC ms calculation using epoch subtraction to avoid local tz quirks
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if dt.tzinfo is None:
            dt_utc = dt.replace(tzinfo=timezone.utc)
        else:
            dt_utc = dt.astimezone(timezone.utc)
        return int((dt_utc - epoch).total_seconds() * 1000)

    def get_data(self, identifier: str, start_date: datetime, end_date: Optional[datetime] = None):
        if end_date is None:
            end_date = start_date

        start_ms = self._to_utc_ms(start_date)
        end_ms = self._to_utc_ms(end_date)
        url = self._range_url(identifier, start_ms, end_ms)

        results: list[Dict[str, Any]] = []
        next_url: Optional[str] = url
        params = {"adjusted": "true", "sort": "asc"}

        while next_url:
            # If Polygon returns a fully-qualified next_url, don't pass params again
            request_params = None if next_url.startswith("http") else params
            resp = self._client.get(next_url, params=request_params)
            resp.raise_for_status()
            payload = resp.json()
            if payload and payload.get("results"):
                results.extend(payload["results"])
                next_url = payload.get("next_url")
            else:
                break

        # Normalize to date->dict mapping using polygon result fields
        normalized: Dict[datetime, Dict[str, Any]] = {}
        for row in results:
            ts = int(row.get("t"))
            dt = datetime.utcfromtimestamp(ts / 1000)
            normalized[dt] = {
                "open": float(row.get("o", 0.0)),
                "high": float(row.get("h", 0.0)),
                "low": float(row.get("l", 0.0)),
                "close": float(row.get("c", 0.0)),
                "volume": float(row.get("v", 0.0)),
            }

        return normalized


