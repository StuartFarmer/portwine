from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from .base import DataProvider


class FREDProvider(DataProvider):
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str, *, client: Optional[httpx.Client] = None, base_url: Optional[str] = None):
        if not api_key:
            raise ValueError("FRED API key is required")
        self.api_key = api_key
        if base_url is not None:
            self.base_url = base_url
        self._client = client or httpx.Client()

    def get_data(self, identifier: str, start_date: datetime, end_date: Optional[datetime] = None):
        params = {
            "series_id": identifier,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date.strftime('%Y-%m-%d'),
        }
        if end_date is not None:
            params["observation_end"] = end_date.strftime('%Y-%m-%d')

        resp = self._client.get(self.base_url, params=params)
        resp.raise_for_status()
        payload = resp.json() or {}
        observations = payload.get("observations", [])

        normalized: Dict[datetime, Dict[str, Any]] = {}
        for obs in observations:
            dt = datetime.strptime(obs["date"], "%Y-%m-%d")
            value_str = obs.get("value", "nan")
            try:
                value = float(value_str)
            except ValueError:
                value = float('nan')
            normalized[dt] = {"close": value, "open": value, "high": value, "low": value, "volume": 0.0}

        return normalized


