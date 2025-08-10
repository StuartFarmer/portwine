from __future__ import annotations

from collections import OrderedDict as OrderedDictType
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np

from .provider import DataProvider
from .store import DataStore


class DataSource:
    """
    Read-through cache that conforms to the DataStore API.

    Behavior:
    - get / get_all: read store first; on miss/partial, fetch from provider, add to store, then serve from store
    - other methods: pass through to the underlying store

    Time handling:
    - Accepts and forwards numpy.datetime64 where possible
    - Underlying store may accept datetime/np.datetime64; conversions are localized there
    """

    def __init__(self, provider: DataProvider, store: DataStore, *, refresh_policy: Any = None):
        self.provider = provider
        self.store = store
        self.refresh_policy = refresh_policy

    # Conform to DataStore API
    def add(self, identifier: str, data: Dict[Union[str, datetime, np.datetime64], Dict[str, Any]]):
        self.store.add(identifier, data)

    def get(self, identifier: str, dt: np.datetime64) -> Optional[Dict[str, Any]]:
        # First try the store
        record = self.store.get(identifier, dt)
        if record is not None:
            return record

        # Miss: fetch from provider, ingest, then serve
        self._fetch_and_ingest(identifier, dt, dt)
        return self.store.get(identifier, dt)

    def get_all(
        self,
        identifier: str,
        start_date: np.datetime64,
        end_date: Optional[np.datetime64] = None,
    ) -> Optional[OrderedDictType[datetime, Dict[str, Any]]]:
        # Attempt to read what's available
        result = self.store.get_all(identifier, start_date, end_date)
        # Determine if we should refresh/fetch: missing entirely or likely partial coverage
        needs_fetch = result is None
        if not needs_fetch:
            # If end_date is None, we cannot know completeness; rely on caller/refresh_policy
            # Otherwise, conservatively fetch again to backfill potential gaps
            if end_date is not None:
                needs_fetch = True

        if needs_fetch:
            self._fetch_and_ingest(identifier, start_date, end_date)
            result = self.store.get_all(identifier, start_date, end_date)

        return result

    def get_latest(self, identifier: str) -> Optional[Dict[str, Any]]:
        return self.store.get_latest(identifier)

    def latest(self, identifier: str) -> Optional[datetime]:
        return self.store.latest(identifier)

    def exists(
        self,
        identifier: str,
        start_date: Optional[np.datetime64] = None,
        end_date: Optional[np.datetime64] = None,
    ) -> bool:
        return self.store.exists(identifier, start_date, end_date)

    def identifiers(self) -> Iterable[str]:
        return self.store.identifiers()

    # Internal helpers
    def _fetch_and_ingest(
        self,
        identifier: str,
        start_date: np.datetime64,
        end_date: Optional[np.datetime64] = None,
    ) -> None:
        """
        Fetch data from provider for the given range and add to the store.

        Provider return formats supported (best-effort):
        - Dict[date_like -> Dict[str, Any]]
        - List[Dict[str, Any]] with one of keys {'date', 'timestamp', 'time'} holding a date-like
        """
        # Convert numpy.datetime64 to python datetime for providers that expect datetime
        def to_py_datetime(x: Union[np.datetime64, datetime]) -> datetime:
            if isinstance(x, np.datetime64):
                # Convert preserving date component
                return datetime.utcfromtimestamp((x.astype('datetime64[s]').astype('int')))
            return x

        start_dt_py = to_py_datetime(start_date) if start_date is not None else None
        end_dt_py = to_py_datetime(end_date) if end_date is not None else None

        data = self.provider.get_data(identifier, start_dt_py, end_dt_py)

        normalized: Dict[Union[str, datetime], Dict[str, Any]] = {}
        if isinstance(data, dict):
            # Expect mapping of date-like -> fields
            for k, v in data.items():
                normalized[k] = v if isinstance(v, dict) else {'value': v}
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                # Try common date keys
                date_key = None
                for candidate in ('date', 'timestamp', 'time', 'dt'):
                    if candidate in item:
                        date_key = candidate
                        break
                if date_key is None:
                    continue
                dt_value = item[date_key]
                fields = {k: v for k, v in item.items() if k != date_key}
                normalized[dt_value] = fields
        else:
            # Unsupported type; nothing to ingest
            return

        if normalized:
            self.store.add(identifier, normalized)


