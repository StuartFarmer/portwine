from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import Dict, Union, Iterable

import pandas as pd

from portwine.data.stores.base import DataStore


class MockDataStore(DataStore):
    """
    In-memory DataStore implementation for tests.

    - Supports loading from pandas DataFrames, dicts of datetime->dict, or
      dicts of field->array with an inferred date range.
    - Provides optional default_ohlcv fallback returned when data is missing
      for a requested identifier/datetime.
    """

    def __init__(self, default_ohlcv: Dict[str, float] | None = None, default_start: str = "2020-01-01"):
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._default_ohlcv = default_ohlcv
        self._default_start = default_start

    # Loading helpers -----------------------------------------------------
    def load_dataframe(self, identifier: str, df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        self._dataframes[identifier] = df.sort_index()

    def load_date_dict(self, identifier: str, data_by_dt: Dict[Union[str, datetime, pd.Timestamp], Dict]) -> None:
        if not data_by_dt:
            return
        rows = []
        for dt, payload in data_by_dt.items():
            dt_ts = pd.to_datetime(dt)
            rows.append({"__dt__": dt_ts, **(payload or {})})
        df = pd.DataFrame(rows).set_index("__dt__").sort_index()
        self._dataframes[identifier] = df

    def load_arrays(self, identifier: str, arrays: Dict[str, Iterable], start: str | None = None) -> None:
        # Infer length from any array; assume daily frequency
        length = None
        for v in arrays.values():
            try:
                length = len(v)
                break
            except TypeError:
                continue
        if length is None:
            return
        start_str = start or self._default_start
        idx = pd.date_range(start=start_str, periods=length, freq="D")
        df = pd.DataFrame(arrays, index=idx)
        self._dataframes[identifier] = df

    def load_bulk(self, data: Dict[str, Union[pd.DataFrame, Dict, None]], *, default_start: str | None = None) -> None:
        for identifier, value in (data or {}).items():
            if isinstance(value, pd.DataFrame):
                self.load_dataframe(identifier, value)
            elif isinstance(value, dict):
                # Heuristic: dict of datetime->payload vs field->array vs single payload (non-time-series)
                if value and all(isinstance(k, (str, datetime, pd.Timestamp)) for k in value.keys()):
                    # If keys look like datetimes, treat as time-series mapping
                    try:
                        pd.to_datetime(next(iter(value.keys())))
                        self.load_date_dict(identifier, value)
                        continue
                    except Exception:
                        pass
                # If any values are iterables of equal length -> arrays per field
                try:
                    self.load_arrays(identifier, value, start=default_start)
                except Exception:
                    # Fallback: store a single-row time-series at default start treating entire dict as payload
                    self.load_date_dict(identifier, {default_start or self._default_start: value})
            elif value is None:
                # No data; skip
                continue
            else:
                # Unsupported type; store a single row at default start
                self.load_date_dict(identifier, {default_start or self._default_start: value})

    # DataStore API -------------------------------------------------------
    def add(self, identifier: str, data: dict):
        # Merge via bulk loader
        if isinstance(data, dict):
            # If dict-of-dates, convert; else assume field->value single row
            if data and all(isinstance(k, (str, datetime, pd.Timestamp)) for k in data.keys()):
                self.load_date_dict(identifier, data)
            else:
                self.load_date_dict(identifier, {self._default_start: data})

    def get(self, identifier: str, dt: datetime) -> Union[dict, None]:
        df = self._dataframes.get(identifier)
        if df is None or df.empty:
            return self._default_ohlcv.copy() if self._default_ohlcv else None
        ts = pd.to_datetime(dt)
        # Try exact match first
        try:
            row = df.loc[ts]
            return row.to_dict()
        except KeyError:
            pass
        # Try nearest at-or-before timestamp
        try:
            mask = df.index <= ts
            if mask.any():
                latest_idx = df.index[mask].max()
                return df.loc[latest_idx].to_dict()
        except Exception:
            pass
        # Try matching on date regardless of time component
        try:
            date_only = pd.Timestamp(ts.date())
            row = df.loc[date_only]
            return row.to_dict()
        except Exception:
            pass
        return self._default_ohlcv.copy() if self._default_ohlcv else None

    def get_all(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        df = self._dataframes.get(identifier)
        if df is None or df.empty:
            return None
        if end_date is None:
            end_date = df.index.max()
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        df_filtered = df[mask]
        if df_filtered.empty:
            return None
        result = OrderedDict()
        for ts, row in df_filtered.iterrows():
            result[ts] = row.to_dict()
        return result

    def get_latest(self, identifier: str):
        df = self._dataframes.get(identifier)
        if df is None or df.empty:
            return None
        return df.iloc[-1].to_dict()

    def latest(self, identifier: str):
        df = self._dataframes.get(identifier)
        if df is None or df.empty:
            return None
        return df.index.max().to_pydatetime()

    def exists(self, identifier: str, start_date: Union[datetime, None] = None, end_date: Union[datetime, None] = None) -> bool:
        df = self._dataframes.get(identifier)
        if df is None or df.empty:
            return False
        if start_date is None and end_date is None:
            return True
        if start_date is None:
            start_date = df.index.min()
        if end_date is None:
            end_date = df.index.max()
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        return mask.any()

    def identifiers(self):
        return list(self._dataframes.keys())


