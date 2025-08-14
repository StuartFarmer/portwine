"""
CSVDataStore - A DataStore implementation that reads OHLCV data from CSV files.

File structure:
data_dir/
├── <identifier_1>.csv
├── <identifier_2>.csv
└── <identifier_3>.csv

Each CSV file is expected to have a header like:

date,open,high,low,close,adjusted_close,volume
2000-01-03,78.75,78.9375,67.3749,71.9999,43.2888,4674353
...

The "date" column will be parsed as datetime and used as the DataFrame index.
"""

from datetime import datetime
from typing import Union
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import numpy as np

from .base import DataStore


class CSVDataStore(DataStore):
    """
    Read-only DataStore for OHLCV data stored as one CSV file per identifier.

    This class focuses on efficient reads and a minimal API surface compatible
    with the DataStore interface used throughout the project.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, identifier: str) -> Path:
        return self.data_dir / f"{identifier}.csv"

    def _load_dataframe(self, identifier: str) -> pd.DataFrame:
        """
        Load a DataFrame from <identifier>.csv.
        - Parses the 'date' column as datetime and sets it as the index
        - Sorts by index ascending
        - Returns an empty DataFrame if file is missing or unreadable
        """
        file_path = self._get_file_path(identifier)
        if not file_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path, parse_dates=["date"], infer_datetime_format=True)
        except Exception:
            return pd.DataFrame()

        if "date" not in df.columns:
            return pd.DataFrame()

        df = df.set_index("date")
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()
        # Normalize index name to None for consistency with parquet behavior
        df.index.name = None
        return df

    def add(self, identifier: str, data: dict, overwrite: bool = False):
        """
        Persist data to CSV.

        While CSV is not ideal for frequent writes, this provides symmetry with
        other stores. The data dict should be: {datetime|str: {field: value, ...}}
        """
        if not data:
            return

        # Existing data if any
        df_existing = self._load_dataframe(identifier)

        new_rows = []
        for dt_like, values in data.items():
            # Accept datetime, np.datetime64, or str
            if isinstance(dt_like, (np.datetime64, str)):
                dt = pd.to_datetime(dt_like)
            else:
                dt = pd.to_datetime(dt_like)
            row_data = {"date": dt}
            if isinstance(values, dict):
                row_data.update(values)
            new_rows.append(row_data)

        if not new_rows:
            return

        df_new = pd.DataFrame(new_rows).set_index("date")

        if df_existing.empty:
            df_to_save = df_new
        else:
            if overwrite:
                df_existing = df_existing.drop(df_new.index, errors="ignore")
                df_to_save = pd.concat([df_existing, df_new])
            else:
                mask_new = ~df_new.index.isin(df_existing.index)
                df_to_save = pd.concat([df_existing, df_new[mask_new]])

        df_to_save = df_to_save.sort_index()
        # Save with a stable 'date' column
        out_path = self._get_file_path(identifier)
        df_to_save.reset_index().rename(columns={"index": "date"}).to_csv(out_path, index=False)

    def get(self, identifier: str, dt: datetime) -> Union[dict, None]:
        df = self._load_dataframe(identifier)
        if df.empty:
            return None

        dt_ts = pd.to_datetime(dt)

        try:
            row = df.loc[dt_ts]
        except KeyError:
            return None

        if isinstance(row, pd.DataFrame):
            return row.iloc[-1].to_dict()
        return row.to_dict()

    def get_all(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        df = self._load_dataframe(identifier)
        if df.empty:
            return None

        start_ts = pd.to_datetime(start_date) if start_date is not None else df.index.min()
        end_ts = pd.to_datetime(end_date) if end_date is not None else df.index.max()

        if start_ts > end_ts:
            return None

        mask = (df.index >= start_ts) & (df.index <= end_ts)
        df_filtered = df.loc[mask]
        if df_filtered.empty:
            return None

        result = OrderedDict()
        for ts, row in df_filtered.iterrows():
            # pandas.Timestamp is a datetime subclass; keep as-is for compatibility
            result[ts] = row.to_dict()
        return result

    def get_latest(self, identifier: str) -> Union[dict, None]:
        df = self._load_dataframe(identifier)
        if df.empty:
            return None
        return df.iloc[-1].to_dict()

    def latest(self, identifier: str) -> Union[datetime, None]:
        df = self._load_dataframe(identifier)
        if df.empty:
            return None
        return df.index.max()

    def exists(self, identifier: str, start_date: Union[datetime, None] = None, end_date: Union[datetime, None] = None) -> bool:
        df = self._load_dataframe(identifier)
        if df.empty:
            return False

        start_ts = pd.to_datetime(start_date) if start_date is not None else df.index.min()
        end_ts = pd.to_datetime(end_date) if end_date is not None else df.index.max()
        if start_ts > end_ts:
            return False

        mask = (df.index >= start_ts) & (df.index <= end_ts)
        return df.loc[mask].shape[0] > 0

    def identifiers(self):
        ids = []
        for file_path in self.data_dir.glob("*.csv"):
            ids.append(file_path.stem)
        return ids


