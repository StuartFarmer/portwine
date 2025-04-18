import pandas as pd
import numpy as np
from portwine.loaders.base import MarketDataLoader


class DailyToOpenCloseLoader(MarketDataLoader):
    """
    Wraps a daily OHLCV loader and emits two intraday bars per day:
      - 09:30 bar: OHLC all set to the daily open, volume = 0
      - 16:00 bar: OHLC all set to the daily close, volume = daily volume
    """
    def __init__(self, base_loader: MarketDataLoader):
        self.base_loader = base_loader
        super().__init__()

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        # 1) Load the daily OHLCV from the base loader
        df_daily = self.base_loader.load_ticker(ticker)
        if df_daily is None or df_daily.empty:
            return df_daily

        # 2) Ensure the index is datetime
        df_daily = df_daily.copy()
        df_daily.index = pd.to_datetime(df_daily.index)

        # 3) Build intraday records
        records = []
        for date, row in zip(df_daily.index, df_daily.itertuples()):
            # 09:30 bar using the daily open
            ts_open = date.replace(hour=9, minute=30)
            records.append({
                'timestamp': ts_open,
                'open':  row.open,
                'high':  row.open,
                'low':   row.open,
                'close': row.open,
                'volume': 0
            })
            # 16:00 bar using the daily close
            ts_close = date.replace(hour=16, minute=0)
            records.append({
                'timestamp': ts_close,
                'open':  row.close,
                'high':  row.close,
                'low':   row.close,
                'close': row.close,
                'volume': getattr(row, 'volume', np.nan)
            })

        # 4) Assemble into a DataFrame with a DatetimeIndex
        df_intraday = (
            pd.DataFrame.from_records(records)
              .set_index('timestamp')
              .sort_index()
        )

        return df_intraday
