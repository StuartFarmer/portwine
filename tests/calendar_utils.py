import pandas as pd
import numpy as np

from portwine.backtester.core import DailyMarketCalendar


class TestDailyMarketCalendar(DailyMarketCalendar):
    """
    A configurable test calendar that mirrors the production interface while
    allowing tests to control which days are included and the time-of-day.

    Parameters
    - calendar_name: kept for interface parity; not used
    - mode: 'all' or 'odd' (include all calendar days or only odd-numbered days)
    - allowed_year: if a requested range exceeds this year, raise ValueError
    - default_start/default_end: fallback dates when None is provided
    - default_hour:
        * None: inherit time-of-day from provided start_date (default 00:00)
        * int: force this hour for all returned datetimes (e.g., 16)
    """

    def __init__(
        self,
        calendar_name: str = "TEST",
        mode: str = "all",
        allowed_year: int = 2020,
        default_start: str = "2020-01-01",
        default_end: str = "2020-01-10",
        default_hour: int | None = None,
    ):
        # Do not call the real MarketCalendar; use self as the calendar backend
        self.calendar_name = calendar_name
        self.calendar = self

        self.mode = mode
        self.allowed_year = allowed_year
        self.default_start = pd.Timestamp(default_start)
        self.default_end = pd.Timestamp(default_end)
        self.default_hour = default_hour

    @staticmethod
    def _normalize_range(start_date, end_date):
        sd = pd.Timestamp(start_date) if start_date is not None else None
        ed = pd.Timestamp(end_date) if end_date is not None else None
        return sd, ed

    def _coerce_range(self, start_date, end_date):
        sd, ed = self._normalize_range(start_date, end_date)
        if sd is None:
            sd = self.default_start
        if ed is None:
            ed = self.default_end
        if sd > ed:
            raise ValueError("start_date must be on or before end_date")
        if sd.year > self.allowed_year or ed.year > self.allowed_year:
            raise ValueError("No trading days found in the specified date range")
        return sd, ed

    def _select_days(self, sd: pd.Timestamp, ed: pd.Timestamp):
        days = pd.date_range(sd, ed, freq="D")
        if self.mode == "odd":
            selected = [d for d in days if d.day % 2 == 1]
        else:
            selected = list(days)
        return selected

    def _apply_time_of_day(self, base_dates: list[pd.Timestamp], sd: pd.Timestamp):
        if self.default_hour is not None:
            # Force a fixed hour for all outputs
            return [pd.Timestamp(d.date()) + pd.Timedelta(hours=self.default_hour) for d in base_dates]
        # Inherit time-of-day from the provided start_date (default 00:00)
        time_delta = sd - pd.Timestamp(sd.date())
        return [pd.Timestamp(d.date()) + time_delta for d in base_dates]

    # The production code calls self.calendar.schedule(...)
    def schedule(self, start_date, end_date):
        sd, ed = self._coerce_range(start_date, end_date)
        selected = self._select_days(sd, ed)
        if len(selected) == 0:
            raise ValueError("No trading days found in the specified date range")
        closes = self._apply_time_of_day(selected, sd)
        return pd.DataFrame({"market_close": closes}, index=selected)

    # get_datetime_index should return a numpy array of datetimes
    def get_datetime_index(self, start_date, end_date):
        sd, ed = self._coerce_range(start_date, end_date)
        selected = self._select_days(sd, ed)
        if len(selected) == 0:
            raise ValueError("No trading days found in the specified date range")
        dt_idx = self._apply_time_of_day(selected, sd)
        return np.array(dt_idx)


