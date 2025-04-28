import pandas as pd
import pandas_market_calendars as mcal
import time
from datetime import datetime, date, timedelta, timezone
from typing import Iterator, Optional, List, Union


class DailySchedule(Iterator[int]):
    """
    Iterator of UNIX-ms timestamps for market events.

    Modes:
      - Finite: if end_date is set, yields events between start_date and end_date inclusive.
      - Live: if end_date is None, yields all future events from start_date (or today) onward, indefinitely.
    """

    def __init__(
        self,
        *,
        after_open_minutes: Optional[int] = None,
        before_close_minutes: Optional[int] = None,
        calendar_name: str = "NYSE",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        inclusive: bool = False,
    ):
        # Validate offsets
        if after_open_minutes is None and before_close_minutes is None:
            raise ValueError("Must specify after_open_minutes or before_close_minutes")
        if before_close_minutes is not None and interval_seconds is not None and after_open_minutes is None:
            raise ValueError("Cannot specify interval_seconds with close-only schedule")

        self.after_open = after_open_minutes
        self.before_close = before_close_minutes
        self.interval = interval_seconds
        self.inclusive = inclusive
        self.calendar = mcal.get_calendar(calendar_name)
        self.start_date = start_date
        self.end_date = end_date
        self._gen = None

    def __iter__(self):
        if self.end_date is not None:
            self._gen = self._finite_generator()
        else:
            self._gen = self._live_generator(self.start_date)
        return self

    def __next__(self) -> int:
        return next(self._gen)

    def _to_ms(self, ev: Union[datetime, pd.Timestamp]) -> int:
        # Normalize naive → UTC, leave tz-aware alone
        if isinstance(ev, pd.Timestamp):
            if ev.tzinfo is None:
                ev = ev.tz_localize("UTC")
        else:
            if ev.tzinfo is None:
                ev = ev.replace(tzinfo=timezone.utc)
        return int(ev.timestamp() * 1000)

    def _build_events(
        self,
        open_dt: Union[datetime, pd.Timestamp],
        close_dt: Union[datetime, pd.Timestamp],
    ) -> List[Union[datetime, pd.Timestamp]]:
        # Localize bounds
        if isinstance(open_dt, pd.Timestamp):
            if open_dt.tzinfo is None:
                open_dt = open_dt.tz_localize("UTC")
        else:
            if open_dt.tzinfo is None:
                open_dt = open_dt.replace(tzinfo=timezone.utc)

        if isinstance(close_dt, pd.Timestamp):
            if close_dt.tzinfo is None:
                close_dt = close_dt.tz_localize("UTC")
        else:
            if close_dt.tzinfo is None:
                close_dt = close_dt.replace(tzinfo=timezone.utc)

        # Compute window
        start_dt = open_dt if self.after_open is None else open_dt + timedelta(minutes=self.after_open)
        end_dt = close_dt if self.before_close is None else close_dt - timedelta(minutes=self.before_close)

        # Interval validation
        if self.interval is not None:
            window_secs = (end_dt - start_dt).total_seconds()
            if self.interval > window_secs:
                raise ValueError(f"interval_seconds={self.interval} exceeds session window of {window_secs:.0f}s")

        # Build events
        if self.after_open is None:
            # close-only
            return [end_dt]

        if self.before_close is None:
            # open-only or open+interval
            if self.interval is None:
                return [start_dt]
            events: List[Union[datetime, pd.Timestamp]] = []
            t = start_dt
            while t <= close_dt:
                events.append(t)
                t += timedelta(seconds=self.interval)
            return events

        # both open+close, with optional interval
        if self.interval is None:
            return [start_dt, end_dt]
        events = []
        t = start_dt
        last = None
        while t <= end_dt:
            events.append(t)
            last = t
            t += timedelta(seconds=self.interval)
        if self.inclusive and last and last < end_dt:
            events.append(end_dt)
        return events

    def _finite_generator(self):
        sched = self.calendar.schedule(start_date=self.start_date, end_date=self.end_date)
        for _, row in sched.iterrows():
            for ev in self._build_events(row["market_open"], row["market_close"]):
                yield self._to_ms(ev)

    def _live_generator(self, start_date=None):
        # If start_date is None, use today
        if start_date is not None:
            current_date = pd.Timestamp(start_date).date()
        else:
            current_date = date.today()
        # 2) determine tz from calendar
        today_str = date.today().isoformat()
        try:
            today_sched = self.calendar.schedule(start_date=today_str, end_date=today_str)
        except StopIteration:
            return
        tz = getattr(today_sched.index, "tz", None)
        # 3) current time from time.time()
        now_sec = time.time()
        now_ts = pd.Timestamp(now_sec, unit="s", tz=tz) if tz else pd.Timestamp(now_sec, unit="s")
        now_ms = int(now_ts.timestamp() * 1000)
        # 4) start looping from current_date
        while True:
            day_str = current_date.isoformat()
            try:
                day_sched = self.calendar.schedule(start_date=day_str, end_date=day_str)
            except StopIteration:
                return
            if not day_sched.empty:
                row = day_sched.iloc[0]
                for ev in self._build_events(row["market_open"], row["market_close"]):
                    ms = self._to_ms(ev)
                    if ms >= now_ms:
                        yield ms
                now_ms = -1
            current_date += timedelta(days=1)


def daily_schedule(
    after_open_minutes: Optional[int] = None,
    before_close_minutes: Optional[int] = None,
    calendar_name: str = "NYSE",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval_seconds: Optional[int] = None,
    inclusive: bool = False,
) -> Iterator[int]:
    """
    Backward‐compatible wrapper around DailySchedule.
    """
    return iter(
        DailySchedule(
            after_open_minutes=after_open_minutes,
            before_close_minutes=before_close_minutes,
            calendar_name=calendar_name,
            start_date=start_date,
            end_date=end_date,
            interval_seconds=interval_seconds,
            inclusive=inclusive,
        )
    )
