import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from typing import Iterator, Optional


def daily_schedule(
    after_open_minutes: Optional[int] = None,
    before_close_minutes: Optional[int] = None,
    calendar_name: str = 'NYSE',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval_seconds: Optional[int] = None,
    inclusive: bool = False
) -> Iterator[int]:
    """
    Generate UNIX‐ms timestamps for daily market open/close events with offsets.

    If after_open_minutes is None and before_close_minutes is not None: on‐close schedule.
    If after_open_minutes is not None and before_close_minutes is None: on‐open schedule.
    If both are not None: on‐open then on‐close each day.
    If both are None: raises ValueError.

    Args:
        after_open_minutes: Minutes after market open to schedule, or None.
        before_close_minutes: Minutes before market close to schedule, or None.
        calendar_name: Market calendar name (e.g. 'NYSE').
        start_date: ISO date string for start (e.g. '2023-01-01'); defaults to today.
        end_date: ISO date string for end; defaults to start_date.
        interval_seconds: Interval in seconds between points, or None for no interval.
        inclusive: Whether to include the end point if it's not already included.

    Yields:
        UNIX timestamp in milliseconds for each scheduled event.
    """
    if after_open_minutes is None and before_close_minutes is None:
        raise ValueError(
            "Must specify at least one of after_open_minutes or before_close_minutes"
        )

    # Determine if we should start from the current timestamp (not just the date)
    use_now = start_date is None
    calendar = mcal.get_calendar(calendar_name)
    # Default to today if no dates provided (date-only part)
    if start_date is None:
        start_date = datetime.now().date().isoformat()
    if end_date is None:
        end_date = start_date

    schedule_df = calendar.schedule(start_date=start_date, end_date=end_date)
    # If we want to start from the precise current time, capture now in schedule tz
    if use_now:
        idx = schedule_df.index
        tz = idx.tz if hasattr(idx, 'tz') else None
        now_dt = datetime.now(tz) if tz else datetime.now()
        now_ms = int(now_dt.timestamp() * 1000)
    else:
        now_ms = None

    # Iterate each trading day and generate schedule
    for _, row in schedule_df.iterrows():
        market_open = row['market_open']
        market_close = row['market_close']
        # on-close only
        if after_open_minutes is None:
            # cannot specify interval for close-only schedule
            if interval_seconds is not None:
                raise ValueError(
                    "Cannot specify interval_seconds on a close-only schedule"
                )
            ts = market_close - timedelta(minutes=before_close_minutes)
            ms = int(ts.timestamp() * 1000)
            if now_ms is None or ms >= now_ms:
                yield ms
            continue
        # on-open only
        if before_close_minutes is None:
            ts_start = market_open + timedelta(minutes=after_open_minutes)
            if interval_seconds is None:
                ms = int(ts_start.timestamp() * 1000)
                if now_ms is None or ms >= now_ms:
                    yield ms
            else:
                # generate from ts_start to market_close by interval_seconds
                delta = timedelta(seconds=interval_seconds)
                t = ts_start
                while t <= market_close:
                    ms = int(t.timestamp() * 1000)
                    if now_ms is None or ms >= now_ms:
                        yield ms
                    t += delta
            continue
        # both open and close with optional interval
        # compute start and end datetimes
        start_dt = market_open + timedelta(minutes=after_open_minutes)
        end_dt = market_close - timedelta(minutes=before_close_minutes)
        # no interval: just yield start and end when interval_seconds is None
        if interval_seconds is None:
            for dt in (start_dt, end_dt):
                ms = int(dt.timestamp() * 1000)
                if now_ms is None or ms >= now_ms:
                    yield ms
        else:
            # generate points from start to end every interval_seconds
            delta = timedelta(seconds=interval_seconds)
            t = start_dt
            last_dt = None
            while t <= end_dt:
                ms = int(t.timestamp() * 1000)
                if now_ms is None or ms >= now_ms:
                    yield ms
                    last_dt = t
                t += delta
            # if inclusive and end_dt was not hit exactly, include it
            if inclusive and last_dt is not None and last_dt < end_dt:
                ms = int(end_dt.timestamp() * 1000)
                if now_ms is None or ms >= now_ms:
                    yield ms 