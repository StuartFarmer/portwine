import unittest
from datetime import timedelta
import pandas_market_calendars as mcal
from unittest.mock import patch
import pandas as pd
from datetime import datetime

from portwine.scheduler import daily_schedule


class TestIntervalScheduleReal(unittest.TestCase):
    def setUp(self):
        self.calendar_name = 'NYSE'
        # Use two consecutive trading days around known date
        self.start = '2023-03-20'
        self.end = '2023-03-21'

    def test_error_interval_on_close_real(self):
        with self.assertRaises(ValueError):
            list(daily_schedule(
                after_open_minutes=None,
                before_close_minutes=15,
                calendar_name=self.calendar_name,
                start_date=self.start,
                end_date=self.end,
                interval_seconds=600
            ))

    def test_interval_open_only_real(self):
        # 10 minutes after open, every 10 minutes, across two days
        after = 10
        interval = 10 * 60  # seconds
        gen = daily_schedule(
            after_open_minutes=after,
            before_close_minutes=None,
            calendar_name=self.calendar_name,
            start_date=self.start,
            end_date=self.end,
            interval_seconds=interval
        )
        result = list(gen)
        # Build expected using real calendar
        cal = mcal.get_calendar(self.calendar_name)
        sched = cal.schedule(start_date=self.start, end_date=self.end)
        expected = []
        for _, row in sched.iterrows():
            start_dt = row['market_open'] + timedelta(minutes=after)
            end_dt = row['market_close']
            t = start_dt
            while t <= end_dt:
                expected.append(int(t.timestamp() * 1000))
                t += timedelta(seconds=interval)
        self.assertEqual(result, expected)

    def test_interval_with_before_close_real(self):
        # 10 after open, 30 before close, every 10 minutes, across two days
        after = 10
        before = 30
        interval = 10 * 60
        gen = daily_schedule(
            after_open_minutes=after,
            before_close_minutes=before,
            calendar_name=self.calendar_name,
            start_date=self.start,
            end_date=self.end,
            interval_seconds=interval
        )
        result = list(gen)
        cal = mcal.get_calendar(self.calendar_name)
        sched = cal.schedule(start_date=self.start, end_date=self.end)
        expected = []
        for _, row in sched.iterrows():
            start_dt = row['market_open'] + timedelta(minutes=after)
            end_dt = row['market_close'] - timedelta(minutes=before)
            t = start_dt
            while t <= end_dt:
                expected.append(int(t.timestamp() * 1000))
                t += timedelta(seconds=interval)
        self.assertEqual(result, expected)

    def test_non_inclusive_before_close_real(self):
        # 10 after open, 45 before close, every 10 minutes, exclusive
        after = 10
        before = 45
        interval = 10 * 60
        gen = daily_schedule(
            after_open_minutes=after,
            before_close_minutes=before,
            calendar_name=self.calendar_name,
            start_date=self.start,
            end_date=self.start,
            interval_seconds=interval,
            inclusive=False
        )
        result = list(gen)
        # Single-day schedule
        cal = mcal.get_calendar(self.calendar_name)
        sched = cal.schedule(start_date=self.start, end_date=self.start)
        row = sched.iloc[0]
        start_dt = row['market_open'] + timedelta(minutes=after)
        end_dt = row['market_close'] - timedelta(minutes=before)
        expected = []
        t = start_dt
        while t <= end_dt:
            expected.append(int(t.timestamp() * 1000))
            t += timedelta(seconds=interval)
        self.assertEqual(result, expected)

    def test_inclusive_before_close_real(self):
        # 10 after open, 45 before close, every 10 minutes, inclusive
        after = 10
        before = 45
        interval = 10 * 60
        gen = daily_schedule(
            after_open_minutes=after,
            before_close_minutes=before,
            calendar_name=self.calendar_name,
            start_date=self.start,
            end_date=self.start,
            interval_seconds=interval,
            inclusive=True
        )
        result = list(gen)
        cal = mcal.get_calendar(self.calendar_name)
        sched = cal.schedule(start_date=self.start, end_date=self.start)
        row = sched.iloc[0]
        start_dt = row['market_open'] + timedelta(minutes=after)
        end_dt = row['market_close'] - timedelta(minutes=before)
        expected = []
        t = start_dt
        last = None
        while t <= end_dt:
            expected.append(int(t.timestamp() * 1000))
            last = t
            t += timedelta(seconds=interval)
        if last < end_dt:
            expected.append(int(end_dt.timestamp() * 1000))
        self.assertEqual(result, expected)


class TestDailyScheduleReal(unittest.TestCase):
    def setUp(self):
        self.calendar_name = 'NYSE'
        # pick a known recent trading day
        self.test_date = '2023-03-20'

    def test_on_open_only_real(self):
        # 5 minutes after market open
        after = 5
        gen = daily_schedule(
            after_open_minutes=after,
            before_close_minutes=None,
            calendar_name=self.calendar_name,
            start_date=self.test_date,
            end_date=self.test_date
        )
        result = list(gen)
        # Fetch actual calendar open time
        cal = mcal.get_calendar(self.calendar_name)
        sched = cal.schedule(start_date=self.test_date, end_date=self.test_date)
        open_ts = sched['market_open'].iloc[0] + timedelta(minutes=after)
        expected = [int(open_ts.timestamp() * 1000)]
        self.assertEqual(result, expected)

    def test_on_close_only_real(self):
        # 10 minutes before market close
        before = 10
        gen = daily_schedule(
            after_open_minutes=None,
            before_close_minutes=before,
            calendar_name=self.calendar_name,
            start_date=self.test_date,
            end_date=self.test_date
        )
        result = list(gen)
        cal = mcal.get_calendar(self.calendar_name)
        sched = cal.schedule(start_date=self.test_date, end_date=self.test_date)
        close_ts = sched['market_close'].iloc[0] - timedelta(minutes=before)
        expected = [int(close_ts.timestamp() * 1000)]
        self.assertEqual(result, expected)

    def test_open_and_close_real(self):
        # 15 min after open, 20 min before close
        after = 15
        before = 20
        gen = daily_schedule(
            after_open_minutes=after,
            before_close_minutes=before,
            calendar_name=self.calendar_name,
            start_date=self.test_date,
            end_date=self.test_date
        )
        result = list(gen)
        cal = mcal.get_calendar(self.calendar_name)
        sched = cal.schedule(start_date=self.test_date, end_date=self.test_date)
        open_ts = sched['market_open'].iloc[0] + timedelta(minutes=after)
        close_ts = sched['market_close'].iloc[0] - timedelta(minutes=before)
        expected = [int(open_ts.timestamp() * 1000), int(close_ts.timestamp() * 1000)]
        self.assertEqual(result, expected)

    def test_neither_offset_raises(self):
        with self.assertRaises(ValueError):
            list(daily_schedule(
                after_open_minutes=None,
                before_close_minutes=None,
                calendar_name=self.calendar_name,
                start_date=self.test_date,
                end_date=self.test_date
            ))

    def test_start_and_end_date_range(self):
        # Range of two days should yield 2 events each for open-only
        after = 3
        start = '2023-03-20'
        end = '2023-03-21'
        gen = daily_schedule(
            after_open_minutes=after,
            before_close_minutes=None,
            calendar_name=self.calendar_name,
            start_date=start,
            end_date=end
        )
        result = list(gen)
        cal = mcal.get_calendar(self.calendar_name)
        sched = cal.schedule(start_date=start, end_date=end)
        expected = [int((ts + timedelta(minutes=after)).timestamp() * 1000)
                    for ts in sched['market_open']]
        self.assertEqual(result, expected)

    def test_end_date_stopiteration_real(self):
        # Single-day on-close, check StopIteration after one
        before = 1
        gen = daily_schedule(
            after_open_minutes=None,
            before_close_minutes=before,
            calendar_name=self.calendar_name,
            start_date=self.test_date,
            end_date=self.test_date
        )
        it = iter(gen)
        first = next(it)
        with self.assertRaises(StopIteration):
            next(it)


class DummyCalendar:
    """A fake exchange calendar for testing two or more consecutive days."""
    def schedule(self, start_date, end_date):
        # Parse ISO dates
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        days = (end - start).days + 1
        dates = [start + timedelta(days=i) for i in range(days)]
        idx = pd.to_datetime(dates)
        # Market open at 09:30, close at 16:00 local
        opens = [d.replace(hour=9, minute=30) for d in dates]
        closes = [d.replace(hour=16, minute=0) for d in dates]
        df = pd.DataFrame({"market_open": opens, "market_close": closes}, index=idx)
        return df


@patch('portwine.scheduler.mcal.get_calendar', return_value=DummyCalendar())
class TestDailySchedule(unittest.TestCase):
    def test_no_interval_multiple_days(self, mock_gc):
        """When no interval, schedule yields exactly one timestamp per day."""
        # 3-day schedule
        schedule = list(
            daily_schedule(
                after_open_minutes=0,
                before_close_minutes=None,
                calendar_name='TEST',
                start_date='2021-01-01',
                end_date='2021-01-03',
            )
        )
        # Expect 3 timestamps (one per day)
        self.assertEqual(len(schedule), 3)
        # Each successive timestamp is 24h apart
        diffs = [schedule[i+1] - schedule[i] for i in range(2)]
        ms_per_day = 24 * 60 * 60 * 1000
        self.assertTrue(all(diff == ms_per_day for diff in diffs))

    def test_interval_multiple_days(self, mock_gc):
        """When interval_SECONDS, schedule yields multiple per day, and rolls over."""
        # Hourly interval, 2-day schedule
        schedule = list(
            daily_schedule(
                after_open_minutes=0,
                before_close_minutes=None,
                interval_seconds=3600,
                calendar_name='TEST',
                start_date='2021-01-01',
                end_date='2021-01-02',
            )
        )
        # On each day: open at 09:30, then every hour until <=16:00
        # That yields at times: 09:30,10:30,11:30,12:30,13:30,14:30,15:30 => 7 per day
        self.assertEqual(len(schedule), 7 * 2)
        # Check first-day spacing is exactly 1h
        first_day = schedule[:7]
        hourly_ms = 3600 * 1000
        diffs = [first_day[i+1] - first_day[i] for i in range(6)]
        self.assertTrue(all(diff == hourly_ms for diff in diffs))


if __name__ == '__main__':
    unittest.main() 