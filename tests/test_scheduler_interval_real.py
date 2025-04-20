import unittest
from datetime import timedelta
import pandas_market_calendars as mcal

from portwine.execution.scheduler import daily_schedule


class TestIntervalScheduleReal(unittest.TestCase):
    def setUp(self):
        self.calendar_name = 'NYSE'
        # Use two consecutive trading days around known date
        self.start = '2023-03-20'
        self.end = '2023-03-21'

    def test_error_interval_on_close_real(self):
        # Cannot specify interval_seconds on close-only schedule
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


if __name__ == '__main__':
    unittest.main() 