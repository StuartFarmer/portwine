import unittest
from unittest.mock import patch
import pandas as pd

from portwine.scheduler import daily_schedule


class TestDailyScheduleNow(unittest.TestCase):
    def setUp(self):
        # Fake calendar for a single trading day with open/close at fixed times
        class FakeCal:
            def schedule(self, start_date, end_date):
                idx = [pd.Timestamp('2025-04-21 10:00:00', tz='UTC')]
                opens = idx
                closes = [pd.Timestamp('2025-04-21 10:06:00', tz='UTC')]
                return pd.DataFrame({'market_open': opens, 'market_close': closes}, index=idx)
        self.fake_cal = FakeCal()

    @patch('portwine.scheduler.mcal.get_calendar')
    def test_open_only_with_interval_starts_from_now(self, mock_get_cal):
        mock_get_cal.return_value = self.fake_cal
        gen = daily_schedule(
            after_open_minutes=0,
            before_close_minutes=None,
            calendar_name='TEST',
            start_date=None,
            interval_seconds=60,
        )
        result = list(gen)
        base = pd.Timestamp('2025-04-21 10:00:00', tz='UTC')
        schedule = [base + pd.Timedelta(seconds=60 * i) for i in range(7)]
        now_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
        expected = [int(ts.timestamp() * 1000) for ts in schedule if int(ts.timestamp() * 1000) >= now_ms]
        self.assertEqual(result, expected)

    @patch('portwine.scheduler.mcal.get_calendar')
    def test_close_only_future(self, mock_get_cal):
        mock_get_cal.return_value = self.fake_cal
        gen = daily_schedule(
            after_open_minutes=None,
            before_close_minutes=0,
            calendar_name='TEST',
            start_date=None,
        )
        result = list(gen)
        close_ms = int(pd.Timestamp('2025-04-21 10:06:00', tz='UTC').timestamp() * 1000)
        now_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
        expected = [close_ms] if close_ms >= now_ms else []
        self.assertEqual(result, expected)

    @patch('portwine.scheduler.mcal.get_calendar')
    def test_explicit_start_date_ignores_now(self, mock_get_cal):
        mock_get_cal.return_value = self.fake_cal
        gen = daily_schedule(
            after_open_minutes=0,
            before_close_minutes=None,
            calendar_name='TEST',
            start_date='2025-04-21',
            interval_seconds=60,
        )
        result = list(gen)
        base = pd.Timestamp('2025-04-21 10:00:00', tz='UTC')
        expected = [int((base + pd.Timedelta(seconds=60 * i)).timestamp() * 1000) for i in range(7)]
        self.assertEqual(result, expected) 