import unittest
from unittest.mock import patch
from datetime import datetime
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
    @patch('portwine.scheduler.datetime.now')
    def test_open_only_with_interval_starts_from_now(self, mock_now, mock_get_cal):
        # Simulate current time at 10:02:30 UTC
        now = pd.Timestamp('2025-04-21 10:02:30', tz='UTC').to_pydatetime()
        mock_now.return_value = now
        mock_get_cal.return_value = self.fake_cal

        # after_open=0 min, interval=60s, start_date=None → only from now on
        gen = daily_schedule(
            after_open_minutes=0,
            before_close_minutes=None,
            calendar_name='TEST',
            start_date=None,
            interval_seconds=60,
        )
        result = list(gen)
        expected = [
            int(pd.Timestamp('2025-04-21 10:03:00', tz='UTC').timestamp() * 1000),
            int(pd.Timestamp('2025-04-21 10:04:00', tz='UTC').timestamp() * 1000),
            int(pd.Timestamp('2025-04-21 10:05:00', tz='UTC').timestamp() * 1000),
            int(pd.Timestamp('2025-04-21 10:06:00', tz='UTC').timestamp() * 1000),
        ]
        self.assertEqual(result, expected)

    @patch('portwine.scheduler.mcal.get_calendar')
    @patch('portwine.scheduler.datetime.now')
    def test_close_only_does_not_emit_past_events(self, mock_now, mock_get_cal):
        # Simulate current time after close
        now = pd.Timestamp('2025-04-21 11:00:00', tz='UTC').to_pydatetime()
        mock_now.return_value = now
        mock_get_cal.return_value = self.fake_cal

        # before_close=0 min → close at 10:06 → since now > close, yields nothing
        gen = daily_schedule(
            after_open_minutes=None,
            before_close_minutes=0,
            calendar_name='TEST',
        )
        result = list(gen)
        self.assertEqual(result, [])

    @patch('portwine.scheduler.mcal.get_calendar')
    @patch('portwine.scheduler.datetime.now')
    def test_close_only_includes_exact_now(self, mock_now, mock_get_cal):
        # Simulate current time exactly at close
        now = pd.Timestamp('2025-04-21 10:06:00', tz='UTC').to_pydatetime()
        mock_now.return_value = now
        mock_get_cal.return_value = self.fake_cal

        # before_close=0 → event at 10:06 → should include since >= now
        gen = daily_schedule(
            after_open_minutes=None,
            before_close_minutes=0,
            calendar_name='TEST',
        )
        result = list(gen)
        expected = [int(pd.Timestamp('2025-04-21 10:06:00', tz='UTC').timestamp() * 1000)]
        self.assertEqual(result, expected)

    @patch('portwine.scheduler.mcal.get_calendar')
    @patch('portwine.scheduler.datetime.now')
    def test_explicit_start_date_ignores_now(self, mock_now, mock_get_cal):
        # Simulate current time well after close
        now = pd.Timestamp('2025-04-21 12:00:00', tz='UTC').to_pydatetime()
        mock_now.return_value = now
        mock_get_cal.return_value = self.fake_cal

        # Explicit start_date should yield full open->close sequence despite now
        gen = daily_schedule(
            after_open_minutes=0,
            before_close_minutes=None,
            calendar_name='TEST',
            start_date='2025-04-21',
            interval_seconds=60,
        )
        result = list(gen)
        expected = [
            int(pd.Timestamp('2025-04-21 10:00:00', tz='UTC').timestamp() * 1000),
            int(pd.Timestamp('2025-04-21 10:01:00', tz='UTC').timestamp() * 1000),
            int(pd.Timestamp('2025-04-21 10:02:00', tz='UTC').timestamp() * 1000),
            int(pd.Timestamp('2025-04-21 10:03:00', tz='UTC').timestamp() * 1000),
            int(pd.Timestamp('2025-04-21 10:04:00', tz='UTC').timestamp() * 1000),
            int(pd.Timestamp('2025-04-21 10:05:00', tz='UTC').timestamp() * 1000),
            int(pd.Timestamp('2025-04-21 10:06:00', tz='UTC').timestamp() * 1000),
        ]
        self.assertEqual(result, expected) 