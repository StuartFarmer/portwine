import unittest
from datetime import datetime, timedelta, date, timezone
from itertools import islice
from types import SimpleNamespace
from unittest.mock import patch

import portwine.scheduler as scheduler

from unittest import mock

from portwine.scheduler import DailySchedule, daily_schedule

import pandas_market_calendars as mcal
import pandas as pd

import pytest



# helper: convert datetime/Timestamp -> UNIX-ms
def ms(ts):
    return int(pd.Timestamp(ts).value // 1_000_000)

class FakeCalendar:
    """
    Fake pandas_market_calendars Calendar:
    - For each new DailySchedule, returns a fresh FakeCalendar.
    - schedule(start_date, end_date) filters self.df between those dates.
      Raises StopIteration after two calls to end live iteration.
    """
    def __init__(self, df):
        self.df = df
        self.calls = 0

    def schedule(self, start_date, end_date):
        self.calls += 1
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)
        mask = (self.df.index >= sd) & (self.df.index <= ed)
        subset = self.df.loc[mask]
        if self.calls <= 2:
            return subset
        raise StopIteration

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

class FiniteBusinessCalendar:
    """Business-day calendar 09:30–16:00 UTC."""
    def schedule(self, start_date, end_date):
        start = pd.to_datetime(start_date)
        end   = pd.to_datetime(end_date)
        days  = pd.date_range(start, end, freq="D")
        biz   = [d for d in days if d.weekday() < 5]
        opens = [d.replace(hour= 9, minute=30) for d in biz]
        closes= [d.replace(hour=16, minute= 0) for d in biz]
        return pd.DataFrame({
            "market_open":  opens,
            "market_close": closes
        }, index=biz)


class SingleDayCalendar:
    """One session per requested day at fixed open/close times."""
    def __init__(self, open_time: str, close_time: str):
        self.open_time  = open_time
        self.close_time = close_time

    def schedule(self, start_date, end_date):
        opens  = pd.to_datetime(f"{start_date} {self.open_time}",  utc=True)
        closes = pd.to_datetime(f"{start_date} {self.close_time}", utc=True)
        idx    = [pd.to_datetime(start_date)]
        return pd.DataFrame({
            "market_open":  [opens],
            "market_close": [closes]
        }, index=idx)


class TestFiniteDailySchedule(unittest.TestCase):

    @patch("portwine.scheduler.mcal.get_calendar",
           return_value=FiniteBusinessCalendar())
    def test_open_only_single_day(self, mock_gc):
        res = list(daily_schedule(
            after_open_minutes=5,
            before_close_minutes=None,
            calendar_name="TEST",
            start_date="2021-01-04",
            end_date="2021-01-04"
        ))
        self.assertEqual(res, [ms(datetime(2021, 1, 4, 9, 35))])


class TestLiveDailySchedule(unittest.TestCase):
    def setUp(self):
        # sessions at 10:00–10:06 UTC each day
        self.cal = SingleDayCalendar(open_time="10:00:00", close_time="10:06:00")

    def test_live_starts_from_now(self):
        class FakeDate(date):
            @classmethod
            def today(cls):
                return date(2025, 4, 21)

        now_ts = pd.Timestamp("2025-04-21 10:02:00", tz="UTC")
        fake_time = SimpleNamespace(time=lambda: now_ts.value / 1e9)

        with patch("portwine.scheduler.mcal.get_calendar", return_value=self.cal), \
             patch("time.time",                     fake_time.time), \
             patch.dict(scheduler.__dict__,        {"date": FakeDate}):

            gen = daily_schedule(
                after_open_minutes=0,
                before_close_minutes=None,
                interval_seconds=60,
                calendar_name="TEST"
            )

            base   = pd.Timestamp("2025-04-21 10:00:00", tz="UTC")
            all_ts = [base + timedelta(seconds=60*i) for i in range(7)]
            now_ms = ms(now_ts)
            expected = [ms(t) for t in all_ts if ms(t) >= now_ms]
            result   = list(islice(gen, len(expected)))

        self.assertEqual(result, expected)

    def test_live_rolls_after_close(self):
        class FakeDate(date):
            @classmethod
            def today(cls):
                return date(2025, 4, 21)

        now_ts = pd.Timestamp("2025-04-21 10:10:00", tz="UTC")
        fake_time = SimpleNamespace(time=lambda: now_ts.value / 1e9)

        with patch("portwine.scheduler.mcal.get_calendar", return_value=self.cal), \
             patch("time.time",                     fake_time.time), \
             patch.dict(scheduler.__dict__,        {"date": FakeDate}):

            gen = daily_schedule(
                after_open_minutes=0,
                before_close_minutes=None,
                calendar_name="TEST"
            )
            first = next(gen)

        tomorrow = pd.Timestamp("2025-04-22 10:00:00", tz="UTC")
        self.assertEqual(first, ms(tomorrow))

    def test_live_skips_weekends(self):
        class FakeDate(date):
            @classmethod
            def today(cls):
                return date(2021, 1, 2)  # Saturday

        fake_time = SimpleNamespace(time=lambda: 0)

        with patch("portwine.scheduler.mcal.get_calendar",
                   return_value=FiniteBusinessCalendar()), \
             patch("time.time",            fake_time.time), \
             patch.dict(scheduler.__dict__, {"date": FakeDate}):

            gen   = daily_schedule(
                after_open_minutes=0,
                before_close_minutes=None,
                calendar_name="TEST"
            )
            first = next(gen)

        # Monday Jan 4, 2021 at 09:30
        self.assertEqual(first, ms(datetime(2021,1,4,9,30)))

class TestDailySchedule(unittest.TestCase):
    def setUp(self):
        # Two-day schedule with naive UTC times
        dates = pd.to_datetime(["2025-04-01", "2025-04-02"])
        opens = dates + pd.Timedelta(hours=13)
        closes = dates + pd.Timedelta(hours=20)
        self.schedule_df = pd.DataFrame({
            "market_open": opens,
            "market_close": closes
        }, index=dates)

        # Patch get_calendar to return a new FakeCalendar each time
        patcher = mock.patch(
            'portwine.scheduler.mcal.get_calendar',
            new=lambda name: FakeCalendar(self.schedule_df)
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_error_when_no_offsets(self):
        with self.assertRaises(ValueError):
            DailySchedule()

    def test_error_close_only_with_interval(self):
        with self.assertRaises(ValueError):
            DailySchedule(before_close_minutes=5, interval_seconds=60)

    def test_to_ms_naive_and_tzaware(self):
        ds = DailySchedule(after_open_minutes=1, before_close_minutes=1, start_date="2025-04-01")
        dt_naive = datetime(2025, 4, 1, 13, 1)
        ms1 = ds._to_ms(dt_naive)
        dt_aware = datetime(2025, 4, 1, 13, 1, tzinfo=timezone.utc)
        ms2 = ds._to_ms(dt_aware)
        self.assertEqual(ms1, ms2)

    def test_build_events_various_modes(self):
        row = self.schedule_df.iloc[0]
        open_dt, close_dt = row["market_open"], row["market_close"]

        # close-only
        ds = DailySchedule(after_open_minutes=None, before_close_minutes=10)
        evs = ds._build_events(open_dt, close_dt)
        exp = [
            (close_dt - timedelta(minutes=10)).tz_localize("UTC")
        ]
        self.assertEqual(evs, exp)

        # open-only
        ds = DailySchedule(after_open_minutes=15, before_close_minutes=None)
        evs = ds._build_events(open_dt, close_dt)
        exp = [
            (open_dt + timedelta(minutes=15)).tz_localize("UTC")
        ]
        self.assertEqual(evs, exp)

        # open+close
        ds = DailySchedule(after_open_minutes=5, before_close_minutes=5)
        evs = ds._build_events(open_dt, close_dt)
        exp = [
            (open_dt + timedelta(minutes=5)).tz_localize("UTC"),
            (close_dt - timedelta(minutes=5)).tz_localize("UTC"),
        ]
        self.assertEqual(evs, exp)

        # open+interval (every hour)
        ds = DailySchedule(after_open_minutes=0, before_close_minutes=None, interval_seconds=3600)
        evs = ds._build_events(open_dt, close_dt)
        hours = int((close_dt - open_dt).total_seconds() // 3600) + 1
        self.assertEqual(len(evs), hours)
        # ensure they are all tz-aware
        for t in evs:
            self.assertIsNotNone(t.tzinfo)

        # open+close+interval without inclusive
        ds = DailySchedule(after_open_minutes=0, before_close_minutes=0,
                           interval_seconds=3600, inclusive=False)
        evs = ds._build_events(open_dt, close_dt)
        self.assertEqual(len(evs), hours)

        # open+close+interval with inclusive
        ds = DailySchedule(after_open_minutes=0, before_close_minutes=30,
                           interval_seconds=3600, inclusive=True)
        evs = ds._build_events(open_dt, close_dt)
        self.assertIn((close_dt - timedelta(minutes=30)).tz_localize("UTC"), evs)

    def test_finite_generator_open_and_close(self):
        ds = DailySchedule(after_open_minutes=10, before_close_minutes=20,
                           start_date="2025-04-01", end_date="2025-04-02")
        ms_list = list(ds)
        self.assertEqual(len(ms_list), 4)
        first = int((self.schedule_df.market_open.iloc[0] + timedelta(minutes=10))
                    .tz_localize("UTC").timestamp() * 1000)
        last = int((self.schedule_df.market_close.iloc[1] - timedelta(minutes=20))
                   .tz_localize("UTC").timestamp() * 1000)
        self.assertEqual(ms_list[0], first)
        self.assertEqual(ms_list[-1], last)

    def test_finite_generator_open_only(self):
        ds = DailySchedule(after_open_minutes=30, start_date="2025-04-02")
        ms_list = list(ds)
        self.assertEqual(len(ms_list), 1)
        exp = int((self.schedule_df.market_open.iloc[1] + timedelta(minutes=30))
                  .tz_localize("UTC").timestamp() * 1000)
        self.assertEqual(ms_list[0], exp)

    def test_daily_schedule_helper(self):
        # Compare helper vs explicit for same parameters
        ms1 = list(daily_schedule(after_open_minutes=1, start_date="2025-04-01", end_date="2025-04-01"))
        ms2 = list(DailySchedule(after_open_minutes=1, start_date="2025-04-01", end_date="2025-04-01"))
        self.assertEqual(ms1, ms2)

    def test_live_generator_yields_nothing_when_all_in_past(self):
        ds = DailySchedule(after_open_minutes=0)
        evs = list(ds)
        # Schedule dates (04-01,04-02) are before now → no events
        self.assertEqual(evs, [])


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
        result = list(islice(gen, 1))
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


import unittest
from unittest.mock import patch
import pandas as pd
from itertools import islice

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
        # Build expected series for first day
        base = pd.Timestamp('2025-04-21 10:00:00', tz='UTC')
        schedule = [base + pd.Timedelta(seconds=60 * i) for i in range(7)]
        now_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
        expected = [int(ts.timestamp() * 1000) for ts in schedule if int(ts.timestamp() * 1000) >= now_ms]
        # Only consume as many events as expected to avoid infinite loop
        result = list(islice(gen, len(expected)))
        self.assertEqual(result, expected)

    @patch('portwine.scheduler.mcal.get_calendar')
    def test_close_only_future(self, mock_get_cal):
        mock_get_cal.return_value = self.fake_cal
        # Use finite mode by specifying start_date and end_date so we always get the close timestamp
        gen = daily_schedule(
            after_open_minutes=None,
            before_close_minutes=0,
            calendar_name='TEST',
            start_date='2025-04-21',
            end_date='2025-04-21',
        )
        result = list(gen)
        close_ms = int(pd.Timestamp('2025-04-21 10:06:00', tz='UTC').timestamp() * 1000)
        expected = [close_ms]
        self.assertEqual(result, expected)

    @patch('portwine.scheduler.mcal.get_calendar')
    def test_explicit_start_date_ignores_now(self, mock_get_cal):
        mock_get_cal.return_value = self.fake_cal
        gen = daily_schedule(
            after_open_minutes=0,
            before_close_minutes=None,
            calendar_name='TEST',
            start_date='2025-04-21',
            end_date='2025-04-21',
            interval_seconds=60,
        )
        result = list(gen)
        base = pd.Timestamp('2025-04-21 10:00:00', tz='UTC')
        expected = [int((base + pd.Timedelta(seconds=60 * i)).timestamp() * 1000) for i in range(7)]
        self.assertEqual(result, expected)

class TestScheduleHandoff(unittest.TestCase):
    def setUp(self):
        # Calendar with three consecutive days
        self.dates = pd.to_datetime(["2025-04-25", "2025-04-26", "2025-04-28"])  # skip weekend
        opens = self.dates + pd.Timedelta(hours=9, minutes=30)
        closes = self.dates + pd.Timedelta(hours=16)
        self.schedule_df = pd.DataFrame({
            "market_open": opens,
            "market_close": closes
        }, index=self.dates)
        patcher = mock.patch(
            'portwine.scheduler.mcal.get_calendar',
            new=lambda name: FakeCalendar(self.schedule_df)
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    @pytest.mark.xfail(reason="Sanity check: this test is expected to fail on purpose.")
    def test_schedule_handoff_yields_old_timestamps(self):
        # Simulate warmup up to 2025-04-28 09:48 UTC
        warmup_cutoff = pd.Timestamp("2025-04-28 09:48:00", tz="UTC")
        gen = daily_schedule(
            after_open_minutes=0,
            before_close_minutes=None,
            calendar_name="TEST",
            start_date="2025-04-25",
            interval_seconds=120
        )
        # Advance generator up to warmup_cutoff
        last_dt = None
        for ts in gen:
            dt = pd.to_datetime(ts, unit='ms', utc=True)
            if dt >= warmup_cutoff:
                break
            last_dt = dt
        # Now, continue iterating and collect the next 5 timestamps
        next_ts = [pd.to_datetime(next(gen), unit='ms', utc=True) for _ in range(5)]
        # Assert that some of the next timestamps are from before warmup_cutoff (i.e., from previous days)
        self.assertTrue(any(dt < warmup_cutoff for dt in next_ts),
                        f"Expected at least one timestamp before warmup_cutoff, got: {next_ts}")

if __name__ == "__main__":
    unittest.main()
