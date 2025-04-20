import unittest
from unittest.mock import patch

from portwine.execution import ExecutionBase


class FakeExec(ExecutionBase):
    """Override step for testing run()."""
    def __init__(self):
        # Do not call super().__init__; run() only uses step()
        self.calls = []

    def step(self, timestamp_ms=None):
        # Record invocation
        self.calls.append(timestamp_ms)
        # Return empty list to satisfy signature
        return []


def schedule_generator(base_time, intervals):
    """
    Yield timestamps at base_time + each interval.
    intervals are in ms.
    """
    for offset in intervals:
        yield base_time + offset


class TestExecutionRun(unittest.TestCase):
    def test_run_calls_step_for_each_timestamp(self):
        base_time_s = 1.0
        # The run() computes now_ms = int(time.time() * 1000)
        # For base_time_s=1.0, now_ms=1000
        # schedule yields [1100, 1200, 1300]
        intervals = [100, 200, 300]
        schedule = schedule_generator(1000, intervals)

        fake_exec = FakeExec()
        # Patch time.time and time.sleep so no real waiting
        with patch('portwine.execution.base.time.time', return_value=base_time_s), \
             patch('portwine.execution.base.time.sleep') as mock_sleep:
            fake_exec.run(schedule)

        # Ensure step() was called with each scheduled timestamp
        expected = [1000 + i for i in intervals]
        self.assertEqual(fake_exec.calls, expected)
        # Sleep should have been called for each timestamp (3 calls)
        self.assertEqual(mock_sleep.call_count, len(intervals))
        # Check that sleep was called with the correct durations
        # Durations in seconds: (timestamp - now_ms) / 1000 => [0.1, 0.2, 0.3]
        calls = [call.args[0] for call in mock_sleep.call_args_list]
        self.assertEqual(calls, [i / 1000.0 for i in intervals])


if __name__ == '__main__':
    unittest.main() 