"""
Integration tests for the DailyExecutor class.

This test suite demonstrates how to use the DailyExecutor with actual components
and configuration files, focusing on the interaction between the components
rather than isolated unit testing.
"""

import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from portwine.utils.daily_executor import DailyExecutor


class SimpleTestStrategy:
    """
    Simple strategy implementation for integration testing.
    """
    
    def __init__(self, tickers, **kwargs):
        self.tickers = tickers
        self.params = kwargs
        
    def generate_signals(self):
        """Return equal weight allocation for all tickers."""
        weight = 1.0 / len(self.tickers)
        return {ticker: weight for ticker in self.tickers}
    
    def shutdown(self):
        """Clean up resources."""
        pass


class SimpleTestExecutor:
    """
    Simple execution_complex system for integration testing.
    """
    
    def __init__(self, strategy=None, market_data_loader=None, **kwargs):
        self.strategy = strategy
        self.market_data_loader = market_data_loader
        self.params = kwargs
        self.last_signals = None
        
    def execute(self):
        """Execute trades based on strategy signals."""
        if self.strategy:
            self.last_signals = self.strategy.generate_signals()
        return True
    
    def step(self):
        """Alternative execution_complex method."""
        return self.execute()
    
    def shutdown(self):
        """Clean up resources."""
        pass


class SimpleTestDataLoader:
    """
    Simple data loader for integration testing.
    """
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.data = {}
        
    def update_data(self):
        """Update market data."""
        # In a real implementation, this would fetch data
        self.data = {
            "AAPL": {"close": 150.0, "open": 149.0},
            "MSFT": {"close": 300.0, "open": 299.0},
            "GOOG": {"close": 2500.0, "open": 2490.0},
        }
        return True
    
    def shutdown(self):
        """Clean up resources."""
        pass


class TestDailyExecutorIntegration(unittest.TestCase):
    """Integration tests for the DailyExecutor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a basic config for testing
        self.config = {
            "strategy": {
                "class": "__main__.SimpleTestStrategy",
                "tickers": ["AAPL", "MSFT", "GOOG"],
                "params": {
                    "lookback_days": 5
                }
            },
            "execution_complex": {
                "class": "__main__.SimpleTestExecutor",
                "params": {
                    "paper": True
                }
            },
            "data_loader": {
                "class": "__main__.SimpleTestDataLoader",
                "params": {
                    "timeframe": "1Day"
                }
            },
            "schedule": {
                "run_time": "15:45",
                "time_zone": "US/Eastern",
                "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "market_hours_only": True
            }
        }
        
        # Create intraday config for testing
        self.intraday_config = self.config.copy()
        self.intraday_config["schedule"] = {
            "run_time": "market_open+5m",
            "time_zone": "US/Eastern",
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "exchange": "NYSE",
            "market_hours_only": True,
            "intraday": "interval:30m"
        }
        
        # Write configs to files
        self.config_path = os.path.join(self.temp_dir.name, "test_config.json")
        self.intraday_config_path = os.path.join(self.temp_dir.name, "test_intraday_config.json")
        
        with open(self.config_path, "w") as f:
            json.dump(self.config, f)
        
        with open(self.intraday_config_path, "w") as f:
            json.dump(self.intraday_config, f)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_standard_workflow(self):
        """Test the standard workflow of init, initialize, run."""
        # Create executor from config
        executor = DailyExecutor(self.config)
        
        # Patch the _import_class method to return our test classes
        with patch.object(DailyExecutor, '_import_class') as mock_import:
            # Configure the mock to return our test classes
            def side_effect(class_path):
                if class_path == "__main__.SimpleTestStrategy":
                    return SimpleTestStrategy
                elif class_path == "__main__.SimpleTestExecutor":
                    return SimpleTestExecutor
                elif class_path == "__main__.SimpleTestDataLoader":
                    return SimpleTestDataLoader
                return None
            mock_import.side_effect = side_effect
            
            # Initialize components
            executor.initialize()
            
            # Check components were initialized properly
            self.assertIsInstance(executor.strategy, SimpleTestStrategy)
            self.assertIsInstance(executor.executor, SimpleTestExecutor)
            self.assertIsInstance(executor.data_loader, SimpleTestDataLoader)
            
            # Mock market open status
            with patch.object(executor, '_is_market_open', return_value=True):
                # Run once
                executor.run_once()
            
            # Check that the strategy generated signals and execution_complex processed them
            self.assertIsNotNone(executor.executor.last_signals)
            self.assertEqual(len(executor.executor.last_signals), 3)  # 3 tickers
            
            # Check proper allocation
            for ticker in executor.strategy.tickers:
                self.assertEqual(executor.executor.last_signals[ticker], 1.0 / 3.0)
            
            # Clean shutdown
            executor.shutdown()
    
    def test_from_config_file(self):
        """Test loading from config file."""
        # Load from config file with import class patching
        with patch.object(DailyExecutor, '_import_class') as mock_import:
            # Configure the mock to return our test classes
            def side_effect(class_path):
                if class_path == "__main__.SimpleTestStrategy":
                    return SimpleTestStrategy
                elif class_path == "__main__.SimpleTestExecutor":
                    return SimpleTestExecutor
                elif class_path == "__main__.SimpleTestDataLoader":
                    return SimpleTestDataLoader
                return None
            mock_import.side_effect = side_effect
            
            executor = DailyExecutor.from_config_file(self.config_path)
            
            # Check settings were loaded correctly
            self.assertEqual(executor.run_time, "15:45")
            self.assertEqual(executor.time_zone, "US/Eastern")
            self.assertEqual(executor.days, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
            
            # Initialize and run
            executor.initialize()
            
            with patch.object(executor, '_is_market_open', return_value=True):
                executor.run_once()
            
            # Check results
            self.assertIsNotNone(executor.executor.last_signals)
    
    def test_intraday_schedule(self):
        """Test intraday scheduling."""
        # Load intraday config with import class patching
        with patch.object(DailyExecutor, '_import_class') as mock_import:
            # Configure the mock
            def side_effect(class_path):
                if class_path == "__main__.SimpleTestStrategy":
                    return SimpleTestStrategy
                elif class_path == "__main__.SimpleTestExecutor":
                    return SimpleTestExecutor
                elif class_path == "__main__.SimpleTestDataLoader":
                    return SimpleTestDataLoader
                return None
            mock_import.side_effect = side_effect
            
            executor = DailyExecutor.from_config_file(self.intraday_config_path)
            
            # Check intraday settings
            self.assertEqual(executor.run_time, "market_open+5m")
            self.assertEqual(executor.intraday_schedule, "interval:30m")
            
            # Initialize components
            executor.initialize()
            
            # Mock the schedule setup methods and _log_next_run_time
            with patch('portwine.utils.daily_executor.schedule') as mock_schedule, \
                 patch.object(executor, '_log_next_run_time') as mock_log_next_run:
                # Configure mock run_pending to exit the loop
                mock_schedule.run_pending.side_effect = KeyboardInterrupt
                
                # Create a mock for every() method chain
                mock_every = MagicMock()
                mock_schedule.every.return_value = mock_every
                
                # Create a mock for each day of the week
                mock_day = MagicMock()
                mock_every.monday = mock_day
                mock_every.tuesday = mock_day
                mock_every.wednesday = mock_day
                mock_every.thursday = mock_day
                mock_every.friday = mock_day
                
                # Mock at method to return a further chain
                mock_at = MagicMock()
                mock_day.at.return_value = mock_at
                
                # Ensure do method is available on the chain
                mock_do = MagicMock()
                mock_at.do = mock_do
                
                try:
                    executor.run_scheduled()
                except KeyboardInterrupt:
                    pass
                
                # Verify schedule setup was called
                mock_schedule.every.assert_called()
                mock_day.at.assert_called()
    
    def test_market_based_timing(self):
        """Test market-based timing."""
        # Modify config for market-based schedule
        market_config = self.config.copy()
        market_config["schedule"]["run_time"] = "market_close-15m"
        
        # Create executor with import class patching
        with patch.object(DailyExecutor, '_import_class') as mock_import:
            # Configure the mock
            def side_effect(class_path):
                if class_path == "__main__.SimpleTestStrategy":
                    return SimpleTestStrategy
                elif class_path == "__main__.SimpleTestExecutor":
                    return SimpleTestExecutor
                elif class_path == "__main__.SimpleTestDataLoader":
                    return SimpleTestDataLoader
                return None
            mock_import.side_effect = side_effect
            
            executor = DailyExecutor(market_config)
            executor.initialize()
            
            # Mock time expression parsing, schedule module, and _log_next_run_time
            with patch('portwine.utils.daily_executor.schedule') as mock_schedule, \
                 patch.object(executor, '_log_next_run_time') as mock_log_next_run:
                # Configure mock run_pending to exit the loop
                mock_schedule.run_pending.side_effect = KeyboardInterrupt
                
                # Create a mock for every() method chain
                mock_every = MagicMock()
                mock_schedule.every.return_value = mock_every
                
                # Create a mock for each day of the week
                mock_day = MagicMock()
                mock_every.monday = mock_day
                mock_every.tuesday = mock_day
                mock_every.wednesday = mock_day
                mock_every.thursday = mock_day
                mock_every.friday = mock_day
                
                # Mock at method to return a further chain
                mock_at = MagicMock()
                mock_day.at.return_value = mock_at
                
                # Ensure do method is available on the chain
                mock_do = MagicMock()
                mock_at.do = mock_do
                
                try:
                    executor.run_scheduled()
                except KeyboardInterrupt:
                    pass
                
                # Verify market-based scheduling was set up
                mock_schedule.every.assert_called()
                mock_day.at.assert_called_with("00:01")


if __name__ == '__main__':
    unittest.main() 