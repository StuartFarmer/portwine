"""
Tests for the DailyExecutor class.

This test suite uses mock components to test the scheduling and execution_complex
functionality of the DailyExecutor without requiring actual market data
or trading systems.
"""

import unittest
import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call

import pytz
import pandas as pd

# Import the module directly to fix class resolution issues
from portwine.utils.daily_executor import DailyExecutor

# Define mock classes directly in the test file to avoid namespace issues
class MockStrategy:
    """Mock strategy for testing."""
    
    def __init__(self, tickers, **kwargs):
        self.tickers = tickers
        self.params = kwargs
        self.called_shutdown = False
    
    def generate_signals(self):
        """Return mock signals."""
        return {"AAPL": 0.5, "MSFT": 0.5}
    
    def shutdown(self):
        """Record shutdown was called."""
        self.called_shutdown = True


class MockExecutor:
    """Mock execution_complex system for testing."""
    
    def __init__(self, strategy=None, market_data_loader=None, **kwargs):
        self.strategy = strategy
        self.market_data_loader = market_data_loader
        self.params = kwargs
        self.executed = False
        self.called_shutdown = False
    
    def execute(self):
        """Record execution_complex was called."""
        self.executed = True
        return True
    
    def step(self):
        """Alternative execution_complex method."""
        return self.execute()
    
    def shutdown(self):
        """Record shutdown was called."""
        self.called_shutdown = True


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.updated = False
        self.called_shutdown = False
    
    def update_data(self):
        """Record update was called."""
        self.updated = True
        return True
    
    def shutdown(self):
        """Record shutdown was called."""
        self.called_shutdown = True


class TestDailyExecutor(unittest.TestCase):
    """Test cases for the DailyExecutor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a basic config for testing
        self.config = {
            "strategy": {
                "class": "__main__.MockStrategy",
                "tickers": ["AAPL", "MSFT"],
                "params": {
                    "lookback_days": 5
                }
            },
            "execution_complex": {
                "class": "__main__.MockExecutor",
                "params": {
                    "paper": True
                }
            },
            "data_loader": {
                "class": "__main__.MockDataLoader",
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
    
    def test_initialization(self):
        """Test initialization of executor."""
        # Patch _import_class to return the mock classes
        with patch.object(DailyExecutor, '_import_class') as mock_import:
            # Set up the side effect to return the proper class
            def side_effect(class_path):
                if class_path == "__main__.MockStrategy":
                    return MockStrategy
                elif class_path == "__main__.MockExecutor":
                    return MockExecutor
                elif class_path == "__main__.MockDataLoader":
                    return MockDataLoader
                return None
            
            mock_import.side_effect = side_effect
            
            # Create executor with provided config
            executor = DailyExecutor(self.config)
            executor.initialize()
            
            # Check components are correctly initialized
            self.assertIsInstance(executor.strategy, MockStrategy)
            self.assertIsInstance(executor.executor, MockExecutor)
            self.assertIsInstance(executor.data_loader, MockDataLoader)
            
            # Check config parsing
            self.assertEqual(executor.run_time, "15:45")
            self.assertEqual(executor.time_zone, "US/Eastern")
            self.assertEqual(executor.days, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
            self.assertTrue(executor.market_hours_only)
    
    def test_run_once(self):
        """Test run_once method."""
        # Patch _import_class to return the mock classes
        with patch.object(DailyExecutor, '_import_class') as mock_import:
            # Set up the side effect to return the proper class
            def side_effect(class_path):
                if class_path == "__main__.MockStrategy":
                    return MockStrategy
                elif class_path == "__main__.MockExecutor":
                    return MockExecutor
                elif class_path == "__main__.MockDataLoader":
                    return MockDataLoader
                return None
            
            mock_import.side_effect = side_effect
            
            # Create and initialize executor
            executor = DailyExecutor(self.config)
            executor.initialize()
            
            # Mock market is open
            with patch.object(executor, '_is_market_open', return_value=True):
                # Run once and check results
                result = executor.run_once()
                
                # Check data was updated and execution_complex was performed
                self.assertTrue(result)
                self.assertTrue(executor.data_loader.updated)
                self.assertTrue(executor.executor.executed)
    
    def test_is_market_open(self):
        """Test _is_market_open method."""
        # Create executor with market_hours_only=True
        executor = DailyExecutor(self.config)
        
        # During market hours
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.weekday.return_value = 0  # Monday
            mock_datetime.now.return_value.hour = 10  # 10 AM
            mock_datetime.now.return_value.minute = 30
            
            # Should return True during market hours
            self.assertTrue(executor._is_market_open())
        
        # Outside market hours
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.weekday.return_value = 0  # Monday
            mock_datetime.now.return_value.hour = 3  # 3 AM
            mock_datetime.now.return_value.minute = 30
            
            # Should return False outside market hours
            self.assertFalse(executor._is_market_open())
        
        # Weekend
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.weekday.return_value = 5  # Saturday
            mock_datetime.now.return_value.hour = 10  # 10 AM
            mock_datetime.now.return_value.minute = 30
            
            # Should return False on weekend
            self.assertFalse(executor._is_market_open())
        
        # With market_hours_only=False
        config_no_market_hours = self.config.copy()
        config_no_market_hours["schedule"]["market_hours_only"] = False
        executor = DailyExecutor(config_no_market_hours)
        
        # Outside market hours but market_hours_only=False
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.weekday.return_value = 0  # Monday
            mock_datetime.now.return_value.hour = 3  # 3 AM
            mock_datetime.now.return_value.minute = 30
            
            # Should return True if market_hours_only=False
            self.assertTrue(executor._is_market_open())
    
    def test_setup_intraday_schedule(self):
        """Test setup_intraday_schedule method."""
        # Patch _import_class to return the mock classes
        with patch.object(DailyExecutor, '_import_class') as mock_import:
            # Set up the side effect to return the proper class
            def side_effect(class_path):
                if class_path == "__main__.MockStrategy":
                    return MockStrategy
                elif class_path == "__main__.MockExecutor":
                    return MockExecutor
                elif class_path == "__main__.MockDataLoader":
                    return MockDataLoader
                return None
            
            mock_import.side_effect = side_effect
            
            # Create executor with intraday config
            executor = DailyExecutor.from_config_file(self.intraday_config_path)
            
            # Check intraday settings
            self.assertEqual(executor.intraday_schedule, "interval:30m")
            
            # Initialize executor
            executor.initialize()
            
            # Mock schedule module
            with patch('portwine.utils.daily_executor.schedule') as mock_schedule:
                # Setup mock for method chaining
                mock_every = MagicMock()
                mock_schedule.every.return_value = mock_every
                
                mock_minutes = MagicMock()
                mock_every.minutes.return_value = mock_minutes
                
                # Call setup_intraday_schedule
                executor.setup_intraday_schedule()
                
                # Verify correct schedule calls
                mock_schedule.every.assert_called_with(30)
                mock_every.minutes.assert_called_once()
                mock_minutes.do.assert_called_with(executor.run_once)
    
    def test_run_scheduled_with_intraday(self):
        """Test run_scheduled with intraday config."""
        # Patch _import_class to return the mock classes
        with patch.object(DailyExecutor, '_import_class') as mock_import:
            # Set up the side effect to return the proper class
            def side_effect(class_path):
                if class_path == "__main__.MockStrategy":
                    return MockStrategy
                elif class_path == "__main__.MockExecutor":
                    return MockExecutor
                elif class_path == "__main__.MockDataLoader":
                    return MockDataLoader
                return None
            
            mock_import.side_effect = side_effect
            
            # Create executor with intraday config
            executor = DailyExecutor.from_config_file(self.intraday_config_path)
            
            # Initialize executor
            executor.initialize()
            
            # Mock schedule module
            with patch('portwine.utils.daily_executor.schedule') as mock_schedule, \
                 patch.object(executor, 'setup_intraday_schedule') as mock_setup_intraday:
                 
                # Configure mock run_pending to raise KeyboardInterrupt after first call
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
                
                # Run scheduled, should exit on KeyboardInterrupt
                try:
                    executor.run_scheduled()
                except KeyboardInterrupt:
                    pass
                
                # Check setup_intraday_schedule was called
                mock_setup_intraday.assert_called_once()
    
    def test_shutdown(self):
        """Test shutdown method."""
        # Patch _import_class to return the mock classes
        with patch.object(DailyExecutor, '_import_class') as mock_import:
            # Set up the side effect to return the proper class
            def side_effect(class_path):
                if class_path == "__main__.MockStrategy":
                    return MockStrategy
                elif class_path == "__main__.MockExecutor":
                    return MockExecutor
                elif class_path == "__main__.MockDataLoader":
                    return MockDataLoader
                return None
            
            mock_import.side_effect = side_effect
            
            # Create and initialize executor
            executor = DailyExecutor(self.config)
            executor.initialize()
            
            # Call shutdown
            executor.shutdown()
            
            # Check all components were shutdown
            self.assertTrue(executor.strategy.called_shutdown)
            self.assertTrue(executor.executor.called_shutdown)
            self.assertTrue(executor.data_loader.called_shutdown)
    
    def test_from_config_file(self):
        """Test from_config_file static method."""
        # Patch _import_class to return the mock classes when initialized
        with patch.object(DailyExecutor, '_import_class') as mock_import:
            # Set up the side effect to return the proper class
            def side_effect(class_path):
                if class_path == "__main__.MockStrategy":
                    return MockStrategy
                elif class_path == "__main__.MockExecutor":
                    return MockExecutor
                elif class_path == "__main__.MockDataLoader":
                    return MockDataLoader
                return None
            
            mock_import.side_effect = side_effect
            
            # Create from config file
            executor = DailyExecutor.from_config_file(self.config_path)
            
            # Check basic config parsing
            self.assertEqual(executor.run_time, "15:45")
            self.assertEqual(executor.time_zone, "US/Eastern")
            
            # Initialize and check components
            executor.initialize()
            self.assertIsInstance(executor.strategy, MockStrategy)
            self.assertIsInstance(executor.executor, MockExecutor)
            self.assertIsInstance(executor.data_loader, MockDataLoader)


if __name__ == '__main__':
    unittest.main() 