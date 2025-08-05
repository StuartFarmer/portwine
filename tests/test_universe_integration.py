"""
Integration tests for Universe class with Backtester and Strategy.
"""

import pytest
import pandas as pd
import os
from datetime import date, datetime
from portwine.universe import Universe, CSVUniverse
from portwine.strategies.base import StrategyBase
from portwine.backtester import Backtester
from portwine.loaders.eodhd import EODHDMarketDataLoader


class TestUniverseIntegration:
    """Integration tests for Universe with Backtester and Strategy."""
    
    def setup_method(self):
        """Create test files before each test."""
        self.test_files = []
        self.test_data_dir = "tests/test_data"
    
    def teardown_method(self):
        """Clean up test files after each test."""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def create_test_universe_csv(self, data, filename):
        """Helper to create test universe CSV file."""
        file_path = f"tests/test_data/{filename}"
        os.makedirs("tests/test_data", exist_ok=True)
        
        with open(file_path, 'w') as f:
            for row in data:
                f.write(f"{row[0]},{row[1]}\n")
        
        self.test_files.append(file_path)
        return file_path
    
    def test_static_universe_from_list(self):
        """Test that passing a list creates a static universe."""
        tickers = ["AAPL", "GOOGL", "MSFT"]
        
        class TestStrategy(StrategyBase):
            def step(self, current_date, daily_data):
                valid_tickers = [t for t in daily_data.keys() if daily_data.get(t) is not None]
                n = len(valid_tickers)
                weight = 1.0 / n if n > 0 else 0.0
                return {ticker: weight for ticker in valid_tickers}
        
        strategy = TestStrategy(tickers)
        
        # Verify universe was created
        assert strategy.universe is not None
        # Static universe should preserve deduplicated order as list
        # Should return a list with all tickers (order not guaranteed)
        assert isinstance(strategy.tickers, list)
        assert set(strategy.tickers) == {"AAPL", "GOOGL", "MSFT"}
        
        # Verify static universe behavior
        assert strategy.universe.get_constituents("2024-01-01") == {"AAPL", "GOOGL", "MSFT"}
        assert strategy.universe.get_constituents("1970-01-01") == {"AAPL", "GOOGL", "MSFT"}
        assert strategy.universe.get_constituents("2030-01-01") == {"AAPL", "GOOGL", "MSFT"}
        
        # Verify all_tickers
        assert strategy.universe.all_tickers == {"AAPL", "GOOGL", "MSFT"}
    
    def test_dynamic_universe(self):
        """Test that passing a universe object works correctly."""
        universe_data = [
            ["2024-01-01", "AAPL"],
            ["2024-02-01", "AAPL,MSFT"],
            ["2024-03-01", "MSFT"],
        ]
        
        universe_csv = self.create_test_universe_csv(universe_data, "dynamic_universe.csv")
        universe = CSVUniverse(universe_csv)
        
        class TestStrategy(StrategyBase):
            def step(self, current_date, daily_data):
                valid_tickers = [t for t in daily_data.keys() if daily_data.get(t) is not None]
                n = len(valid_tickers)
                weight = 1.0 / n if n > 0 else 0.0
                return {ticker: weight for ticker in valid_tickers}
        
        strategy = TestStrategy(universe)
        
        # Verify universe was used directly
        assert strategy.universe is universe
        # Dynamic universe may not preserve order; check set membership
        assert set(strategy.tickers) == {"AAPL", "MSFT"}
        
        # Verify dynamic universe behavior
        assert strategy.universe.get_constituents("2024-01-15") == {"AAPL"}
        assert strategy.universe.get_constituents("2024-02-15") == {"AAPL", "MSFT"}
        assert strategy.universe.get_constituents("2024-03-15") == {"MSFT"}
    
    def test_universe_strategy_integration(self):
        """Test that a strategy can use a universe object."""
        # Create a universe that alternates between AAPL and MSFT
        universe_data = [
            ["2024-01-01", "AAPL"],
            ["2024-02-01", "AAPL,MSFT"],
            ["2024-03-01", "MSFT"],
            ["2024-04-01", "MSFT,AAPL"],
            ["2024-05-01", "AAPL"],
        ]
        
        universe_csv = self.create_test_universe_csv(universe_data, "test_universe.csv")
        universe = CSVUniverse(universe_csv)
        
        # Create a simple strategy that uses the universe
        class UniverseStrategy(StrategyBase):
            def step(self, current_date, daily_data):
                # Strategy only sees current universe tickers in daily_data
                valid_tickers = [t for t in daily_data.keys() if daily_data.get(t) is not None]
                n = len(valid_tickers)
                weight = 1.0 / n if n > 0 else 0.0
                return {ticker: weight for ticker in valid_tickers}
        
        # Create strategy with universe
        strategy = UniverseStrategy(universe)
        
        # Test that strategy has access to universe
        assert strategy.universe is not None
        
        # Test that all_tickers contains all possible tickers
        all_tickers = strategy.universe.all_tickers
        assert all_tickers == {"AAPL", "MSFT"}
        
        # Test that strategy.tickers contains all possible tickers initially
        assert set(strategy.tickers) == {"AAPL", "MSFT"}
    
    def test_universe_backtester_integration(self):
        """Test full integration between Universe, Strategy, and Backtester."""
        # Create a universe that rotates through all available tickers
        universe_data = [
            ["2024-01-01", "AAPL"],
            ["2024-02-01", "AAPL,MSFT"],
            ["2024-03-01", "MSFT,NFLX"],
            ["2024-04-01", "NFLX,V"],
            ["2024-05-01", "V,AAPL"],
        ]
        
        universe_csv = self.create_test_universe_csv(universe_data, "integration_universe.csv")
        universe = CSVUniverse(universe_csv)
        
        # Create a strategy that tracks what data it receives
        class TrackingUniverseStrategy(StrategyBase):
            def __init__(self, tickers):
                super().__init__(tickers)
                self.step_calls = []
                self.data_received = []
            
            def step(self, current_date, daily_data):
                # Track what data we receive
                self.step_calls.append(current_date)
                self.data_received.append(list(daily_data.keys()))
                
                # Equal weight allocation among received tickers
                valid_tickers = [t for t in daily_data.keys() if daily_data.get(t) is not None]
                n = len(valid_tickers)
                weight = 1.0 / n if n > 0 else 0.0
                return {ticker: weight for ticker in valid_tickers}
        
        # Create strategy with universe
        strategy = TrackingUniverseStrategy(universe)
        
        # Set up data loader using existing test data
        data_loader = EODHDMarketDataLoader(data_path=self.test_data_dir)
        
        # Create backtester
        backtester = Backtester(market_data_loader=data_loader)
        
        # Run backtest
        results = backtester.run_backtest(
            strategy=strategy,
            start_date="2024-01-01",
            end_date="2024-05-31",
            verbose=False
        )
        
        # Verify results exist
        assert results is not None
        assert 'signals_df' in results
        assert 'strategy_returns' in results
        
        # Verify strategy was called
        assert len(strategy.step_calls) > 0
        
        # Verify that strategy only received current universe tickers
        # Early calls should only have AAPL
        early_data = strategy.data_received[0]
        assert "AAPL" in early_data
        assert "MSFT" not in early_data  # MSFT not in universe yet
        
        # Later calls should have different tickers
        later_data = strategy.data_received[-1]
        assert len(later_data) > 0  # Should have some tickers
        
        # Check that signals_df contains all possible tickers
        signals_df = results['signals_df']
        expected_tickers = {"AAPL", "MSFT", "NFLX", "V"}
        assert set(signals_df.columns) == expected_tickers
    
    def test_backtester_unified_interface(self):
        """Test that backtester works with unified universe interface."""
        # Test with static universe (from list)
        static_tickers = ["AAPL", "MSFT"]
        
        class StaticStrategy(StrategyBase):
            def step(self, current_date, daily_data):
                valid_tickers = [t for t in daily_data.keys() if daily_data.get(t) is not None]
                n = len(valid_tickers)
                weight = 1.0 / n if n > 0 else 0.0
                return {ticker: weight for ticker in valid_tickers}
        
        static_strategy = StaticStrategy(static_tickers)
        
        # Test with dynamic universe
        universe_data = [
            ["2024-01-01", "AAPL"],
            ["2024-02-01", "AAPL,MSFT"],
        ]
        
        universe_csv = self.create_test_universe_csv(universe_data, "test_unified.csv")
        dynamic_universe = CSVUniverse(universe_csv)
        
        class DynamicStrategy(StrategyBase):
            def step(self, current_date, daily_data):
                valid_tickers = [t for t in daily_data.keys() if daily_data.get(t) is not None]
                n = len(valid_tickers)
                weight = 1.0 / n if n > 0 else 0.0
                return {ticker: weight for ticker in valid_tickers}
        
        dynamic_strategy = DynamicStrategy(dynamic_universe)
        
        # Set up backtester
        data_loader = EODHDMarketDataLoader(data_path=self.test_data_dir)
        backtester = Backtester(market_data_loader=data_loader)
        
        # Both strategies should work with the same backtester interface
        static_results = backtester.run_backtest(
            strategy=static_strategy,
            start_date="2024-01-01",
            end_date="2024-02-28",
            verbose=False
        )
        
        dynamic_results = backtester.run_backtest(
            strategy=dynamic_strategy,
            start_date="2024-01-01",
            end_date="2024-02-28",
            verbose=False
        )
        
        # Both should produce results
        assert static_results is not None
        assert dynamic_results is not None
        
        # Static strategy should have consistent allocations
        static_signals = static_results['signals_df']
        assert "AAPL" in static_signals.columns
        assert "MSFT" in static_signals.columns
        
        # Dynamic strategy should have changing allocations
        dynamic_signals = dynamic_results['signals_df']
        assert "AAPL" in dynamic_signals.columns
        assert "MSFT" in dynamic_signals.columns
    
    def test_strategy_cannot_assign_weights_to_tickers_not_in_universe(self):
        """Test that backtester fails when strategy tries to assign weights to tickers not in current universe."""
        # Create a universe that only has AAPL initially
        universe_data = [
            ["2024-01-01", "AAPL"],
            ["2024-02-01", "AAPL,MSFT"],
        ]
        
        universe_csv = self.create_test_universe_csv(universe_data, "test_invalid_weights.csv")
        universe = CSVUniverse(universe_csv)
        
        # Create a strategy that tries to assign weights to tickers not in the current universe
        class InvalidStrategy(StrategyBase):
            def step(self, current_date, daily_data):
                # This strategy tries to assign weights to MSFT even when it's not in the universe
                # This should cause the backtester to fail
                return {"AAPL": 0.5, "MSFT": 0.5}  # MSFT not in universe on 2024-01-01
        
        strategy = InvalidStrategy(universe)
        
        # Set up backtester
        data_loader = EODHDMarketDataLoader(data_path=self.test_data_dir)
        backtester = Backtester(market_data_loader=data_loader)
        
        # This should fail because the strategy is trying to assign weights to MSFT
        # when it's not in the current universe
        with pytest.raises(ValueError, match=".*not in current universe.*"):
            backtester.run_backtest(
                strategy=strategy,
                start_date="2024-01-01",
                end_date="2024-01-31",
                verbose=False
            )
    
    def test_strategy_can_assign_weights_to_tickers_in_universe(self):
        """Test that strategy can assign weights to tickers that are in the current universe."""
        # Create a universe that has AAPL initially, then adds MSFT
        universe_data = [
            ["2024-01-01", "AAPL"],
            ["2024-02-01", "AAPL,MSFT"],
        ]
        
        universe_csv = self.create_test_universe_csv(universe_data, "test_valid_weights.csv")
        universe = CSVUniverse(universe_csv)
        
        # Create a strategy that assigns weights to tickers in the current universe
        class ValidStrategy(StrategyBase):
            def step(self, current_date, daily_data):
                # This strategy only assigns weights to tickers that are in the current universe
                valid_tickers = [t for t in daily_data.keys() if daily_data.get(t) is not None]
                n = len(valid_tickers)
                weight = 1.0 / n if n > 0 else 0.0
                return {ticker: weight for ticker in valid_tickers}
        
        strategy = ValidStrategy(universe)
        
        # Set up backtester
        data_loader = EODHDMarketDataLoader(data_path=self.test_data_dir)
        backtester = Backtester(market_data_loader=data_loader)
        
        # This should work because the strategy only assigns weights to tickers in the universe
        results = backtester.run_backtest(
            strategy=strategy,
            start_date="2024-01-01",
            end_date="2024-02-28",
            verbose=False
        )
        
        assert results is not None
        assert 'signals_df' in results
        
        # Verify that weights are correctly assigned
        signals_df = results['signals_df']
        
        # Early period should only have AAPL with weight 1.0
        early_date = pd.Timestamp("2024-01-15")
        if early_date in signals_df.index:
            early_signals = signals_df.loc[early_date]
            assert early_signals["AAPL"] == 1.0
            assert early_signals["MSFT"] == 0.0  # MSFT not in universe yet
        
        # Later period should have both with equal weights
        later_date = pd.Timestamp("2024-02-15")
        if later_date in signals_df.index:
            later_signals = signals_df.loc[later_date]
            assert later_signals["AAPL"] == 0.5
            assert later_signals["MSFT"] == 0.5 