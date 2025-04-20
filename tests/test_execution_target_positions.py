import unittest
from datetime import timezone
import os

from portwine.execution.base import ExecutionBase
from portwine.execution.base import PortfolioExceededError
from portwine.execution.base import PortfolioExceededError
from portwine.loaders.eodhd import EODHDMarketDataLoader


class DummyStrategy:
    def __init__(self, tickers):
        self.tickers = tickers


def make_exec_base():
    # market_data_loader and broker not used for target positions
    return ExecutionBase(
        strategy=DummyStrategy(tickers=['AAPL', 'MSFT']),
        market_data_loader=None,
        broker=None,
        alternative_data_loader=None,
        timezone=timezone.utc
    )


class TestCalculateTargetPositions(unittest.TestCase):
    def setUp(self):
        self.exec_base = make_exec_base()
        self.portfolio_value = 100_000.0
        self.prices = {'AAPL': 100.0, 'MSFT': 200.0, 'X': 50.0}

    def test_all_in_one_ticker(self):
        """All-in on AAPL yields 1000 shares, MSFT zero."""
        target_weights = {"AAPL": 1.0, "MSFT": 0.0}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, self.prices)
        self.assertEqual(positions["AAPL"], 1000)
        self.assertEqual(positions["MSFT"], 0)

    def test_mixed_tickers(self):
        """Mixed equal weights yields correct shares."""
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, self.prices)
        self.assertEqual(positions["AAPL"], 500)
        self.assertEqual(positions["MSFT"], 250)

    def test_no_fractional_rounds_down(self):
        """Non-fractional rounding for MSFT."""
        prices = {'AAPL': 100.0, 'MSFT': 200.1}
        target_weights = {"AAPL": 0.7, "MSFT": 0.3}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, prices, fractional=False)
        # AAPL exact
        self.assertEqual(positions["AAPL"], 700)
        # MSFT raw = 30000/200.1 ≈ 149.93 -> floor -> 149
        self.assertEqual(positions["MSFT"], 149)

    def test_fractional_keeps_value(self):
        """Fractional True preserves raw values."""
        prices = {'AAPL': 100.0, 'MSFT': 200.1}
        target_weights = {"AAPL": 0.7, "MSFT": 0.3}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, prices, fractional=True)
        self.assertEqual(positions["AAPL"], 700)
        self.assertAlmostEqual(positions["MSFT"], 30000.0/200.1, places=8)

    def test_symbols_not_in_prices_skipped(self):
        """Symbols missing price are skipped."""
        target_weights = {"AAPL": 1.0, "UNKNOWN": 0.5}
        prices = {"AAPL": 100.0}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, prices)
        self.assertIn("AAPL", positions)
        self.assertNotIn("UNKNOWN", positions)


class TestCalculateTargetPositionsWithRealData(unittest.TestCase):
    """Tests _calculate_target_positions using real EODHD test data"""
    @classmethod
    def setUpClass(cls):
        data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        loader = EODHDMarketDataLoader(data_path=data_dir)
        cls.df_aapl = loader.load_ticker('AAPL')
        cls.df_msft = loader.load_ticker('MSFT')
        cls.exec_base = make_exec_base()
        cls.portfolio_value = 100_000.0

    def test_real_all_in_one_ticker(self):
        # Use first available date
        dt = self.df_aapl.index[0]
        price_aapl = float(self.df_aapl.loc[dt, 'close'])
        price_msft = float(self.df_msft.loc[dt, 'close'])
        target_weights = {'AAPL': 1.0, 'MSFT': 0.0}
        prices = {'AAPL': price_aapl, 'MSFT': price_msft}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, prices, fractional=False
        )
        # Expected AAPL shares = portfolio_value / price
        expected_shares = int(self.portfolio_value / price_aapl)
        self.assertEqual(positions['AAPL'], expected_shares)
        self.assertEqual(positions['MSFT'], 0)

    def test_real_mixed_tickers(self):
        # Use first available date
        dt = self.df_aapl.index[0]
        price_aapl = float(self.df_aapl.loc[dt, 'close'])
        price_msft = float(self.df_msft.loc[dt, 'close'])
        target_weights = {'AAPL': 0.5, 'MSFT': 0.5}
        prices = {'AAPL': price_aapl, 'MSFT': price_msft}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, prices, fractional=False
        )
        self.assertEqual(positions['AAPL'], int((self.portfolio_value * 0.5) / price_aapl))
        self.assertEqual(positions['MSFT'], int((self.portfolio_value * 0.5) / price_msft))

    def test_real_no_fractional_rounds_down(self):
        # Use date where MSFT close price is not an integer divisor
        dt = self.df_aapl.index[1]
        price_aapl = float(self.df_aapl.loc[dt, 'close'])
        price_msft = float(self.df_msft.loc[dt, 'close'])
        target_weights = {'AAPL': 0.7, 'MSFT': 0.3}
        prices = {'AAPL': price_aapl, 'MSFT': price_msft}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, prices, fractional=False
        )
        # MSFT should floor down
        raw_msft = (self.portfolio_value * 0.3) / price_msft
        import math
        self.assertEqual(positions['MSFT'], math.floor(raw_msft))

    def test_real_fractional_keeps(self):
        # Use same date as above
        dt = self.df_aapl.index[1]
        price_aapl = float(self.df_aapl.loc[dt, 'close'])
        price_msft = float(self.df_msft.loc[dt, 'close'])
        target_weights = {'AAPL': 0.7, 'MSFT': 0.3}
        prices = {'AAPL': price_aapl, 'MSFT': price_msft}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, prices, fractional=True
        )
        self.assertAlmostEqual(positions['AAPL'], (self.portfolio_value * 0.7) / price_aapl)
        self.assertAlmostEqual(positions['MSFT'], (self.portfolio_value * 0.3) / price_msft)


if __name__ == '__main__':
    unittest.main() 