import unittest
from datetime import timezone

from portwine.execution.base import ExecutionBase, PortfolioExceededError


class DummyStrategy:
    def __init__(self, tickers):
        self.tickers = tickers


def make_exec_base(tickers):
    # market_data_loader and broker not used for weight calc
    return ExecutionBase(
        strategy=DummyStrategy(tickers=tickers),
        market_data_loader=None,
        broker=None,
        alternative_data_loader=None,
        timezone=timezone.utc
    )


class TestCalculateCurrentWeights(unittest.TestCase):
    def test_single_ticker(self):
        """Current weights for a single ticker."""
        exec_base = make_exec_base(['AAPL'])
        positions = [('AAPL', 20.0)]
        portfolio_value = 100_000.0
        prices = {'AAPL': 200.0}
        weights = exec_base._calculate_current_weights(positions, portfolio_value, prices)
        self.assertIn('AAPL', weights)
        self.assertAlmostEqual(weights['AAPL'], 0.04)

    def test_includes_ticker_with_no_position(self):
        """Weights includes tickers with zero positions."""
        exec_base = make_exec_base(['AAPL', 'MSFT'])
        positions = [('AAPL', 10.0)]
        portfolio_value = 100_000.0
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        weights = exec_base._calculate_current_weights(positions, portfolio_value, prices)
        self.assertAlmostEqual(weights['AAPL'], 0.01)
        self.assertAlmostEqual(weights['MSFT'], 0.0)

    def test_multiple_tickers(self):
        """Weights for multiple tickers with positions."""
        exec_base = make_exec_base(['AAPL', 'MSFT'])
        positions = [('AAPL', 10.0), ('MSFT', 20.0)]
        portfolio_value = 100_000.0
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        weights = exec_base._calculate_current_weights(positions, portfolio_value, prices)
        self.assertAlmostEqual(weights['AAPL'], 0.01)
        self.assertAlmostEqual(weights['MSFT'], 0.04)

    def test_exceeds_portfolio_raises(self):
        """Raise when total weights exceed 1 and raises=True."""
        exec_base = make_exec_base(['AAPL', 'MSFT'])
        positions = [('AAPL', 1000.0), ('MSFT', 20.0)]
        portfolio_value = 100_000.0
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        with self.assertRaises(PortfolioExceededError):
            exec_base._calculate_current_weights(positions, portfolio_value, prices, raises=True)

    def test_exceeds_portfolio_returns_when_not_raising(self):
        """Return clipped weights even if sum > 1 when raises=False."""
        exec_base = make_exec_base(['AAPL', 'MSFT'])
        positions = [('AAPL', 1000.0), ('MSFT', 20.0)]
        portfolio_value = 100_000.0
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        weights = exec_base._calculate_current_weights(positions, portfolio_value, prices, raises=False)
        self.assertAlmostEqual(weights['AAPL'], 1.0)
        self.assertAlmostEqual(weights['MSFT'], 0.04)


if __name__ == '__main__':
    unittest.main() 