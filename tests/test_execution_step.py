import unittest
from datetime import timezone

from portwine.execution import ExecutionBase
from portwine.brokers.base import Order, Position, Account
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader


class DummyLoader(MarketDataLoader):
    def __init__(self, prices):
        self.prices = prices

    def next(self, tickers, dt):
        # Return bar data with 'close' prices for each ticker
        return {ticker: {'close': self.prices.get(ticker, 0.0)} for ticker in tickers}


class DummyBroker:
    """A fake broker for step() testing."""
    def __init__(self, positions: dict, equity: float):
        self._positions = positions
        self._account = Account(equity=equity, last_updated_at=0)
        self.calls = []

    def get_account(self) -> Account:
        return self._account

    def get_positions(self) -> dict:
        # Return Position objects for each ticker
        return {t: Position(t, q, 0) for t, q in self._positions.items()}

    def market_is_open(self, dt) -> bool:
        return True

    def submit_order(self, symbol: str, quantity: float) -> Order:
        # Record call and return an Order object
        self.calls.append((symbol, quantity))
        return Order(
            order_id=f"id-{symbol}",
            ticker=symbol,
            side='buy' if quantity > 0 else 'sell',
            quantity=quantity,
            order_type='market',
            status='filled',
            time_in_force='day',
            average_price=0.0,
            remaining_quantity=0.0,
            created_at=0,
            last_updated_at=0,
        )


def make_exec_base(tickers, loader, broker):
    return ExecutionBase(
        strategy=StrategyBase(tickers),
        market_data_loader=loader,
        broker=broker,
        alternative_data_loader=None,
        timezone=timezone.utc
    )


class TestExecutionStep(unittest.TestCase):
    def test_initial_buy_single(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = DummyLoader(prices)
        broker = DummyBroker({'AAPL': 0.0, 'MSFT': 0.0}, equity=100_000)
        # Strategy returns weights: full AAPL, none for MSFT
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
            def step(self, dt, data):
                return {'AAPL': 1.0, 'MSFT': 0.0}
        strategy = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strategy

        executed = exec_base.step(0)
        # Only AAPL buy: 100_000 / 100 = 1000 shares
        self.assertEqual(len(executed), 1)
        order = executed[0]
        self.assertIsInstance(order, Order)
        self.assertEqual(order.ticker, 'AAPL')
        self.assertEqual(order.quantity, 1000.0)
        self.assertEqual(order.side, 'buy')
        # Broker saw the submission
        self.assertEqual(broker.calls, [('AAPL', 1000.0)])

    def test_initial_buy_multiple(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = DummyLoader(prices)
        broker = DummyBroker({'AAPL': 0.0, 'MSFT': 0.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
            def step(self, dt, data):
                return {'AAPL': 0.5, 'MSFT': 0.5}
        strategy = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strategy

        executed = exec_base.step(0)
        # AAPL: 100_000 * 0.5 / 100 = 500, MSFT: 100_000 * 0.5 / 200 = 250
        expected = [('AAPL', 500.0, 'buy'), ('MSFT', 250.0, 'buy')]
        self.assertEqual(len(executed), 2)
        for exp, order in zip(expected, executed):
            self.assertEqual(order.ticker, exp[0])
            self.assertEqual(order.quantity, exp[1])
            self.assertEqual(order.side, exp[2])
        self.assertEqual(broker.calls, [(e[0], e[1]) for e in expected])

    def test_add_on_buy_single(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = DummyLoader(prices)
        # Current AAPL 500 shares
        broker = DummyBroker({'AAPL': 500.0, 'MSFT': 0.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
            def step(self, dt, data):
                return {'AAPL': 1.0, 'MSFT': 0.0}
        strategy = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strategy

        executed = exec_base.step(0)
        # AAPL target = 1000, change = 500
        self.assertEqual(len(executed), 1)
        order = executed[0]
        self.assertEqual(order.ticker, 'AAPL')
        self.assertEqual(order.quantity, 500.0)
        self.assertEqual(order.side, 'buy')

    def test_add_on_buy_multiple(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = DummyLoader(prices)
        # Current positions: AAPL 250, MSFT 100
        broker = DummyBroker({'AAPL': 250.0, 'MSFT': 100.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
            def step(self, dt, data):
                return {'AAPL': 0.5, 'MSFT': 0.5}
        strategy = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strategy

        executed = exec_base.step(0)
        # AAPL: 500-250=250 buy, MSFT: 250-100=150 buy
        expected = [('AAPL', 250.0, 'buy'), ('MSFT', 150.0, 'buy')]
        self.assertEqual(len(executed), 2)
        for exp, order in zip(expected, executed):
            self.assertEqual(order.ticker, exp[0])
            self.assertEqual(order.quantity, exp[1])
            self.assertEqual(order.side, exp[2])

    def test_reduce_single(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = DummyLoader(prices)
        # Current AAPL 1000 shares
        broker = DummyBroker({'AAPL': 1000.0, 'MSFT': 0.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
            def step(self, dt, data):
                return {'AAPL': 0.5, 'MSFT': 0.0}
        strategy = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strategy

        executed = exec_base.step(0)
        # AAPL target = 500, change = -500 -> sell
        self.assertEqual(len(executed), 1)
        order = executed[0]
        self.assertEqual(order.ticker, 'AAPL')
        self.assertEqual(order.quantity, 500.0)
        self.assertEqual(order.side, 'sell')

    def test_reduce_multiple(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = DummyLoader(prices)
        # Current positions: AAPL 500, MSFT 250
        broker = DummyBroker({'AAPL': 500.0, 'MSFT': 250.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
            def step(self, dt, data):
                return {'AAPL': 0.25, 'MSFT': 0.25}
        strategy = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strategy

        executed = exec_base.step(0)
        expected = [('AAPL', 250.0, 'sell'), ('MSFT', 125.0, 'sell')]
        self.assertEqual(len(executed), 2)
        for exp, order in zip(expected, executed):
            self.assertEqual(order.ticker, exp[0])
            self.assertEqual(order.quantity, exp[1])
            self.assertEqual(order.side, exp[2])

    def test_mixed_add_and_reduce(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = DummyLoader(prices)
        # Current positions: AAPL 500, MSFT 250
        broker = DummyBroker({'AAPL': 500.0, 'MSFT': 250.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
            def step(self, dt, data):
                return {'AAPL': 0.25, 'MSFT': 0.75}
        strategy = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strategy

        executed = exec_base.step(0)
        expected = [('AAPL', 250.0, 'sell'), ('MSFT', 125.0, 'buy')]
        self.assertEqual(len(executed), 2)
        for exp, order in zip(expected, executed):
            self.assertEqual(order.ticker, exp[0])
            self.assertEqual(order.quantity, exp[1])
            self.assertEqual(order.side, exp[2])

    def test_initial_buy_and_add_on(self):
        prices = {'AAPL': 100.0, 'MSFT': 100.0}
        loader = DummyLoader(prices)
        # Current: AAPL 0, MSFT 10
        broker = DummyBroker({'AAPL': 0.0, 'MSFT': 10.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
            def step(self, dt, data):
                return {'AAPL': 1.0, 'MSFT': 0.1}
        strategy = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strategy

        executed = exec_base.step(0)
        # AAPL buy 1000, MSFT buy (100k*0.1/100=100)-10=90
        expected = [('AAPL', 1000.0, 'buy'), ('MSFT', 90.0, 'buy')]
        self.assertEqual(len(executed), 2)
        for exp, order in zip(expected, executed):
            self.assertEqual(order.ticker, exp[0])
            self.assertEqual(order.quantity, exp[1])
            self.assertEqual(order.side, exp[2])

    def test_initial_buy_and_sell(self):
        prices = {'AAPL': 100.0, 'MSFT': 100.0}
        loader = DummyLoader(prices)
        # Current: AAPL 0, MSFT 100
        broker = DummyBroker({'AAPL': 0.0, 'MSFT': 100.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
            def step(self, dt, data):
                return {'AAPL': 1.0, 'MSFT': 0.0}
        strategy = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strategy

        executed = exec_base.step(0)
        # AAPL buy 1000, MSFT sell 100
        expected = [('AAPL', 1000.0, 'buy'), ('MSFT', 100.0, 'sell')]
        self.assertEqual(len(executed), 2)
        for exp, order in zip(expected, executed):
            self.assertEqual(order.ticker, exp[0])
            self.assertEqual(order.quantity, exp[1])
            self.assertEqual(order.side, exp[2])


if __name__ == '__main__':
    unittest.main() 