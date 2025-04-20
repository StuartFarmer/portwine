import unittest
from datetime import timezone

from portwine.execution.base import ExecutionBase
from portwine.execution.brokers.base import Order, Position, Account
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader


class MultiStepLoader(MarketDataLoader):
    def __init__(self, prices):
        self.prices = prices

    def next(self, tickers, dt):
        # Return bar data with 'close' prices for each ticker
        return {t: {'close': self.prices.get(t, 0.0)} for t in tickers}


class MultiStepBroker:
    """Broker that keeps track of positions via submit_order."""
    def __init__(self, equity: float):
        self._positions = {}
        self._account = Account(equity=equity, last_updated_at=0)
        self.calls = []

    def get_account(self) -> Account:
        return self._account

    def get_positions(self) -> dict:
        return {t: Position(t, q, 0) for t, q in self._positions.items()}

    def market_is_open(self, dt) -> bool:
        return True

    def submit_order(self, symbol: str, quantity: float) -> Order:
        # track call
        self.calls.append((symbol, quantity))
        # update position: buy positive, sell negative
        prev = self._positions.get(symbol, 0.0)
        new = prev + quantity
        self._positions[symbol] = new
        side = 'buy' if quantity > 0 else 'sell'
        qty = quantity if quantity > 0 else -quantity
        return Order(
            order_id=f"id-{symbol}{len(self.calls)}",
            ticker=symbol,
            side=side,
            quantity=qty,
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


class TestExecutionStepMulti(unittest.TestCase):
    def test_initial_buy_single(self):
        prices = {'AAPL': 100.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL'])
            def step(self, dt, data):
                return {'AAPL': 1.0}
        strat = Strat()
        exec_base = make_exec_base(['AAPL'], loader, broker)
        exec_base.strategy = strat

        # Day 1: initial buy
        orders1 = exec_base.step(0)
        self.assertEqual(len(orders1), 1)
        self.assertEqual(orders1[0].ticker, 'AAPL')
        self.assertEqual(orders1[0].quantity, 1000.0)
        self.assertEqual(orders1[0].side, 'buy')

        # Day 2: no change
        orders2 = exec_base.step(0)
        self.assertEqual(orders2, [])

    def test_initial_buy_multiple(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
            def step(self, dt, data):
                return {'AAPL': 0.5, 'MSFT': 0.5}
        strat = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strat

        # Day 1: buy both
        orders1 = exec_base.step(0)
        expected1 = [('AAPL', 500.0, 'buy'), ('MSFT', 250.0, 'buy')]
        self.assertEqual(len(orders1), 2)
        for (t, q, s), o in zip(expected1, orders1):
            self.assertEqual(o.ticker, t)
            self.assertEqual(o.quantity, q)
            self.assertEqual(o.side, s)

        # Day 2: no further change
        orders2 = exec_base.step(0)
        self.assertEqual(orders2, [])

    def test_add_on_buy_single(self):
        prices = {'AAPL': 100.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL'])
                self.day = 0
            def step(self, dt, data):
                self.day += 1
                return {'AAPL': 0.9} if self.day == 1 else {'AAPL': 1.0}
        strat = Strat()
        exec_base = make_exec_base(['AAPL'], loader, broker)
        exec_base.strategy = strat

        exec_base.step(0)  # Day1 buys 900
        orders2 = exec_base.step(0)
        # Day2 buys remaining 100
        self.assertEqual(len(orders2), 1)
        o = orders2[0]
        self.assertEqual(o.ticker, 'AAPL')
        self.assertEqual(o.quantity, 100.0)
        self.assertEqual(o.side, 'buy')

    def test_add_on_buy_multiple(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
                self.day = 0
            def step(self, dt, data):
                self.day += 1
                if self.day == 1:
                    return {'AAPL': 0.9, 'MSFT': 0.02}
                return {'AAPL': 1.0, 'MSFT': 0.05}
        strat = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strat

        exec_base.step(0)  # Day1 buys 900, 10
        orders2 = exec_base.step(0)
        expected2 = [('AAPL', 100.0, 'buy'), ('MSFT', 15.0, 'buy')]
        self.assertEqual(len(orders2), 2)
        for (t, q, s), o in zip(expected2, orders2):
            self.assertEqual(o.ticker, t)
            self.assertEqual(o.quantity, q)
            self.assertEqual(o.side, s)

    def test_reduce_single(self):
        prices = {'AAPL': 100.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL'])
                self.day = 0
            def step(self, dt, data):
                self.day += 1
                return {'AAPL': 1.0} if self.day == 1 else {'AAPL': 0.4}
        strat = Strat()
        exec_base = make_exec_base(['AAPL'], loader, broker)
        exec_base.strategy = strat

        exec_base.step(0)  # Day1 buy 1000
        orders2 = exec_base.step(0)
        self.assertEqual(len(orders2), 1)
        o = orders2[0]
        self.assertEqual(o.ticker, 'AAPL')
        self.assertEqual(o.quantity, 600.0)
        self.assertEqual(o.side, 'sell')

    def test_reduce_multiple(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
                self.day = 0
            def step(self, dt, data):
                self.day += 1
                return {'AAPL': 0.9, 'MSFT': 0.1} if self.day == 1 else {'AAPL': 0.4, 'MSFT': 0.02}
        strat = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strat

        exec_base.step(0)  # Day1 buy 900,50
        orders2 = exec_base.step(0)
        expected2 = [('AAPL', 500.0, 'sell'), ('MSFT', 40.0, 'sell')]
        self.assertEqual(len(orders2), 2)
        for (t, q, s), o in zip(expected2, orders2):
            self.assertEqual(o.ticker, t)
            self.assertEqual(o.quantity, q)
            self.assertEqual(o.side, s)

    def test_mixed_add_and_reduce(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
                self.day = 0
            def step(self, dt, data):
                self.day += 1
                if self.day == 1:
                    return {'AAPL': 0.5, 'MSFT': 0.1}
                return {'AAPL': 1.0, 'MSFT': 0.05}
        strat = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strat

        exec_base.step(0)  # Day1 buy 500,50
        orders2 = exec_base.step(0)
        expected2 = [('AAPL', 500.0, 'buy'), ('MSFT', 25.0, 'sell')]
        self.assertEqual(len(orders2), 2)
        for (t, q, s), o in zip(expected2, orders2):
            self.assertEqual(o.ticker, t)
            self.assertEqual(o.quantity, q)
            self.assertEqual(o.side, s)

    def test_initial_buy_and_add_on(self):
        prices = {'AAPL': 100.0, 'MSFT': 100.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
                self.day = 0
            def step(self, dt, data):
                self.day += 1
                return {'AAPL': 1.0, 'MSFT': 0.1} if self.day == 1 else {'AAPL': 1.0, 'MSFT': 0.25}
        strat = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strat

        exec_base.step(0)  # Day1 buy 1000,100
        orders2 = exec_base.step(0)
        expected2 = [('MSFT', 150.0, 'buy')]
        self.assertEqual(len(orders2), 1)
        self.assertEqual(orders2[0].ticker, 'MSFT')
        self.assertEqual(orders2[0].quantity, 150.0)
        self.assertEqual(orders2[0].side, 'buy')

    def test_initial_buy_and_sell(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL', 'MSFT'])
                self.day = 0
            def step(self, dt, data):
                self.day += 1
                return {'AAPL': 1.0, 'MSFT': 0.5} if self.day == 1 else {'AAPL': 1.0, 'MSFT': 0.0}
        strat = Strat()
        exec_base = make_exec_base(['AAPL', 'MSFT'], loader, broker)
        exec_base.strategy = strat

        exec_base.step(0)  # Day1 buy 1000,250
        orders2 = exec_base.step(0)
        expected2 = [('MSFT', 250.0, 'sell')]
        self.assertEqual(len(orders2), 1)
        self.assertEqual(orders2[0].ticker, 'MSFT')
        self.assertEqual(orders2[0].quantity, 250.0)
        self.assertEqual(orders2[0].side, 'sell')


if __name__ == '__main__':
    unittest.main() 