from typing import Optional, List, Dict, Any
import pandas as pd
from portwine.loaders.base import MarketDataLoader
from portwine.brokers.base import BrokerBase


class BrokerDataLoader(MarketDataLoader):
    """
    Alternative data loader providing broker account fields (e.g., 'equity')
    in live mode via BrokerBase and in offline/backtest via evolving initial_equity.
    """
    SOURCE_IDENTIFIER = "BROKER"

    def __init__(self,
                 broker: Optional[BrokerBase] = None,
                 initial_equity: Optional[float] = None):
        super().__init__()
        if broker is None and initial_equity is None:
            raise ValueError("Give either a broker or an initial_equity")
        self.broker = broker
        self.equity = initial_equity  # Only used in backtest/offline mode

    def next(self, tickers: List[str], ts: pd.Timestamp) -> Dict[str, Dict[str, float] | None]:
        """
        Return a dict for each ticker; if prefixed with 'BROKER', return {'equity': value}, else None.
        """
        out: Dict[str, Dict[str, float] | None] = {}
        for t in tickers:
            # Only handle tickers with a prefix; non-colon tickers are not for BROKER
            if ":" not in t:
                out[t] = None
                continue
            src, key = t.split(":", 1)
            if src != self.SOURCE_IDENTIFIER:
                out[t] = None
                continue

            # live vs. offline
            if self.broker is not None:
                account = self.broker.get_account()
                eq = account.equity
            else:
                eq = self.equity

            out[t] = {"equity": float(eq)}
        return out

    def update(self,
               ts: pd.Timestamp,
               raw_sigs: Dict[str, Any],
               raw_rets: Dict[str, float],
               strat_ret: float) -> None:
        """
        Backtest-only hook: evolve self.equity by applying strategy return.
        """
        if self.broker is None and strat_ret is not None:
            self.equity *= (1 + strat_ret) 