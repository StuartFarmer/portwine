import numpy as np
from collections import deque
from portwine.strategies.base import StrategyBase

class OLMARStrategy(StrategyBase):
    """
    Implements the On-Line Moving Average Reversion (OLMAR) strategy as described by:
    Li, Bin, and Steven Hoi.
    "On-line portfolio selection with moving average reversion."
    Proceedings of the 29th International Conference on Machine Learning (ICML), 2012.

    Updated so that once at least 10 tickers have w-day history, the OLMAR
    update runs on that subset.  New tickers only enter once they accumulate
    enough history; until then their weights remain zero.
    """

    def __init__(self, tickers, eps=10.0, window=5):
        super().__init__(tickers)
        self.eps = eps
        self.window = window

        # b_t: current weights (len = number of tickers)
        self.b_t = np.ones(len(tickers)) / len(tickers)

        # rolling window of last `window` closes for each ticker
        self.recent_closes = {tkr: deque(maxlen=self.window) for tkr in tickers}

        # store yesterday's close to compute price relatives
        self.prev_close = {}

    def step(self, current_date, daily_data):
        # 1) build price_relatives and track which tickers have data today
        price_relatives = []
        valid_today = []
        for i, tkr in enumerate(self.tickers):
            info = daily_data.get(tkr)
            if info is None or info.get("close") is None:
                price_relatives.append(0.0)
                continue
            close_price = info["close"]
            self.recent_closes[tkr].append(close_price)

            # compute price-relative
            prev = self.prev_close.get(tkr)
            if prev and prev > 0:
                price_relatives.append(close_price / prev)
                valid_today.append(i)
            else:
                price_relatives.append(1.0)
                valid_today.append(i)

        # update prev_close for next bar
        for tkr in self.tickers:
            info = daily_data.get(tkr)
            if info and info.get("close") is not None:
                self.prev_close[tkr] = info["close"]

        price_relatives = np.array(price_relatives, dtype=float)

        # 2) determine which tickers have at least `window` days history
        valid_idxs = [
            i for i, tkr in enumerate(self.tickers)
            if len(self.recent_closes[tkr]) >= self.window
        ]

        # 3) if ≥10 valid tickers, run OLMAR update on that subset
        if len(valid_idxs) >= 10:
            # build sub-vectors
            b_sub = self.b_t[valid_idxs]
            x_pred_sub = []
            for i in valid_idxs:
                closes = list(self.recent_closes[self.tickers[i]])
                avg_p = np.mean(closes)
                today_p = closes[-1]
                x_pred_sub.append(avg_p / today_p if today_p > 0 else 1.0)
            x_pred_sub = np.array(x_pred_sub, dtype=float)

            # update subset weights
            b_sub_new = self._olmar_update(b_sub, x_pred_sub)

            # scatter back into full-length vector
            b_temp = np.zeros_like(self.b_t)
            for w_new, i in zip(b_sub_new, valid_idxs):
                b_temp[i] = w_new

            # zero out tickers missing today, then normalize
            self.b_t = self._filter_missing_and_normalize(b_temp, price_relatives)

        else:
            # fallback: equal-weight among whatever tickers traded today
            idxs = [i for i, x in enumerate(price_relatives) if x != 0.0]
            if idxs:
                w = 1.0 / len(idxs)
                b_temp = np.zeros_like(self.b_t)
                for i in idxs:
                    b_temp[i] = w
                self.b_t = b_temp
            # otherwise keep last self.b_t

        # 4) return weight dict
        return {tkr: float(self.b_t[i]) for i, tkr in enumerate(self.tickers)}

    def _olmar_update(self, b_t, x_pred):
        x_mean = x_pred.mean()
        gap = self.eps - b_t.dot(x_pred)
        diff = x_pred - x_mean
        denom = diff.dot(diff)
        if denom < 1e-12:
            return b_t
        lam = max(0.0, gap / denom)
        b_temp = b_t + lam * diff
        return self._simplex_proj(b_temp)

    def _filter_missing_and_normalize(self, weights, price_relatives):
        w = weights.copy()
        # zero-out tickers with no data today
        for i, rel in enumerate(price_relatives):
            if rel == 0.0:
                w[i] = 0.0
        s = w.sum()
        if s > 1e-12:
            w /= s
        return w

    def _simplex_proj(self, v):
        if v.size == 0:
            return v
        u = np.sort(v)[::-1]
        sv = 0.0
        rho = 0
        for j in range(len(u)):
            sv += u[j]
            if u[j] - (sv - 1.0) / (j + 1) > 0:
                rho = j + 1
        theta = max(0.0, (u[:rho].sum() - 1.0) / rho)
        w = np.maximum(v - theta, 0.0)
        if w.sum() > 0:
            w /= w.sum()
        return w
