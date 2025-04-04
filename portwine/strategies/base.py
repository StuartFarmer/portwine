class StrategyBase:
    """
    Base class for a trading strategy. Subclass this to implement a custom strategy.

    A 'step' method is called each day with that day's data. The method should return
    a dictionary of signals/weights for each ticker on that day.

    A strategy may also declare a separate set of 'alternative_data_tickers'
    it depends on (e.g., indices, macro data, etc.). The backtester can then fetch
    that data for the strategy's use.
    """

    def __init__(self, tickers):
        """
        Parameters
        ----------
        tickers : list
            List of primary ticker symbols that the strategy will manage.
        """
        self.tickers = tickers

    def step(self, current_date, daily_data):
        """
        Called each day with that day's data for each ticker.

        Parameters
        ----------
        current_date : pd.Timestamp
        daily_data : dict
            daily_data[ticker] = {
                'open': ..., 'high': ..., 'low': ...,
                'close': ..., 'volume': ...
            }
            or None if no data for that ticker on this date.

            daily_data[ticker] can also return any arbitrary dictionary or value as well.

            This is good for macroeconomic indices, alternative data, etc.

        Returns
        -------
        signals : dict
            { ticker -> float weight }, where the weights are the fraction
            of capital allocated to each ticker (long/short).
        """
        # Default: equally weight among all *primary* tickers that have data
        valid_tickers = [t for t in self.tickers if daily_data.get(t) is not None]
        n = len(valid_tickers)
        weight = 1.0 / n if n > 0 else 0.0
        signals = {tkr: weight for tkr in valid_tickers}
        return signals

    def rebalance_portfolio(self, current_positions, current_date):
        """
        Optional rebalancing logic can be overridden here.
        By default, returns current_positions unmodified.
        """
        return current_positions