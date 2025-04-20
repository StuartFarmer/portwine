### Test Stock Data

This folder has 4 tickers that can be used for high quality testing and modelling.

AAPL - Starts at 2024-01-01, ends at 2024-12-31. Should be used at the base ticker.
MSFT - Starts at 2023-12-01, ends at 2024-12-31. Can be added to test different start points.
NFLX - Starts at 2024-01-01, ends at 2024-12-31. 3 random rows / days are removed. Can be used to test edge cases with missing data. Missing dates: 2024-02-09, 2024-08-05, 2024-10-30
V - Starts at 2023-12-01, ends at 2024-12-31. 4 random rows / days are removed. Can be used to test edge cases with missing data and different start points. Missing dates: 2023-12-15, 2024-04-25, 2024-07-19, 2024-11-20

Use the EODHDMarketDataLoader for this folder.
