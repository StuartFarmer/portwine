import pandas as pd
import numpy as np
from unittest.mock import Mock

# Copy the MockRestrictedDataInterface and CustomIntradayLoader from the test file
class MockRestrictedDataInterface:
    def __init__(self, mock_data=None):
        # Create a mock data loader
        self.data_loader = Mock()
        self.mock_data = mock_data or {}
        self.set_timestamp_calls = []
        self.get_calls = []
        self.current_timestamp = None
        
        # Configure the mock data loader to return proper data
        def mock_next(tickers, timestamp):
            result = {}
            for ticker in tickers:
                if ticker in self.mock_data:
                    data = self.mock_data[ticker]
                    if self.current_timestamp is not None:
                        dt_python = pd.Timestamp(self.current_timestamp)
                        # Use the exact timestamps that match the data
                        dates = pd.to_datetime([
                            '2025-04-14 09:30', '2025-04-14 16:00',
                            '2025-04-15 09:30', '2025-04-15 16:00',
                        ])
                        try:
                            idx = dates.get_loc(dt_python)
                            result[ticker] = {
                                'close': float(data['close'][idx]),
                                'open': float(data['open'][idx]),
                                'high': float(data['high'][idx]),
                                'low': float(data['low'][idx]),
                                'volume': float(data['volume'][idx])
                            }
                        except (KeyError, IndexError):
                            # Fallback to first value if index not found
                            result[ticker] = {
                                'close': float(data['close'][0]),
                                'open': float(data['open'][0]),
                                'high': float(data['high'][0]),
                                'low': float(data['low'][0]),
                                'volume': float(data['volume'][0])
                            }
                    else:
                        # Fallback to first value if no timestamp set
                        result[ticker] = {
                            'close': float(data['close'][0]),
                            'open': float(data['open'][0]),
                            'high': float(data['high'][0]),
                            'low': float(data['low'][0]),
                            'volume': float(data['volume'][0])
                        }
                else:
                    result[ticker] = {
                        'close': 100.0,
                        'open': 100.0,
                        'high': 105.0,
                        'low': 95.0,
                        'volume': 1000000
                    }
            return result
        
        self.data_loader.next = mock_next

    def set_current_timestamp(self, dt):
        self.set_timestamp_calls.append(dt)
        self.current_timestamp = dt

    def set_restricted_tickers(self, tickers):
        self.restricted_tickers = tickers

    def __getitem__(self, ticker):
        self.get_calls.append(ticker)
        if ticker in self.mock_data:
            data = self.mock_data[ticker]
            if self.current_timestamp is not None:
                dt_python = pd.Timestamp(self.current_timestamp)
                # Use the exact timestamps that match the data
                dates = pd.to_datetime([
                    '2025-04-14 09:30', '2025-04-14 16:00',
                    '2025-04-15 09:30', '2025-04-15 16:00',
                ])
                try:
                    idx = dates.get_loc(dt_python)
                    return {
                        'close': data['close'][idx],
                        'open': data['open'][idx],
                        'high': data['high'][idx],
                        'low': data['low'][idx],
                        'volume': data['volume'][idx]
                    }
                except KeyError:
                    print(f"KeyError: {dt_python} not found in dates")
                    return None
            return data
        return None

    def exists(self, ticker, start_date, end_date):
        return ticker in self.mock_data

class CustomIntradayLoader:
    """
    Synthetic intraday OHLCV for ticker 'TEST' with varying close prices:
      2025‑04‑14 09:30 -> 100
      2025‑04‑14 16:00 -> 104
      2025‑04‑15 09:30 -> 108
      2025‑04‑15 16:00 -> 102
    """
    def __init__(self):
        dates = pd.to_datetime([
            '2025-04-14 09:30', '2025-04-14 16:00',
            '2025-04-15 09:30', '2025-04-15 16:00',
        ])
        self.df = pd.DataFrame({
            'open':   [100,   104,   108,   102],
            'high':   [101,   105,   109,   103],
            'low':    [ 99,   103,   107,   101],
            'close':  [100,   104,   108,   102],
            'volume': [1000,  1000,  1000,  1000],
        }, index=dates)

    def fetch_data(self, tickers):
        return {t: self.df for t in tickers}

# Test the data interface
if __name__ == "__main__":
    # Create mock data interface
    data_interface = MockRestrictedDataInterface()
    
    # Convert CustomIntradayLoader data to the new format
    loader = CustomIntradayLoader()
    for ticker in ['TEST']:  # CustomIntradayLoader returns same data for all tickers
        data_interface.mock_data[ticker] = {
            'close': loader.df['close'].values,
            'open': loader.df['open'].values,
            'high': loader.df['high'].values,
            'low': loader.df['low'].values,
            'volume': loader.df['volume'].values
        }
    
    print("Mock data:", data_interface.mock_data)
    
    # Test timestamps
    test_timestamps = [
        '2025-04-14 09:30',
        '2025-04-14 16:00', 
        '2025-04-15 09:30',
        '2025-04-15 16:00'
    ]
    
    for ts_str in test_timestamps:
        data_interface.set_current_timestamp(ts_str)
        result = data_interface['TEST']
        print(f"Timestamp: {ts_str}")
        print(f"Result: {result}")
        print(f"Close price: {result['close'] if result else 'None'}")
        print("---")
    
    # Test the exact data collection that happens in the backtester
    print("\n=== Testing backtester data collection ===")
    all_tickers = ['TEST']
    close_array = np.zeros((len(test_timestamps), len(all_tickers)), dtype=np.float64)
    
    for i, ts_str in enumerate(test_timestamps):
        data_interface.set_current_timestamp(ts_str)
        # Simulate the exact line from add_close_prices
        close_array[i, :] = np.array([
            data_interface[ticker]['close'] if data_interface[ticker] is not None else 0.0 
            for ticker in all_tickers
        ])
        print(f"Index {i}, Timestamp {ts_str}: {close_array[i, :]}")
    
    print(f"Final close array:\n{close_array}")
    
    # Test returns calculation
    ret_array = np.zeros_like(close_array)
    for i in range(1, len(test_timestamps)):  # Skip first day (no previous data)
        for j in range(len(all_tickers)):
            prev_close = close_array[i-1, j]
            curr_close = close_array[i, j]
            
            if prev_close > 0:
                ret_array[i, j] = (curr_close - prev_close) / prev_close
            else:
                ret_array[i, j] = 0.0
    
    print(f"Returns array:\n{ret_array}")
    
    # Expected returns
    expected_returns = {
        '2025-04-14 09:30': 0.0,
        '2025-04-14 16:00': 0.04,  # (104-100)/100
        '2025-04-15 09:30': (108-104)/104,  # ≈ 0.038461538
        '2025-04-15 16:00': (102-108)/108,  # ≈ -0.055555556
    }
    
    print("\nExpected vs Actual returns:")
    for i, ts_str in enumerate(test_timestamps):
        expected = expected_returns[ts_str]
        actual = ret_array[i, 0] if i < len(ret_array) else 0.0
        print(f"{ts_str}: Expected {expected}, Actual {actual}") 