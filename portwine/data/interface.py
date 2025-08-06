from datetime import datetime

class DataInterface:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.current_timestamp = None

    def set_current_timestamp(self, timestamp: datetime):
        self.current_timestamp = timestamp
    
    def __getitem__(self, ticker: str):
        """
        Access data for a ticker using bracket notation: interface['AAPL']
        
        Returns the latest OHLCV data for the ticker at the current timestamp.
        This enables lazy loading and caching without passing large dictionaries to strategies.
        
        Args:
            ticker: The ticker symbol to retrieve data for
            
        Returns:
            dict: OHLCV data dictionary with keys ['open', 'high', 'low', 'close', 'volume']
            
        Raises:
            ValueError: If current_timestamp is not set
            KeyError: If the ticker is not found or has no data
        """
        if self.current_timestamp is None:
            raise ValueError("Current timestamp not set. Call set_current_timestamp() first.")
        
        # Get data for this ticker at the current timestamp
        data = self.data_loader.next([ticker], self.current_timestamp)
        
        if ticker not in data or data[ticker] is None:
            raise KeyError(f"No data found for ticker: {ticker}")
        
        return data[ticker]