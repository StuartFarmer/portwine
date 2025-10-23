from .base import DataProvider
from .polygon import PolygonProvider
from .alpaca import AlpacaProvider
from .eodhd import EODHDProvider
from .fred import FREDProvider

# Loader adapters for backward compatibility
from .loader_adapters import (
    ProviderBasedLoader,
    AlpacaMarketDataLoader,
    EODHDMarketDataLoader,
    PolygonMarketDataLoader,
    FREDMarketDataLoader,
    BrokerDataLoader,
    MarketDataLoader,
)

__all__ = [
    'DataProvider',
    'PolygonProvider',
    'AlpacaProvider',
    'EODHDProvider',
    'FREDProvider',
    # Loader adapters
    'ProviderBasedLoader',
    'AlpacaMarketDataLoader',
    'EODHDMarketDataLoader',
    'PolygonMarketDataLoader',
    'FREDMarketDataLoader',
    'BrokerDataLoader',
    'MarketDataLoader',
]


