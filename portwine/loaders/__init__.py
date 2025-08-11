"""
Legacy Loaders Module - DEPRECATED

This module is deprecated and will be removed in a future version.
All loader classes have been moved to use the new provider system.

For new code, import directly from portwine.data.providers.loader_adapters
"""

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type checking imports - these won't be imported at runtime
    from portwine.data.providers.loader_adapters import (
        ProviderBasedLoader,
        AlpacaMarketDataLoader,
        EODHDMarketDataLoader,
        PolygonMarketDataLoader,
        FREDMarketDataLoader,
        BrokerDataLoader,
        MarketDataLoader,
    )

# Show deprecation warning when this module is imported
warnings.warn(
    "The portwine.loaders module is deprecated and will be removed in a future version. "
    "Import directly from portwine.data.providers.loader_adapters instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new adapters to maintain backward compatibility
from portwine.data.providers.loader_adapters import (
    ProviderBasedLoader,
    AlpacaMarketDataLoader,
    EODHDMarketDataLoader,
    PolygonMarketDataLoader,
    FREDMarketDataLoader,
    BrokerDataLoader,
    MarketDataLoader,
)

# Legacy aliases for backward compatibility
__all__ = [
    'ProviderBasedLoader',
    'AlpacaMarketDataLoader',
    'EODHDMarketDataLoader', 
    'PolygonMarketDataLoader',
    'FREDMarketDataLoader',
    'BrokerDataLoader',
    'MarketDataLoader',
]
