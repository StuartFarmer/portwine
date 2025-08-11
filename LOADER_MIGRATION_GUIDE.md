# Loader Migration Guide

## Overview

This guide explains how to migrate from the legacy `portwine.loaders` module to the new `portwine.data.providers` system. The legacy loaders have been deprecated and will be removed in a future version.

## What Changed

### Old System (Deprecated)
- **Location**: `portwine.loaders.*`
- **Classes**: `MarketDataLoader`, `AlpacaMarketDataLoader`, `EODHDMarketDataLoader`, etc.
- **Interface**: Custom loader interface with `next()`, `fetch_data()`, etc.

### New System (Recommended)
- **Location**: `portwine.data.providers.*`
- **Classes**: `DataProvider`, `AlpacaProvider`, `EODHDProvider`, etc.
- **Interface**: Standardized provider interface with `get_data()`, `get_data_async()`

## Migration Path

### Option 1: Use Adapter Classes (Recommended for gradual migration)

The adapter classes provide full backward compatibility while using the new provider system internally:

```python
# OLD (deprecated)
from portwine.loaders import AlpacaMarketDataLoader

# NEW (recommended)
from portwine.data.providers.loader_adapters import AlpacaMarketDataLoader

# Usage remains exactly the same
loader = AlpacaMarketDataLoader(api_key="your_key", api_secret="your_secret")
data = loader.next(["AAPL"], timestamp)
```

**Benefits:**
- ✅ No code changes required
- ✅ Uses new provider system internally
- ✅ Shows deprecation warnings to encourage migration
- ✅ Will be automatically updated when providers improve

### Option 2: Direct Provider Usage (Recommended for new code)

For new code, use the providers directly:

```python
from portwine.data.providers import AlpacaProvider
from datetime import datetime, timedelta

# Create provider
provider = AlpacaProvider(api_key="your_key", api_secret="your_secret")

# Get data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
data = provider.get_data("AAPL", start_date, end_date)

# Data format: {datetime: {'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}}
```

**Benefits:**
- ✅ Cleaner, more focused API
- ✅ Better separation of concerns
- ✅ Easier to test and mock
- ✅ More flexible data formats

### Option 3: Use DataStore Interface

For advanced use cases, use the DataStore interface:

```python
from portwine.data.stores.adapter import MarketDataLoaderAdapter
from portwine.data.providers.loader_adapters import AlpacaMarketDataLoader

# Create loader
loader = AlpacaMarketDataLoader(api_key="your_key", api_secret="your_secret")

# Wrap in adapter for DataStore interface
store = MarketDataLoaderAdapter(loader)

# Use DataStore methods
data = store.get("AAPL", timestamp)
all_data = store.get_all("AAPL", start_date, end_date)
```

## Migration Checklist

- [ ] Update imports from `portwine.loaders.*` to `portwine.data.providers.loader_adapters.*`
- [ ] Test that existing code works with the new imports
- [ ] Consider migrating to direct provider usage for new features
- [ ] Update any custom loader implementations to inherit from `ProviderBasedLoader`
- [ ] Remove any direct dependencies on the old loader implementation details

## Deprecation Timeline

- **Current**: Legacy imports show deprecation warnings but continue to work
- **Next Release**: Legacy imports will show stronger warnings
- **Future Release**: Legacy imports will be removed entirely

## Example Migrations

### Simple Loader Usage

```python
# OLD
from portwine.loaders import EODHDMarketDataLoader
loader = EODHDMarketDataLoader(api_key="key")

# NEW (Option 1 - Adapter)
from portwine.data.providers.loader_adapters import EODHDMarketDataLoader
loader = EODHDMarketDataLoader(api_key="key")

# NEW (Option 2 - Direct Provider)
from portwine.data.providers import EODHDProvider
provider = EODHDProvider(api_key="key")
```

### Custom Loader Implementation

```python
# OLD
from portwine.loaders.base import MarketDataLoader

class MyCustomLoader(MarketDataLoader):
    def load_ticker(self, ticker):
        # implementation
        pass

# NEW
from portwine.data.providers.loader_adapters import ProviderBasedLoader

class MyCustomLoader(ProviderBasedLoader):
    def _get_provider(self, ticker):
        # Return appropriate provider or None
        pass
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're using the correct import path:
- ✅ `from portwine.data.providers.loader_adapters import AlpacaMarketDataLoader`
- ❌ `from portwine.loaders import AlpacaMarketDataLoader` (deprecated)

### Deprecation Warnings
Deprecation warnings are expected and indicate that you should migrate. They won't break your code but will become errors in future versions.

### Performance Issues
The adapter classes maintain the same performance characteristics as the old loaders. If you experience performance issues, consider migrating to direct provider usage.

## Support

If you encounter issues during migration:
1. Check that you're using the correct import paths
2. Verify that your API keys and configuration are correct
3. Test with a simple example first
4. Check the provider-specific documentation for any configuration requirements

## Questions?

For questions about the migration or the new provider system, please refer to the main project documentation or create an issue in the project repository.
