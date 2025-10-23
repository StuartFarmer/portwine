#!/usr/bin/env python3
"""
Example script demonstrating migration from old loaders to new providers.

This script shows three different approaches:
1. Legacy loader usage (deprecated)
2. Adapter-based loader usage (recommended for gradual migration)
3. Direct provider usage (recommended for new code)
"""

import warnings
from datetime import datetime, timedelta

# Suppress deprecation warnings for this example
warnings.filterwarnings("ignore", category=DeprecationWarning)

def example_legacy_loader():
    """Example using legacy loader (deprecated)."""
    print("=== Legacy Loader Usage (Deprecated) ===")
    
    try:
        # This will show deprecation warnings in real usage
        from portwine.loaders import AlpacaMarketDataLoader
        
        # Note: This won't work without actual API keys
        print("Legacy loader imported successfully")
        print("Note: This approach is deprecated and will be removed in future versions")
        
    except ImportError as e:
        print(f"Legacy loader import failed: {e}")
        print("This is expected if the old loaders have been removed")

def example_adapter_loader():
    """Example using adapter-based loader (recommended for gradual migration)."""
    print("\n=== Adapter-Based Loader Usage (Recommended for Migration) ===")
    
    try:
        from portwine.data.providers.loader_adapters import AlpacaMarketDataLoader
        
        # Usage is identical to legacy loader
        print("Adapter-based loader imported successfully")
        print("Usage is identical to legacy loader:")
        print("  loader = AlpacaMarketDataLoader(api_key='key', api_secret='secret')")
        print("  data = loader.next(['AAPL'], timestamp)")
        print("  data = loader.fetch_data(['AAPL'])")
        
    except ImportError as e:
        print(f"Adapter loader import failed: {e}")

def example_direct_provider():
    """Example using direct provider (recommended for new code)."""
    print("\n=== Direct Provider Usage (Recommended for New Code) ===")
    
    try:
        from portwine.data.providers import AlpacaProvider
        
        print("Direct provider imported successfully")
        print("Clean, focused API:")
        print("  provider = AlpacaProvider(api_key='key', api_secret='secret')")
        print("  data = provider.get_data('AAPL', start_date, end_date)")
        
        # Show the data format
        print("\nData format:")
        print("  {datetime: {'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}}")
        
    except ImportError as e:
        print(f"Direct provider import failed: {e}")

def example_data_store_interface():
    """Example using DataStore interface with adapter."""
    print("\n=== DataStore Interface Usage (Advanced) ===")
    
    try:
        from portwine.data.stores.adapter import MarketDataLoaderAdapter
        from portwine.data.providers.loader_adapters import AlpacaMarketDataLoader
        
        print("DataStore interface imported successfully")
        print("Combines loader with DataStore interface:")
        print("  loader = AlpacaMarketDataLoader(api_key='key', api_secret='secret')")
        print("  store = MarketDataLoaderAdapter(loader)")
        print("  data = store.get('AAPL', timestamp)")
        print("  all_data = store.get_all('AAPL', start_date, end_date)")
        
    except ImportError as e:
        print(f"DataStore interface import failed: {e}")

def example_migration_steps():
    """Show the migration steps."""
    print("\n=== Migration Steps ===")
    print("1. Update imports:")
    print("   OLD: from portwine.loaders import AlpacaMarketDataLoader")
    print("   NEW: from portwine.data.providers.loader_adapters import AlpacaMarketDataLoader")
    print()
    print("2. No code changes required - usage remains identical")
    print()
    print("3. For new code, consider using direct providers:")
    print("   from portwine.data.providers import AlpacaProvider")
    print()
    print("4. Test thoroughly after migration")
    print()
    print("5. Remove legacy imports when ready")

def main():
    """Run all examples."""
    print("PortWine Loader Migration Examples")
    print("=" * 50)
    
    example_legacy_loader()
    example_adapter_loader()
    example_direct_provider()
    example_data_store_interface()
    example_migration_steps()
    
    print("\n" + "=" * 50)
    print("Migration completed! Check LOADER_MIGRATION_GUIDE.md for detailed information.")

if __name__ == "__main__":
    main()
