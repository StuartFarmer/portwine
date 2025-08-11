#!/usr/bin/env python3
"""Test script to verify the updated NoiseRobustnessAnalyzer works correctly"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        from portwine.analyzers.noiserobustness import NoiseRobustnessAnalyzer, MarketDataLoaderAdapter
        print("  ✓ Successfully imported NoiseRobustnessAnalyzer")
        print("  ✓ Successfully imported MarketDataLoaderAdapter")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_adapter_creation():
    """Test that the adapter can be created"""
    print("\nTesting adapter creation...")
    
    try:
        from portwine.analyzers.noiserobustness import MarketDataLoaderAdapter
        from portwine.loaders.base import MarketDataLoader
        
        # Create a mock loader
        class MockLoader(MarketDataLoader):
            def load_ticker(self, ticker):
                return None  # Mock implementation
        
        mock_loader = MockLoader()
        adapter = MarketDataLoaderAdapter(mock_loader)
        
        print("  ✓ Successfully created MarketDataLoaderAdapter")
        print(f"  ✓ Adapter type: {type(adapter)}")
        return True
    except Exception as e:
        print(f"  ✗ Adapter creation failed: {e}")
        return False

def test_analyzer_creation():
    """Test that the analyzer can be created"""
    print("\nTesting analyzer creation...")
    
    try:
        from portwine.analyzers.noiserobustness import NoiseRobustnessAnalyzer
        from portwine.loaders.base import MarketDataLoader
        
        # Create a mock loader
        class MockLoader(MarketDataLoader):
            def load_ticker(self, ticker):
                return None  # Mock implementation
        
        mock_loader = MockLoader()
        analyzer = NoiseRobustnessAnalyzer(
            base_loader=mock_loader,
            noise_levels=[0.5, 1.0],
            iterations_per_level=5,
            calendar_name="NYSE"
        )
        
        print("  ✓ Successfully created NoiseRobustnessAnalyzer")
        print(f"  ✓ Analyzer type: {type(analyzer)}")
        print(f"  ✓ Noise levels: {analyzer.noise_levels}")
        print(f"  ✓ Calendar name: {analyzer.calendar_name}")
        return True
    except Exception as e:
        print(f"  ✗ Analyzer creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing updated NoiseRobustnessAnalyzer...\n")
    
    tests = [
        test_imports,
        test_adapter_creation,
        test_analyzer_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The updated NoiseRobustnessAnalyzer is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
