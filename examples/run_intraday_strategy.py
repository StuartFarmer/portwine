#!/usr/bin/env python3
"""
Example script demonstrating how to use the enhanced DailyExecutor with intraday scheduling.

This script shows:
1. How to load and run with an intraday configuration
2. How to programmatically create a configuration with various schedule types
"""
import os
import argparse
import logging
import sys
from portwine.utils.daily_executor import DailyExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntraDayStrategy")

def create_intraday_config():
    """Create a configuration dictionary for intraday trading."""
    return {
        "strategy": {
            "class": "portwine.strategies.momentum.MomentumStrategy",
            "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
            "params": {
                "lookback_days": 5,
                "top_n": 2,
                "position_size": 0.1
            }
        },
        "execution_complex": {
            "class": "portwine.execution_alpaca.AlpacaExecution",
            "params": {
                "paper": True,
                "api_key": os.environ.get("ALPACA_API_KEY"),
                "api_secret": os.environ.get("ALPACA_API_SECRET")
            }
        },
        "data_loader": {
            "class": "portwine.loaders.alpaca.AlpacaMarketDataLoader",
            "params": {
                "api_key": os.environ.get("ALPACA_API_KEY"),
                "api_secret": os.environ.get("ALPACA_API_SECRET"),
                "timeframe": "15Min",
                "limit": 100
            }
        },
        "schedule": {
            "run_time": "market_open+5m",  # Run 5 minutes after market opens
            "time_zone": "US/Eastern",
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "exchange": "NYSE",
            "market_hours_only": True,
            "intraday": "interval:30m"  # Run every 30 minutes during market hours
        },
        "logging": {
            "level": "INFO",
            "file": "intraday_trading.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }

def main():
    """Run the intraday strategy."""
    parser = argparse.ArgumentParser(description="Run an intraday trading strategy")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file (optional)"
    )
    parser.add_argument(
        "--run-once", 
        action="store_true", 
        help="Run the strategy once instead of scheduling"
    )
    parser.add_argument(
        "--interval", 
        type=str, 
        default="30m",
        help="Intraday interval (e.g., 15m, 30m, 1h) - only used if creating config programmatically"
    )
    args = parser.parse_args()

    try:
        # Get the current directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if args.config:
            # Load from specified config file
            config_path = os.path.join(current_dir, args.config)
            if not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                return
                
            executor = DailyExecutor.from_config_file(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            # Create config programmatically
            config = create_intraday_config()
            
            # Override interval if specified
            if args.interval:
                config["schedule"]["intraday"] = f"interval:{args.interval}"
                
            executor = DailyExecutor(config)
            logger.info(f"Created intraday configuration with interval: {args.interval}")
        
        # Initialize components
        executor.initialize()
        
        if args.run_once:
            logger.info("Running strategy once...")
            executor.run_once()
        else:
            logger.info("Starting scheduled execution_complex...")
            executor.run_scheduled()
            
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'executor' in locals():
            executor.shutdown()

if __name__ == "__main__":
    main() 