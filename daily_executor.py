#!/usr/bin/env python
"""
Daily Trading Executor for Portwine

This script provides a command-line interface for the DailyExecutor class
in portwine.utils.daily_executor.

Example usage:
    python daily_executor.py --strategy MomentumStrategy --tickers AAPL,MSFT,GOOG
    python daily_executor.py --config config.json
    python daily_executor.py --paper False  # Live trading
"""

import argparse
import importlib
import json
import logging
import os
import sys
import traceback
from typing import Dict, Any, List

# Add the parent directory to the path if running the script directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the DailyExecutor from utils
from portwine.utils.daily_executor import DailyExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading.log")
    ]
)
logger = logging.getLogger("portwine_executor")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Daily Trading Executor for Portwine")
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    
    # Strategy options
    parser.add_argument('--strategy', type=str, help='Strategy class to use')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers to trade')
    
    # Execution options
    parser.add_argument('--execution_complex', type=str, default='portwine.execution_alpaca.AlpacaExecution',
                      help='Execution class to use')
    parser.add_argument('--data-loader', type=str, 
                      default='portwine.loaders.alpaca.AlpacaMarketDataLoader',
                      help='Data loader class to use')
    parser.add_argument('--paper', type=lambda x: (str(x).lower() == 'true'), default=True,
                      help='Use paper trading (default: True)')
    
    # Scheduling options
    parser.add_argument('--time', type=str, default='15:45',
                      help='Time to run daily (24-hour format, default: 15:45)')
    parser.add_argument('--timezone', type=str, default='America/New_York',
                      help='Timezone for execution_complex time (default: America/New_York)')
    
    # API credentials
    parser.add_argument('--api-key', type=str, help='API key for execution_complex system')
    parser.add_argument('--api-secret', type=str, help='API secret for execution_complex system')
    
    # Run once flag
    parser.add_argument('--run-once', action='store_true',
                      help='Run once immediately instead of scheduling daily')
    
    return parser.parse_args()


def build_config_from_args(args):
    """Build a configuration dictionary from command line arguments."""
    config = {}
    
    # Strategy configuration
    if args.strategy:
        strategy_class = args.strategy
        if '.' not in strategy_class:
            # Add default module path if not specified
            strategy_class = f"portwine.strategies.{strategy_class.lower()}.{strategy_class}"
        
        config["strategy"] = {
            "class": strategy_class,
            "params": {}
        }
        
        if args.tickers:
            config["strategy"]["tickers"] = args.tickers.split(',')
    
    # Execution configuration
    if args.execution:
        config["execution_complex"] = {
            "class": args.execution,
            "params": {
                "paper": args.paper
            }
        }
    
    # Data loader configuration
    if args.data_loader:
        config["data_loader"] = {
            "class": args.data_loader,
            "params": {}
        }
    
    # Scheduling configuration
    config["schedule"] = {
        "run_time": args.time,
        "time_zone": args.timezone,
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    }
    
    # API credentials
    if args.api_key:
        config["api_key"] = args.api_key
    if args.api_secret:
        config["api_secret"] = args.api_secret
    
    return config


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    try:
        if args.config:
            # Use configuration file
            executor = DailyExecutor.from_config_file(args.config)
        else:
            # Build configuration from command line arguments
            config = build_config_from_args(args)
            executor = DailyExecutor(config)
        
        # Initialize components
        executor.initialize()
        
        # Run once or scheduled
        if args.run_once:
            logger.info("Running trading execution_complex once...")
            executor.run_once()
        else:
            logger.info("Starting scheduled trading execution_complex...")
            executor.run_scheduled()
            
    except KeyboardInterrupt:
        logger.info("Execution stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'executor' in locals():
            executor.shutdown()


if __name__ == "__main__":
    main() 