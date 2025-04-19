#!/usr/bin/env python
"""
Example script demonstrating how to use the DailyExecutor class.

This script shows how to:
1. Load a configuration file
2. Create a DailyExecutor instance
3. Initialize the executor with components specified in the config
4. Run the executor either once or on a schedule
"""

import argparse
import logging
import sys
import os
from pathlib import Path

from custom_daily_executor import CustomDailyExecutor


def setup_logging(config):
    """Set up logging based on configuration."""
    logging_config = config.get("logging", {})
    level = logging_config.get("level", "INFO")
    log_file = logging_config.get("file")
    log_format = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        level=getattr(logging, level),
        format=log_format,
        handlers=handlers
    )


def main():
    """Run the daily executor with the provided configuration."""
    parser = argparse.ArgumentParser(description="Run a trading strategy using DailyExecutor")
    parser.add_argument(
        "--config", 
        type=str, 
        default="example_config.json", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--run-once", 
        action="store_true", 
        help="Run the strategy once instead of scheduling"
    )
    args = parser.parse_args()

    # Resolve the config path
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found.")
        sys.exit(1)

    try:
        # Create executor from config file
        executor = CustomDailyExecutor.from_config_file(config_path)
        
        # Initialize all components
        executor.initialize()
        
        if args.run_once:
            print("Running strategy once...")
            executor.run_once()
        else:
            print("Starting scheduled execution_complex...")
            executor.run_scheduled()
    
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'executor' in locals():
            executor.shutdown()


if __name__ == "__main__":
    main() 