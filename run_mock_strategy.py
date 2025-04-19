#!/usr/bin/env python
"""
Example script demonstrating how to use the daily executor with mock execution.

This allows for testing strategies without connecting to actual trading APIs.
"""
import os
import logging
import argparse
import sys

from custom_daily_executor import CustomDailyExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MockStrategy")

def main():
    """Run the daily executor with mock execution"""
    parser = argparse.ArgumentParser(description="Run a trading strategy with mock execution")
    parser.add_argument(
        "--config", 
        type=str, 
        default="mock_config.json", 
        help="Path to mock configuration file"
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
        logger.error(f"Configuration file {config_path} not found.")
        sys.exit(1)

    try:
        # Create executor from config file using the custom executor
        # that properly handles initialization
        logger.info(f"Creating executor with configuration: {config_path}")
        executor = CustomDailyExecutor.from_config_file(config_path)
        
        # Initialize all components
        logger.info("Initializing executor components")
        executor.initialize()
        
        if args.run_once:
            logger.info("Running strategy once...")
            executor.run_once()
        else:
            logger.info("Starting scheduled execution...")
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