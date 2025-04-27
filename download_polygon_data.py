#!/usr/bin/env python3
"""
Script to download historical market data from Polygon.io and save it to a directory.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from portwine.loaders.polygon import PolygonMarketDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download historical market data from Polygon.io')
    parser.add_argument('--tickers', nargs='+', required=True, help='List of ticker symbols to download')
    parser.add_argument('--output-dir', required=True, help='Directory to save the data')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD). If not provided, will fetch full history.')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD). Defaults to today.')
    parser.add_argument('--timespan', default='day', choices=['minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'],
                      help='Timespan for data aggregation')
    parser.add_argument('--multiplier', type=int, default=1, help='Multiplier for timespan (e.g., 1 day, 5 minute)')
    parser.add_argument('--api-key', help='Polygon.io API key. If not provided, uses POLYGON_API_KEY environment variable')
    
    return parser.parse_args()

def download_data(tickers: List[str], output_dir: str, start_date: str = None, end_date: str = None,
                 timespan: str = 'day', multiplier: int = 1, api_key: str = None):
    """
    Download historical data for the specified tickers and save to the output directory.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to download
    output_dir : str
        Directory to save the data
    start_date : str, optional
        Start date in YYYY-MM-DD format. If not provided, will fetch full history.
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to today.
    timespan : str, default 'day'
        Timespan for data aggregation
    multiplier : int, default 1
        Multiplier for timespan
    api_key : str, optional
        Polygon.io API key
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize the loader
    loader = PolygonMarketDataLoader(
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        cache_dir=str(output_path),
        timespan=timespan,
        multiplier=multiplier
    )
    
    # Download data for each ticker
    for ticker in tickers:
        logger.info(f"Downloading data for {ticker}...")
        try:
            df = loader.load_ticker(ticker)
            if df is not None and not df.empty:
                # Save to parquet file
                output_file = output_path / f"{ticker}.parquet"
                df.to_parquet(output_file)
                logger.info(f"Saved {len(df)} records for {ticker} to {output_file}")
                logger.info(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
            else:
                logger.warning(f"No data found for {ticker}")
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {e}")

def main():
    """Main entry point."""
    args = parse_args()
    
    download_data(
        tickers=args.tickers,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        timespan=args.timespan,
        multiplier=args.multiplier,
        api_key=args.api_key
    )

if __name__ == '__main__':
    main() 