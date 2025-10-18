"""
Efficient Multi-Year Data Downloader with Smart Options

This script provides multiple strategies for downloading historical data:
1. Fast demo mode (last 6 months, recommended for testing)
2. Full 5-year download (takes hours, for production)
3. Custom date range
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dukascopy_collector import DukascopyCollector
from datetime import datetime, timedelta
import pandas as pd
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def estimate_download_time(start: datetime, end: datetime, timeframe: str) -> dict:
    """Estimate download time and data size"""
    days = (end - start).days
    
    # Forex market: ~120 hours/week (5 days * 24h)
    weeks = days / 7
    total_hours = weeks * 120
    
    # Dukascopy rate: ~10-20 hours/minute with rate limiting
    estimated_minutes = total_hours / 15
    
    # Candles estimation
    if timeframe == '1m':
        candles_per_hour = 60
    elif timeframe == '5m':
        candles_per_hour = 12
    elif timeframe == '15m':
        candles_per_hour = 4
    elif timeframe == '1h':
        candles_per_hour = 1
    else:
        candles_per_hour = 60  # default
    
    total_candles = total_hours * candles_per_hour
    file_size_mb = (total_candles * 100) / (1024 * 1024)  # ~100 bytes per candle
    
    return {
        'days': days,
        'trading_hours': int(total_hours),
        'estimated_minutes': int(estimated_minutes),
        'estimated_candles': int(total_candles),
        'file_size_mb': file_size_mb
    }


def download_data(
    pair: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str,
    output_dir: str = 'data'
):
    """Download historical data with progress tracking"""
    
    # Estimate download
    stats = estimate_download_time(start_date, end_date, timeframe)
    
    logger.info("=" * 80)
    logger.info("DOWNLOAD CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Pair: {pair}")
    logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Duration: {stats['days']} days")
    logger.info(f"Timeframe: {timeframe}")
    logger.info("")
    logger.info("ESTIMATES:")
    logger.info(f"  Trading hours: ~{stats['trading_hours']:,}")
    logger.info(f"  Expected candles: ~{stats['estimated_candles']:,}")
    logger.info(f"  File size: ~{stats['file_size_mb']:.1f} MB")
    logger.info(f"  Download time: ~{stats['estimated_minutes']} minutes")
    logger.info("=" * 80)
    
    # Confirmation
    response = input("\nProceed with download? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        logger.info("Download cancelled.")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize collector
    collector = DukascopyCollector(data_dir=output_dir)
    
    # Download
    logger.info("\nStarting download...")
    logger.info("(This may take a while. Press Ctrl+C to stop)")
    
    try:
        df = collector.get_candles(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        if df is not None and len(df) > 0:
            # Summary
            logger.info("")
            logger.info("=" * 80)
            logger.info("✓ DOWNLOAD COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"Total candles: {len(df):,}")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"Trading days: {df['timestamp'].dt.date.nunique()}")
            logger.info(f"Output directory: {output_dir}")
            logger.info("=" * 80)
            
            return df
        else:
            logger.error("No data downloaded!")
            return None
            
    except KeyboardInterrupt:
        logger.warning("\n\n✗ Download interrupted by user")
        logger.info("Note: Partial data may have been saved to disk.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Smart Forex Data Downloader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo (6 months, recommended for testing)
  python scripts/download_efficient.py --mode demo --pair EURJPY
  
  # Full 5 years (takes 2-3 hours)
  python scripts/download_efficient.py --mode full --pair EURJPY
  
  # Custom date range
  python scripts/download_efficient.py --mode custom --pair EURUSD --start 2023-01-01 --end 2024-01-01
  
  # Different timeframe (faster download)
  python scripts/download_efficient.py --mode demo --pair GBPUSD --timeframe 5m
        """
    )
    
    parser.add_argument('--mode', type=str, choices=['demo', 'full', 'custom'], required=True,
                       help='Download mode: demo (6 months), full (5 years), custom (specify dates)')
    parser.add_argument('--pair', type=str, default='EURJPY',
                       help='Forex pair (e.g., EURJPY, EURUSD, GBPUSD)')
    parser.add_argument('--timeframe', type=str, default='1m', 
                       choices=['1m', '5m', '15m', '1h', '4h'],
                       help='Timeframe (1m=slowest but most data, 5m=faster, 15m=fast)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD) for custom mode')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD) for custom mode')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    # Calculate date range
    end = datetime.now()
    
    if args.mode == 'demo':
        start = end - timedelta(days=180)  # 6 months
        logger.info("📊 DEMO MODE: Downloading last 6 months (recommended for testing)")
        
    elif args.mode == 'full':
        start = end - timedelta(days=365 * 5)  # 5 years
        logger.info("📊 FULL MODE: Downloading 5 years (this will take 2-3 hours!)")
        
    elif args.mode == 'custom':
        if not args.start or not args.end:
            logger.error("Custom mode requires --start and --end dates")
            return
        start = pd.to_datetime(args.start)
        end = pd.to_datetime(args.end)
        logger.info("📊 CUSTOM MODE: Downloading custom date range")
    
    # Download
    df = download_data(
        pair=args.pair,
        start_date=start,
        end_date=end,
        timeframe=args.timeframe,
        output_dir=args.output_dir
    )
    
    if df is not None:
        logger.info("\n✓ SUCCESS! Data is ready for training.")
        logger.info("\nNext steps:")
        logger.info("  1. Review data: ls -lh data/")
        logger.info("  2. Train model: python scripts/train_model.py")
        logger.info("  3. Run bot: python tcn_quotex_bot.py")


if __name__ == "__main__":
    main()
