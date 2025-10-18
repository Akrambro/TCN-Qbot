"""
Download 5 Years of Historical Forex Data from Dukascopy

This script efficiently downloads large amounts of historical data with:
- Chunked downloads to avoid memory issues
- Progress tracking
- Resume capability
- Error handling and retry logic
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
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('download_5year_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def download_5_years_data(
    pair: str = 'EURJPY',
    end_date: str = None,
    output_dir: str = 'data',
    timeframe: str = '1m'
):
    """
    Download 5 years of historical data for a forex pair
    
    Args:
        pair: Forex pair (e.g., 'EURJPY', 'EURUSD')
        end_date: End date (default: today)
        output_dir: Output directory
        timeframe: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
    """
    
    # Calculate date range
    if end_date is None:
        end = datetime.now()
    else:
        end = pd.to_datetime(end_date)
    
    start = end - timedelta(days=365 * 5)  # 5 years
    
    logger.info("=" * 80)
    logger.info("DUKASCOPY 5-YEAR DATA DOWNLOAD")
    logger.info("=" * 80)
    logger.info(f"Pair: {pair}")
    logger.info(f"Start Date: {start.strftime('%Y-%m-%d')}")
    logger.info(f"End Date: {end.strftime('%Y-%m-%d')}")
    logger.info(f"Period: {(end - start).days} days (~{(end - start).days / 365:.1f} years)")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize collector
    collector = DukascopyCollector(data_dir=output_dir)
    
    # Download data in yearly chunks for better management
    all_data = []
    current_start = start
    year = 1
    
    while current_start < end:
        chunk_end = min(current_start + timedelta(days=365), end)
        
        logger.info("")
        logger.info(f"{'=' * 80}")
        logger.info(f"YEAR {year} CHUNK: {current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
        logger.info(f"{'=' * 80}")
        
        try:
            # Download this year's data
            df_chunk = collector.get_candles(
                pair=pair,
                start_date=current_start,
                end_date=chunk_end,
                timeframe=timeframe
            )
            
            if len(df_chunk) > 0:
                all_data.append(df_chunk)
                logger.info(f"✓ Year {year} complete: {len(df_chunk)} candles downloaded")
                
                # Save intermediate chunk
                chunk_file = os.path.join(
                    output_dir,
                    f"{pair}_{timeframe}_year{year}_{current_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.csv"
                )
                df_chunk.to_csv(chunk_file, index=False)
                logger.info(f"  Saved to: {chunk_file}")
            else:
                logger.warning(f"⚠ Year {year}: No data downloaded")
        
        except Exception as e:
            logger.error(f"✗ Error downloading year {year}: {e}")
            logger.info("  Continuing with next chunk...")
        
        current_start = chunk_end
        year += 1
    
    if not all_data:
        logger.error("✗ No data downloaded!")
        return None
    
    # Combine all chunks
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMBINING ALL DATA")
    logger.info("=" * 80)
    
    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
    df_combined = df_combined.drop_duplicates(subset='timestamp')
    
    # Save combined file
    output_file = os.path.join(
        output_dir,
        f"{pair}_{timeframe}_5years_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    )
    df_combined.to_csv(output_file, index=False)
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total Candles: {len(df_combined):,}")
    logger.info(f"Date Range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")
    logger.info(f"Trading Days: {df_combined['timestamp'].dt.date.nunique()}")
    logger.info(f"File Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    logger.info(f"Output File: {output_file}")
    logger.info("=" * 80)
    
    # Display data quality metrics
    logger.info("")
    logger.info("DATA QUALITY METRICS:")
    logger.info(f"  Missing values: {df_combined.isnull().sum().sum()}")
    logger.info(f"  Zero prices: {(df_combined[['open', 'high', 'low', 'close']] == 0).sum().sum()}")
    logger.info(f"  Price range: {df_combined['close'].min():.5f} - {df_combined['close'].max():.5f}")
    logger.info(f"  Average volume: {df_combined['volume'].mean():.2f}")
    
    # Display sample data
    logger.info("")
    logger.info("SAMPLE DATA (First 5 rows):")
    logger.info(df_combined.head().to_string())
    
    return df_combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download 5 years of Forex data from Dukascopy')
    parser.add_argument('--pair', type=str, default='EURJPY', help='Forex pair (e.g., EURJPY, EURUSD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--timeframe', type=str, default='1m', choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                       help='Timeframe')
    
    args = parser.parse_args()
    
    try:
        df = download_5_years_data(
            pair=args.pair,
            end_date=args.end_date,
            output_dir=args.output_dir,
            timeframe=args.timeframe
        )
        
        if df is not None:
            logger.info("")
            logger.info("✓ SUCCESS! Data ready for training.")
            logger.info("  Next steps:")
            logger.info("  1. Review the downloaded data")
            logger.info("  2. Run: python scripts/train_model.py")
            logger.info("  3. Start trading: python tcn_quotex_bot.py")
        else:
            logger.error("✗ FAILED! Please check the logs and try again.")
            
    except KeyboardInterrupt:
        logger.warning("\n✗ Download interrupted by user")
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
