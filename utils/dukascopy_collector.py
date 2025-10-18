"""
Dukascopy Historical Data Collector

Dukascopy provides high-quality tick and candlestick data for Forex pairs.
This collector downloads historical data from Dukascopy's data feed.

Website: https://www.dukascopy.com/swiss/english/marketwatch/historical/
API Docs: https://github.com/Leo4815162342/dukascopy-node
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import struct
import lzma
import os
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DukascopyCollector:
    """
    Download historical Forex data from Dukascopy
    
    Dukascopy stores data in binary format (.bi5 files) organized by:
    - Year/Month/Day/Hour
    - Each hour contains tick data or 1-minute candles
    
    Data structure:
    - Tick data: timestamp, ask, bid, ask_volume, bid_volume
    - Candle data: timestamp, open, high, low, close, volume
    """
    
    BASE_URL = "https://datafeed.dukascopy.com/datafeed"
    
    # Major Forex pairs mapping (Dukascopy format)
    PAIRS = {
        'EURUSD': 'EURUSD',
        'GBPUSD': 'GBPUSD',
        'USDJPY': 'USDJPY',
        'USDCHF': 'USDCHF',
        'AUDUSD': 'AUDUSD',
        'USDCAD': 'USDCAD',
        'NZDUSD': 'NZDUSD',
        'EURJPY': 'EURJPY',
        'EURGBP': 'EURGBP',
        'EURCHF': 'EURCHF',
        'GBPJPY': 'GBPJPY',
        'AUDJPY': 'AUDJPY',
        'EURAUD': 'EURAUD',
        'EURCAD': 'EURCAD',
        'GBPCHF': 'GBPCHF',
        'GBPAUD': 'GBPAUD',
        'GBPCAD': 'GBPCAD',
        'AUDCAD': 'AUDCAD',
        'AUDCHF': 'AUDCHF',
        'NZDJPY': 'NZDJPY',
        'CADJPY': 'CADJPY',
        'CHFJPY': 'CHFJPY',
    }
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize Dukascopy data collector
        
        Args:
            data_dir: Directory to save downloaded data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_candles(
        self,
        pair: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1m'
    ) -> pd.DataFrame:
        """
        Download candlestick data from Dukascopy
        
        Args:
            pair: Currency pair (e.g., 'EURJPY', 'EURUSD')
            start_date: Start date
            end_date: End date
            timeframe: Time frame ('1m' for 1-minute candles)
        
        Returns:
            DataFrame with OHLCV data
        """
        pair = pair.upper().replace('/', '')
        
        if pair not in self.PAIRS:
            raise ValueError(f"Pair {pair} not supported. Available: {list(self.PAIRS.keys())}")
        
        logger.info(f"Downloading {pair} data from {start_date} to {end_date}")
        
        all_candles = []
        current_date = start_date
        
        while current_date < end_date:
            try:
                # Download hourly data
                hourly_data = self._download_hour(pair, current_date)
                if hourly_data is not None and len(hourly_data) > 0:
                    all_candles.append(hourly_data)
                
                # Move to next hour
                current_date += timedelta(hours=1)
                
                # Rate limiting (be nice to Dukascopy servers)
                time.sleep(0.1)
                
                # Progress indicator
                if current_date.hour == 0:
                    logger.info(f"Progress: {current_date.date()}")
                    
            except Exception as e:
                logger.warning(f"Error downloading {current_date}: {e}")
                current_date += timedelta(hours=1)
                continue
        
        if not all_candles:
            raise ValueError("No data downloaded. Check date range and pair symbol.")
        
        # Combine all data
        df = pd.concat(all_candles, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset='timestamp')
        
        # Resample to desired timeframe if needed
        if timeframe != '1m':
            df = self._resample_candles(df, timeframe)
        
        logger.info(f"Downloaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Save to CSV
        filename = f"{pair}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        
        return df
    
    def _download_hour(self, pair: str, timestamp: datetime) -> Optional[pd.DataFrame]:
        """
        Download data for a specific hour
        
        Dukascopy URL format:
        https://datafeed.dukascopy.com/datafeed/{PAIR}/{YEAR}/{MONTH}/{DAY}/{HOUR}h_ticks.bi5
        """
        year = timestamp.year
        # Month is 0-indexed in Dukascopy (0=January, 11=December)
        month = timestamp.month - 1
        day = timestamp.day
        hour = timestamp.hour
        
        # Construct URL for bi5 file
        url = f"{self.BASE_URL}/{pair}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
        
        try:
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 404:
                # No data for this hour (weekends, holidays)
                return None
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            
            # Decompress LZMA data
            decompressed = lzma.decompress(response.content)
            
            # Parse binary data
            candles = self._parse_bi5_ticks(decompressed, timestamp)
            
            return candles
            
        except Exception as e:
            logger.debug(f"Failed to download {url}: {e}")
            return None
    
    def _parse_bi5_ticks(self, data: bytes, hour_timestamp: datetime) -> pd.DataFrame:
        """
        Parse Dukascopy .bi5 tick data format
        
        Each tick record (20 bytes):
        - timestamp (4 bytes, int32): milliseconds since hour start
        - ask (4 bytes, int32): ask price * 100000 (point value)
        - bid (4 bytes, int32): bid price * 100000
        - ask_volume (4 bytes, float32): ask volume
        - bid_volume (4 bytes, float32): bid volume
        """
        if len(data) == 0:
            return pd.DataFrame()
        
        record_size = 20
        num_records = len(data) // record_size
        
        if num_records == 0:
            return pd.DataFrame()
        
        timestamps = []
        asks = []
        bids = []
        
        for i in range(num_records):
            offset = i * record_size
            record = data[offset:offset + record_size]
            
            # Unpack binary data
            # Big-endian format: >i (int32), >f (float32)
            time_ms = struct.unpack('>i', record[0:4])[0]
            ask_point = struct.unpack('>i', record[4:8])[0]
            bid_point = struct.unpack('>i', record[8:12])[0]
            ask_vol = struct.unpack('>f', record[12:16])[0]
            bid_vol = struct.unpack('>f', record[16:20])[0]
            
            # Convert to actual timestamp
            tick_time = hour_timestamp + timedelta(milliseconds=time_ms)
            
            # Convert point values to prices (divide by 100000)
            ask_price = ask_point / 100000.0
            bid_price = bid_point / 100000.0
            
            timestamps.append(tick_time)
            asks.append(ask_price)
            bids.append(bid_price)
        
        # Create DataFrame from ticks
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ask': asks,
            'bid': bids
        })
        
        # Convert ticks to 1-minute candles using mid-price
        df['price'] = (df['ask'] + df['bid']) / 2
        
        # Resample to 1-minute OHLC
        df.set_index('timestamp', inplace=True)
        ohlc = df['price'].resample('1min').ohlc()
        ohlc['volume'] = df['price'].resample('1min').count()  # Tick count as volume
        
        ohlc = ohlc.reset_index()
        ohlc.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Remove NaN rows
        ohlc = ohlc.dropna()
        
        return ohlc
    
    def _resample_candles(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample 1-minute candles to higher timeframes
        
        Args:
            df: DataFrame with 1-minute candles
            timeframe: Target timeframe ('5m', '15m', '1h', etc.)
        
        Returns:
            Resampled DataFrame
        """
        df_copy = df.copy()
        df_copy.set_index('timestamp', inplace=True)
        
        # Map timeframe to pandas offset
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        if timeframe not in freq_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        freq = freq_map[timeframe]
        
        # Resample OHLC
        resampled = df_copy.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        resampled = resampled.dropna()
        resampled = resampled.reset_index()
        
        return resampled
    
    def get_available_pairs(self) -> list:
        """
        Get list of available currency pairs
        
        Returns:
            List of supported pair symbols
        """
        return list(self.PAIRS.keys())
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate downloaded data quality
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_candles': len(df),
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'missing_candles': 0,
            'zero_volume_candles': (df['volume'] == 0).sum(),
            'invalid_ohlc': 0,
            'duplicates': df.duplicated(subset='timestamp').sum()
        }
        
        # Check for invalid OHLC (high < low, etc.)
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        
        results['invalid_ohlc'] = invalid_ohlc
        
        # Check for missing time periods (gaps)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff()
        expected_diff = pd.Timedelta(minutes=1)
        
        # Count gaps larger than expected
        gaps = time_diffs[time_diffs > expected_diff * 2]
        results['missing_candles'] = len(gaps)
        
        return results


def main():
    """
    Example usage of DukascopyCollector
    """
    print("=" * 80)
    print("Dukascopy Historical Data Collector")
    print("=" * 80)
    
    # Initialize collector
    collector = DukascopyCollector(data_dir='data')
    
    # Show available pairs
    print("\nAvailable currency pairs:")
    pairs = collector.get_available_pairs()
    for i, pair in enumerate(pairs, 1):
        print(f"{i:2d}. {pair}", end="  ")
        if i % 5 == 0:
            print()
    print("\n")
    
    # Example: Download EURJPY data
    pair = 'EURJPY'
    
    # Download last 7 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"\nDownloading {pair} data...")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Timeframe: 1-minute candles")
    print("-" * 80)
    
    try:
        df = collector.get_candles(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            timeframe='1m'
        )
        
        print("\n" + "=" * 80)
        print("Data Preview:")
        print("=" * 80)
        print(df.head(10))
        print("\n...")
        print(df.tail(10))
        
        print("\n" + "=" * 80)
        print("Data Summary:")
        print("=" * 80)
        print(f"Total candles: {len(df):,}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\nOHLC Statistics:")
        print(df[['open', 'high', 'low', 'close', 'volume']].describe())
        
        # Validate data
        print("\n" + "=" * 80)
        print("Data Validation:")
        print("=" * 80)
        validation = collector.validate_data(df)
        for key, value in validation.items():
            print(f"{key:20s}: {value}")
        
        print("\n" + "=" * 80)
        print("✅ Data collection completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
