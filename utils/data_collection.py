"""
Data Collection Utilities for Forex Trading

Sources mentioned in AboutBot.pdf:
- HistData.com
- Dukascopy
- Alpha Vantage API
- OANDA API
- MetaTrader export
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from typing import Optional
import os


class ForexDataCollector:
    """
    Collect historical Forex data from various sources
    """
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def collect_from_alpha_vantage(
        self,
        symbol: str = 'EURUSD',
        api_key: str = None,
        interval: str = '1min',
        outputsize: str = 'full'
    ) -> pd.DataFrame:
        """
        Collect data from Alpha Vantage API
        
        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            api_key: Alpha Vantage API key (get free at alphavantage.co)
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            outputsize: 'compact' (100 datapoints) or 'full' (all available)
        
        Returns:
            DataFrame with OHLCV data
        """
        if api_key is None:
            raise ValueError("Alpha Vantage API key required. Get free key at https://www.alphavantage.co/support/#api-key")
        
        print(f"Fetching {symbol} data from Alpha Vantage...")
        
        # Construct API URL
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=FX_INTRADAY&"
            f"from_symbol={symbol[:3]}&"
            f"to_symbol={symbol[3:]}&"
            f"interval={interval}&"
            f"outputsize={outputsize}&"
            f"apikey={api_key}"
        )
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise ValueError(f"API Rate Limit: {data['Note']}")
            
            # Extract time series data
            time_series_key = f'Time Series FX ({interval})'
            if time_series_key not in data:
                raise ValueError(f"No data returned. Check symbol and API key.")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close']
            df = df.astype(float)
            
            # Add volume (not available for FX, use placeholder)
            df['volume'] = 0
            
            print(f"Collected {len(df)} candles from {df.index[0]} to {df.index[-1]}")
            
            # Save to CSV
            filename = f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath)
            print(f"Data saved to {filepath}")
            
            return df
        
        except Exception as e:
            print(f"Error collecting data: {e}")
            return None
    
    def load_from_csv(
        self,
        filepath: str,
        timestamp_col: str = 'timestamp',
        columns: dict = None
    ) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            timestamp_col: Name of timestamp column
            columns: Dictionary mapping file columns to standard names
                    e.g., {'Date': 'timestamp', 'Open': 'open'}
        
        Returns:
            DataFrame with standardized columns
        """
        print(f"Loading data from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Rename columns if mapping provided
        if columns:
            df = df.rename(columns=columns)
        
        # Standardize column names (lowercase)
        df.columns = df.columns.str.lower()
        
        # Convert timestamp
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        print(f"Loaded {len(df)} candles")
        
        return df
    
    def load_from_metatrader(
        self,
        filepath: str
    ) -> pd.DataFrame:
        """
        Load data exported from MetaTrader
        
        MetaTrader export format:
        Date,Time,Open,High,Low,Close,Volume
        """
        print(f"Loading MetaTrader data from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Combine date and time
        if 'Date' in df.columns and 'Time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.drop(['Date', 'Time'], axis=1)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        print(f"Loaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def generate_sample_data(
        self,
        pair: str = 'EURUSD',
        n_candles: int = 10000,
        freq: str = '1min',
        start_price: float = 1.1000,
        volatility: float = 0.0001,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Generate realistic sample Forex data for testing
        
        Args:
            pair: Currency pair name
            n_candles: Number of candles to generate
            freq: Time frequency (1min, 5min, etc.)
            start_price: Starting price
            volatility: Price volatility (standard deviation)
            save: Whether to save to CSV
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Generating {n_candles} candles of sample {pair} data...")
        
        np.random.seed(42)
        
        # Generate timestamps
        dates = pd.date_range(
            start='2024-01-01',
            periods=n_candles,
            freq=freq
        )
        
        # Generate price data with realistic patterns
        # Use geometric Brownian motion
        returns = np.random.randn(n_candles) * volatility
        prices = start_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from prices
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.randn(n_candles) * volatility)),
            'low': prices * (1 - np.abs(np.random.randn(n_candles) * volatility)),
            'close': prices * (1 + np.random.randn(n_candles) * volatility * 0.5),
            'volume': np.random.randint(100, 1000, n_candles)
        })
        
        # Ensure high >= open/close and low <= open/close
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        print(f"Generated data from {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
        
        if save:
            filename = f"{pair}_sample_{freq}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Sample data saved to {filepath}")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate and check quality of Forex data
        
        Returns:
            Dictionary with validation results
        """
        print("\nValidating data quality...")
        
        results = {
            'total_candles': len(df),
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'price_anomalies': 0,
            'valid': True
        }
        
        # Check for price anomalies
        if 'high' in df.columns and 'low' in df.columns:
            anomalies = (df['high'] < df['low']).sum()
            results['price_anomalies'] = anomalies
            if anomalies > 0:
                print(f"⚠️  Found {anomalies} candles where high < low")
                results['valid'] = False
        
        # Check for missing values
        missing_total = sum(results['missing_values'].values())
        if missing_total > 0:
            print(f"⚠️  Found {missing_total} missing values")
            results['valid'] = False
        
        # Check for duplicates
        if results['duplicates'] > 0:
            print(f"⚠️  Found {results['duplicates']} duplicate rows")
            results['valid'] = False
        
        if results['valid']:
            print("✅ Data validation passed")
        
        return results


# README for data directory
DATA_README = """# Data Directory

This directory contains historical Forex data for training and testing the TCN model.

## Data Sources

### 1. HistData.com
- Free tick and 1-minute bar data
- URL: http://www.histdata.com/download-free-forex-data/
- Pairs: EUR/USD, USD/JPY, GBP/USD, etc.
- Format: CSV files organized by year/month

### 2. Dukascopy
- High-quality historical data from Swiss broker
- URL: https://www.dukascopy.com/trading-tools/widgets/quotes/historical_data_feed
- Format: CSV or HST files
- Frequency: Tick to 1-minute bars

### 3. Alpha Vantage API
- Free API with registration
- URL: https://www.alphavantage.co
- Frequency: 1min, 5min, 15min, 30min, 60min
- Limit: 5 API calls per minute (free tier)

### 4. OANDA API
- Practice account provides free data access
- URL: https://developer.oanda.com
- Real-time and historical data
- Requires account registration

### 5. MetaTrader Export
- Export from MT4/MT5 History Center
- Tools > History Center > Export
- Format: CSV with Date, Time, OHLCV

## Data Format

All data should be in CSV format with the following columns:

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.10050,1.10080,1.10040,1.10070,250
2024-01-01 00:01:00,1.10070,1.10090,1.10060,1.10085,300
```

## Minimum Requirements

- **Minimum candles**: 10,000 (preferably 50,000+)
- **Timeframe**: 1-minute or 5-minute
- **Format**: Consistent timestamp format
- **Quality**: No missing values or gaps

## Usage

Use the `ForexDataCollector` class to:
1. Download data from APIs
2. Load data from CSV files
3. Validate data quality
4. Generate sample data for testing
"""


if __name__ == "__main__":
    # Create data README
    with open('data/README.md', 'w') as f:
        f.write(DATA_README)
    print("Created data/README.md")
    
    # Example usage
    collector = ForexDataCollector()
    
    # Generate sample data
    print("\n" + "="*60)
    print("Generating sample data for demonstration")
    print("="*60)
    
    df = collector.generate_sample_data(
        pair='EURUSD',
        n_candles=10000,
        freq='1min',
        save=True
    )
    
    # Validate data
    validation = collector.validate_data(df)
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("To collect real data from Alpha Vantage:")
    print("  1. Get free API key at https://www.alphavantage.co/support/#api-key")
    print("  2. Run: collector.collect_from_alpha_vantage('EURUSD', api_key='YOUR_KEY')")
    print("="*60)
