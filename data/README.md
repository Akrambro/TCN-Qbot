# Data Directory

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

## Quick Start

```python
from utils.data_collection import ForexDataCollector

collector = ForexDataCollector()

# Generate sample data for testing
df = collector.generate_sample_data(
    pair='EURUSD',
    n_candles=10000,
    freq='1min',
    save=True
)

# Or collect from Alpha Vantage
df = collector.collect_from_alpha_vantage(
    symbol='EURUSD',
    api_key='YOUR_API_KEY',
    interval='1min'
)
```
