"""
Create Sample Data Files for TCN Training
=========================================

Creates sample USDJPY and EURUSD data files in the formats expected by the user.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_realistic_forex_data(pair_name, n_candles, start_price, volatility=0.0001):
    """
    Create realistic forex data with patterns
    """
    print(f"Creating {n_candles} candles for {pair_name}...")
    
    np.random.seed(hash(pair_name) % 2**32)
    
    # Generate timestamps (1-minute candles)
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=i) for i in range(n_candles)]
    
    # Generate price with realistic patterns
    prices = []
    price = start_price
    
    # Add multiple trend cycles
    trend_length = 2000  # Trend changes every 2000 candles
    n_trends = n_candles // trend_length
    
    for trend_idx in range(n_trends + 1):
        # Random trend direction and strength
        trend_dir = np.random.choice([-1, 0, 1], p=[0.3, 0.3, 0.4])
        trend_strength = np.random.uniform(0, 0.00003)
        
        # Volatility regime
        vol_multiplier = np.random.uniform(0.5, 2.0)
        
        for i in range(min(trend_length, n_candles - len(prices))):
            # Random walk with drift (trend)
            noise = np.random.normal(0, volatility * vol_multiplier)
            drift = trend_dir * trend_strength
            
            # Mean reversion
            if len(prices) > 100:
                ma = np.mean(prices[-100:])
                mean_reversion = (ma - price) * 0.0005
            else:
                mean_reversion = 0
            
            # Update price
            price = price * (1 + drift + noise + mean_reversion)
            prices.append(price)
    
    prices = np.array(prices[:n_candles])
    
    # Create OHLC
    df = pd.DataFrame()
    df['timestamp'] = timestamps
    df['close'] = prices
    
    # Open is previous close (with small gap)
    df['open'] = df['close'].shift(1).fillna(df['close'])
    df['open'] = df['open'] * (1 + np.random.normal(0, volatility * 0.1, len(df)))
    
    # High and Low
    volatility_series = df['close'].pct_change().rolling(50).std().fillna(volatility)
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, volatility_series * 0.5, len(df))))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, volatility_series * 0.5, len(df))))
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    # Volume (random but realistic)
    df['volume'] = np.random.randint(100, 10000, n_candles)
    
    # Add some volume spikes during high volatility
    vol_spikes = np.abs(df['close'].pct_change()) > volatility * 3
    df.loc[vol_spikes, 'volume'] *= np.random.uniform(2, 5, vol_spikes.sum())
    df['volume'] = df['volume'].astype(int)
    
    return df

def main():
    print("="*80)
    print("Creating Sample Data Files")
    print("="*80)
    
    # Create data directory if needed
    import os
    os.makedirs('data', exist_ok=True)
    
    # Create USDJPY data (100K candles as mentioned by user)
    print("\n1. Creating USDJPY data...")
    usdjpy = create_realistic_forex_data(
        pair_name='USDJPY',
        n_candles=100000,
        start_price=150.00,  # Realistic USDJPY price
        volatility=0.00008
    )
    
    # Save as .esv (which is essentially CSV)
    usdjpy_path = 'data/usdjpy_100k.esv'
    usdjpy.to_csv(usdjpy_path, index=False)
    print(f"   Saved: {usdjpy_path}")
    print(f"   Candles: {len(usdjpy)}")
    print(f"   Date range: {usdjpy['timestamp'].min()} to {usdjpy['timestamp'].max()}")
    print(f"   Price range: {usdjpy['close'].min():.2f} to {usdjpy['close'].max():.2f}")
    
    # Create EURUSD data (larger dataset)
    print("\n2. Creating EURUSD data...")
    eurusd = create_realistic_forex_data(
        pair_name='EURUSD',
        n_candles=500000,  # Bigger dataset as mentioned
        start_price=1.0800,  # Realistic EURUSD price
        volatility=0.00012
    )
    
    eurusd_path = 'data/eurusd.csv'
    eurusd.to_csv(eurusd_path, index=False)
    print(f"   Saved: {eurusd_path}")
    print(f"   Candles: {len(eurusd)}")
    print(f"   Date range: {eurusd['timestamp'].min()} to {eurusd['timestamp'].max()}")
    print(f"   Price range: {eurusd['close'].min():.5f} to {eurusd['close'].max():.5f}")
    
    print("\n" + "="*80)
    print("Data files created successfully!")
    print("="*80)
    print("\nFiles created:")
    print(f"  - {usdjpy_path} (100,000 candles)")
    print(f"  - {eurusd_path} (500,000 candles)")
    print("\nYou can now train the model using:")
    print("  python advanced_train_model.py --data data/usdjpy_100k.esv")
    print("  or")
    print("  python advanced_train_model.py --data data/eurusd.csv")

if __name__ == '__main__':
    main()
