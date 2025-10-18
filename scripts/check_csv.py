#!/usr/bin/env python3
"""
Quick utility to validate USDJPY CSV file format before training
"""

import sys
import pandas as pd
from pathlib import Path

def check_csv(filepath):
    """Validate CSV file for training"""
    print("=" * 60)
    print("📋 CSV File Validation")
    print("=" * 60)
    
    # Check file exists
    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        print(f"\n💡 Expected location: data/usdjpy_100k.csv")
        return False
    
    print(f"✅ File found: {filepath}")
    
    # Check file size
    size_mb = Path(filepath).stat().st_size / (1024 * 1024)
    print(f"📊 File size: {size_mb:.2f} MB")
    
    try:
        # Read CSV
        print("\n🔍 Reading CSV file...")
        df = pd.read_csv(filepath)
        
        # Check row count
        print(f"✅ Total rows: {len(df):,}")
        
        if len(df) < 1000:
            print(f"⚠️  Warning: Only {len(df)} rows. Need at least 1000 for training.")
            return False
        
        # Display columns
        print(f"\n📝 Columns found ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        # Check for required columns (flexible naming)
        required = {
            'timestamp': ['timestamp', 'date', 'time', 'datetime', 'Date', 'Time'],
            'open': ['open', 'o', 'Open', 'OPEN'],
            'high': ['high', 'h', 'High', 'HIGH'],
            'low': ['low', 'l', 'Low', 'LOW'],
            'close': ['close', 'c', 'Close', 'CLOSE']
        }
        
        print(f"\n✅ Required columns check:")
        missing = []
        for req, variants in required.items():
            found = any(col in df.columns for col in variants)
            if found:
                matched = [col for col in df.columns if col in variants][0]
                print(f"   ✅ {req.upper()}: found as '{matched}'")
            else:
                print(f"   ❌ {req.upper()}: MISSING (expected one of: {', '.join(variants)})")
                missing.append(req)
        
        if missing:
            print(f"\n❌ Missing required columns: {', '.join(missing)}")
            return False
        
        # Show sample data
        print(f"\n📊 First 5 rows:")
        print(df.head(5).to_string())
        
        print(f"\n📊 Last 5 rows:")
        print(df.tail(5).to_string())
        
        # Check for missing values
        print(f"\n🔍 Missing values check:")
        missing_counts = df.isnull().sum()
        if missing_counts.sum() == 0:
            print("   ✅ No missing values!")
        else:
            print("   ⚠️  Found missing values:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"      - {col}: {count} missing ({count/len(df)*100:.2f}%)")
        
        # Data quality checks
        print(f"\n📈 Data quality:")
        
        # Find time column
        time_col = None
        for col in df.columns:
            if col.lower() in ['timestamp', 'date', 'time', 'datetime']:
                time_col = col
                break
        
        if time_col:
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                print(f"   ✅ Timestamps parseable")
                print(f"   📅 Date range: {df[time_col].min()} to {df[time_col].max()}")
                
                # Check for time gaps
                time_diff = df[time_col].diff()
                mode_diff = time_diff.mode()[0]
                print(f"   ⏱️  Most common interval: {mode_diff}")
            except:
                print(f"   ⚠️  Could not parse timestamps")
        
        # Check price columns
        price_cols = [col for col in df.columns if col.lower() in ['open', 'high', 'low', 'close', 'o', 'h', 'l', 'c']]
        if price_cols:
            for col in price_cols:
                if df[col].dtype in ['float64', 'int64']:
                    print(f"   ✅ {col}: numeric (min={df[col].min():.4f}, max={df[col].max():.4f})")
                else:
                    print(f"   ⚠️  {col}: non-numeric type {df[col].dtype}")
        
        # Training split preview
        print(f"\n📊 Recommended data split (for {len(df):,} rows):")
        train_size = min(60000, int(len(df) * 0.6))
        val_size = min(30000, int(len(df) * 0.3))
        test_size = len(df) - train_size - val_size
        
        print(f"   Training:   {train_size:,} rows ({train_size/len(df)*100:.1f}%)")
        print(f"   Validation: {val_size:,} rows ({val_size/len(df)*100:.1f}%)")
        print(f"   Testing:    {test_size:,} rows ({test_size/len(df)*100:.1f}%)")
        
        print("\n" + "=" * 60)
        print("✅ CSV FILE IS VALID!")
        print("=" * 60)
        
        print(f"\n🚀 Ready to train! Run:")
        print(f"   python train_usdjpy.py --data_path {filepath}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error reading CSV: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_csv.py <path_to_csv>")
        print("\nExample:")
        print("  python scripts/check_csv.py data/usdjpy_100k.csv")
        sys.exit(1)
    
    filepath = sys.argv[1]
    success = check_csv(filepath)
    sys.exit(0 if success else 1)
