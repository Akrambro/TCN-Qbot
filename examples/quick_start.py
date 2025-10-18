"""
Quick Start Example - Train TCN Model on Sample Data

This script demonstrates the complete workflow:
1. Generate sample data
2. Preprocess features
3. Train TCN model
4. Evaluate performance
5. Run backtest
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'utils')

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from tcn_model import TCNForex, TCNTrainer
from data_preprocessing import ForexDataPreprocessor
from backtesting import BacktestEngine
from risk_management import RiskManager
from logging_utils import TradingLogger


def main():
    print("="*60)
    print("TCN FOREX BOT - QUICK START EXAMPLE")
    print("="*60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    
    # Initialize logger
    logger = TradingLogger(log_dir='logs')
    logger.log_startup({
        'mode': 'training',
        'data': 'sample',
        'samples': 5000
    })
    
    # =========================================================================
    # STEP 1: Generate Sample Data
    # =========================================================================
    print("\n" + "-"*60)
    print("STEP 1: Generating sample data...")
    print("-"*60)
    
    np.random.seed(42)
    n_candles = 5000
    dates = pd.date_range('2024-01-01', periods=n_candles, freq='1min')
    
    # Generate realistic price movements
    returns = np.random.randn(n_candles) * 0.0001
    base_price = 1.1000
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n_candles) * 0.0001)),
        'low': prices * (1 - np.abs(np.random.randn(n_candles) * 0.0001)),
        'close': prices * (1 + np.random.randn(n_candles) * 0.00005),
        'volume': np.random.randint(100, 1000, n_candles)
    })
    
    # Ensure high >= open/close and low <= open/close
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    df = df.set_index('timestamp')
    
    print(f"✅ Generated {len(df)} candles")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    
    # =========================================================================
    # STEP 2: Preprocess Data
    # =========================================================================
    print("\n" + "-"*60)
    print("STEP 2: Preprocessing data...")
    print("-"*60)
    
    preprocessor = ForexDataPreprocessor(
        use_log_returns=True,
        scaler_type='standard',
        add_technical_indicators=True
    )
    
    X, y, feature_columns = preprocessor.full_pipeline(
        df,
        sequence_length=60,
        prediction_horizon=1,
        fit=True
    )
    
    print(f"✅ Preprocessing complete")
    print(f"   Features: {feature_columns[:5]}... ({len(feature_columns)} total)")
    
    # =========================================================================
    # STEP 3: Split Data
    # =========================================================================
    print("\n" + "-"*60)
    print("STEP 3: Splitting data...")
    print("-"*60)
    
    # 60% train, 20% validation, 20% test (temporal split)
    n_samples = len(X)
    train_end = int(0.6 * n_samples)
    val_end = int(0.8 * n_samples)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"✅ Data split complete")
    print(f"   Training: {len(X_train)} samples ({len(X_train)/n_samples:.1%})")
    print(f"   Validation: {len(X_val)} samples ({len(X_val)/n_samples:.1%})")
    print(f"   Test: {len(X_test)} samples ({len(X_test)/n_samples:.1%})")
    
    # =========================================================================
    # STEP 4: Create Data Loaders
    # =========================================================================
    print("\n" + "-"*60)
    print("STEP 4: Creating data loaders...")
    print("-"*60)
    
    batch_size = 32
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Data loaders created (batch size: {batch_size})")
    
    # =========================================================================
    # STEP 5: Build and Train Model
    # =========================================================================
    print("\n" + "-"*60)
    print("STEP 5: Building and training model...")
    print("-"*60)
    
    n_features = X.shape[1]
    model = TCNForex(
        input_channels=n_features,
        num_channels=[16, 16, 8],
        kernel_size=3,
        dropout=0.1
    )
    
    print(f"Model configuration:")
    print(f"  Input channels: {n_features}")
    print(f"  TCN channels: [16, 16, 8]")
    print(f"  Receptive field: {model.get_receptive_field()} time steps")
    
    trainer = TCNTrainer(model)
    
    print(f"\nTraining for 20 epochs...")
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=20,
        learning_rate=0.001,
        early_stopping_patience=5,
        verbose=True
    )
    
    print(f"✅ Training complete!")
    
    # Save model
    model_path = 'models/tcn_quick_start.pt'
    trainer.save_model(model_path)
    print(f"   Model saved to {model_path}")
    
    # =========================================================================
    # STEP 6: Evaluate Model
    # =========================================================================
    print("\n" + "-"*60)
    print("STEP 6: Evaluating model on test set...")
    print("-"*60)
    
    X_test_t = torch.FloatTensor(X_test)
    predictions_proba = trainer.predict(X_test_t)
    predictions = (predictions_proba > 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['DOWN', 'UP']))
    
    # =========================================================================
    # STEP 7: Backtest
    # =========================================================================
    print("\n" + "-"*60)
    print("STEP 7: Running backtest...")
    print("-"*60)
    
    # Get timestamps for test set
    test_timestamps = df.index[val_end + 60:]  # +60 for sequence length
    
    backtester = BacktestEngine(
        initial_balance=1000.0,
        payout_rate=0.80,
        min_confidence=0.60,
        trade_amount=10.0
    )
    
    backtest_results = backtester.run_backtest(
        test_timestamps,
        predictions_proba,
        y_test
    )
    
    # =========================================================================
    # STEP 8: Summary
    # =========================================================================
    print("\n" + "="*60)
    print("QUICK START COMPLETE!")
    print("="*60)
    
    print(f"\nModel Performance:")
    print(f"  Test Accuracy: {accuracy:.2%}")
    print(f"  Win Rate (trading): {backtest_results['win_rate']:.2%}")
    print(f"  Total Return: {backtest_results['total_return_pct']:+.2f}%")
    print(f"  Profit Factor: {backtest_results['profit_factor']:.2f}")
    
    print(f"\nFiles Created:")
    print(f"  📁 {model_path}")
    print(f"  📁 logs/trades_{pd.Timestamp.now().strftime('%Y-%m-%d')}.log")
    
    print(f"\nNext Steps:")
    print(f"  1. Collect real historical data (see data/README.md)")
    print(f"  2. Train with more data using scripts/train_model.py")
    print(f"  3. Adjust hyperparameters in configs/default_config.json")
    print(f"  4. Implement live trading integration")
    
    print("\n" + "="*60)
    
    logger.log_daily_summary({
        'total_trades': backtest_results['total_trades'],
        'wins': backtest_results['wins'],
        'losses': backtest_results['losses'],
        'win_rate': backtest_results['win_rate'],
        'total_pnl': backtest_results['total_pnl'],
        'balance': backtest_results['final_balance']
    })


if __name__ == "__main__":
    main()
