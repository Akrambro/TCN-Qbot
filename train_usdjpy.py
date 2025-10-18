"""
Train TCN Model on USDJPY 1-minute data
========================================

This script trains the TCN model on your 100,000 bars of USDJPY data with:
- Training: 60,000 bars (60%)
- Validation: 30,000 bars (30%)
- Testing: 10,000 bars (10%)

Usage:
    python train_usdjpy.py --data_path /path/to/usdjpy.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tcn_model import TCNForex, TCNTrainer
from src.data_preprocessing import ForexDataPreprocessor
from utils.logging_utils import TradingLogger
from utils.backtesting import BacktestEngine


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data and validate format
    Expected columns: timestamp/date, open, high, low, close, volume (optional)
    """
    logger.info(f"Loading data from: {csv_path}")
    
    # Try to read CSV
    df = pd.read_csv(csv_path)
    
    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Detect column names (case-insensitive)
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['time', 'timestamp', 'date', 'datetime']:
            column_mapping[col] = 'timestamp'
        elif col_lower in ['open', 'o']:
            column_mapping[col] = 'open'
        elif col_lower in ['high', 'h']:
            column_mapping[col] = 'high'
        elif col_lower in ['low', 'l']:
            column_mapping[col] = 'low'
        elif col_lower in ['close', 'c']:
            column_mapping[col] = 'close'
        elif col_lower in ['volume', 'vol', 'v']:
            column_mapping[col] = 'volume'
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.info("Please provide CSV with columns: timestamp, open, high, low, close")
        raise ValueError(f"Missing columns: {missing}")
    
    # Add volume if missing
    if 'volume' not in df.columns:
        df['volume'] = 0
        logger.info("Volume column missing - filled with zeros")
    
    # Parse timestamp if exists
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        except:
            logger.warning("Could not parse timestamp column")
    else:
        # Create synthetic timestamp (1-minute intervals)
        logger.info("No timestamp column - creating synthetic timestamps")
        start_date = pd.Timestamp('2024-07-01')
        df['timestamp'] = pd.date_range(start=start_date, periods=len(df), freq='1min')
    
    logger.info(f"\nData Summary:")
    logger.info(f"  Total bars: {len(df)}")
    logger.info(f"  Columns: {df.columns.tolist()}")
    logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"\nPrice Statistics:")
    logger.info(f"  Open range: {df['open'].min():.5f} - {df['open'].max():.5f}")
    logger.info(f"  Close range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    
    return df


def split_data(df: pd.DataFrame, train_size=60000, val_size=30000):
    """
    Split data into train/validation/test sets
    """
    logger.info("\n" + "="*80)
    logger.info("SPLITTING DATA")
    logger.info("="*80)
    
    total_size = len(df)
    test_size = total_size - train_size - val_size
    
    logger.info(f"Total bars: {total_size}")
    logger.info(f"Training: {train_size} bars ({train_size/total_size*100:.1f}%)")
    logger.info(f"Validation: {val_size} bars ({val_size/total_size*100:.1f}%)")
    logger.info(f"Testing: {test_size} bars ({test_size/total_size*100:.1f}%)")
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    test_df = df.iloc[train_size+val_size:].copy()
    
    logger.info(f"\nTrain period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    logger.info(f"Val period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    logger.info(f"Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    return train_df, val_df, test_df


def preprocess_data(train_df, val_df, test_df, sequence_length=50):
    """
    Preprocess data and create sequences
    """
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING DATA")
    logger.info("="*80)
    
    preprocessor = ForexDataPreprocessor()
    
    # Fit on training data
    logger.info("Computing technical indicators...")
    X_train, y_train, feature_columns = preprocessor.full_pipeline(train_df, sequence_length=sequence_length, fit=True)
    
    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Training labels shape: {y_train.shape}")
    logger.info(f"Number of features: {X_train.shape[2]}")
    logger.info(f"Feature names: {feature_columns}")
    
    # Transform validation and test data
    logger.info("\nTransforming validation data...")
    X_val, y_val, _ = preprocessor.full_pipeline(val_df, sequence_length=sequence_length, fit=False)
    logger.info(f"Validation features shape: {X_val.shape}")
    
    logger.info("\nTransforming test data...")
    X_test, y_test, _ = preprocessor.full_pipeline(test_df, sequence_length=sequence_length, fit=False)
    logger.info(f"Test features shape: {X_test.shape}")
    
    # Check class distribution
    logger.info("\nClass Distribution:")
    logger.info(f"  Train - Up: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.2f}%)")
    logger.info(f"  Val   - Up: {y_val.sum()}/{len(y_val)} ({y_val.mean()*100:.2f}%)")
    logger.info(f"  Test  - Up: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.2f}%)")
    
    return preprocessor, (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_model(X_train, y_train, X_val, y_val, n_features, model_path='models/tcn_usdjpy.pt'):
    """
    Train TCN model
    """
    logger.info("\n" + "="*80)
    logger.info("TRAINING TCN MODEL (IMPROVED)")
    logger.info("="*80)
    
    # Create LARGER model with STRONG REGULARIZATION
    model = TCNForex(
        input_channels=n_features,
        num_channels=[128, 128, 64, 64],  # DOUBLED capacity!
        kernel_size=3,
        dropout=0.5  # INCREASED dropout to prevent overfitting!
    )
    
    logger.info(f"Model architecture (IMPROVED + REGULARIZED):")
    logger.info(f"  Input channels: {n_features}")
    logger.info(f"  Hidden channels: [128, 128, 64, 64] ← LARGER!")
    logger.info(f"  Kernel size: 3")
    logger.info(f"  Dropout: 0.5 ← STRONG regularization!")
    logger.info(f"  Receptive field: {model.receptive_field}")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = TCNTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    logger.info(f"Device: {trainer.device}")
    
    # Create DataLoaders with BALANCED SAMPLING
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    from sklearn.metrics import classification_report
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    # Calculate class distribution
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    
    logger.info(f"\nClass balance:")
    logger.info(f"  Positive samples (UP): {n_pos} ({n_pos/len(y_train)*100:.1f}%)")
    logger.info(f"  Negative samples (DOWN): {n_neg} ({n_neg/len(y_train)*100:.1f}%)")
    
    # Create BALANCED SAMPLER - ensures 50-50 UP/DOWN in each batch
    logger.info("Creating balanced sampler...")
    class_sample_count = np.array([n_neg, n_pos])
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    
    sampler = WeightedRandomSampler(
        weights=samples_weight.type(torch.DoubleTensor),
        num_samples=len(samples_weight),
        replacement=True
    )
    
    logger.info("Using BALANCED BATCH SAMPLING - each batch will have ~50% UP, ~50% DOWN")
    
    # Create loaders (train uses sampler, validation doesn't shuffle)
    logger.info("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    logger.info("Data loaders created successfully!")
    
    # Train with FOCAL LOSS and strong regularization
    logger.info("\n" + "="*80)
    logger.info("TRAINING WITH COMPREHENSIVE FIXES:")
    logger.info("="*80)
    logger.info("✓ Focal Loss (alpha=0.25, gamma=2.0) - prevents bias collapse")
    logger.info("✓ Balanced Batch Sampling - ensures 50-50 UP/DOWN per batch")
    logger.info("✓ Advanced Price Action Features - stronger signals")
    logger.info("✓ Higher Dropout (0.5) - reduces overfitting")
    logger.info("✓ Weight Decay (1e-4) - L2 regularization")
    logger.info("✓ LR Scheduler - adaptive learning rate")
    logger.info("="*80 + "\n")
    
    logger.info("Starting trainer.fit()...")
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=100,
        learning_rate=0.0005,
        early_stopping_patience=30,  # More patience
        use_focal_loss=True,  # FOCAL LOSS!
        focal_alpha=0.25,
        focal_gamma=2.0,
        weight_decay=1e-4,  # L2 REGULARIZATION!
        use_lr_scheduler=True  # LR SCHEDULER!
    )
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer.save_model(model_path)
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer.save_model(model_path)
    logger.info(f"\nModel saved to: {model_path}")
    
    return trainer, history


def evaluate_model(trainer, X_test, y_test, test_df):
    """
    Evaluate model on test set with detailed per-class metrics
    """
    logger.info("\n" + "="*80)
    logger.info("EVALUATING MODEL")
    logger.info("="*80)
    
    # Get predictions (convert numpy array to torch tensor)
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    
    X_test_tensor = torch.FloatTensor(X_test)
    predictions = trainer.predict(X_test_tensor)
    pred_probs = predictions  # Probabilities
    pred_labels = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = (pred_labels == y_test).mean()
    up_accuracy = (pred_labels[y_test == 1] == 1).mean() if (y_test == 1).sum() > 0 else 0
    down_accuracy = (pred_labels[y_test == 0] == 0).mean() if (y_test == 0).sum() > 0 else 0
    
    # Prediction distribution
    n_pred_up = (pred_labels == 1).sum()
    n_pred_down = (pred_labels == 0).sum()
    n_actual_up = (y_test == 1).sum()
    n_actual_down = (y_test == 0).sum()
    
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Overall Accuracy: {accuracy*100:.2f}%")
    logger.info(f"  Up Accuracy: {up_accuracy*100:.2f}%")
    logger.info(f"  Down Accuracy: {down_accuracy*100:.2f}%")
    
    logger.info(f"\nPrediction Distribution (BIAS CHECK):")
    logger.info(f"  Predicted UP: {n_pred_up} ({n_pred_up/len(y_test)*100:.1f}%)")
    logger.info(f"  Predicted DOWN: {n_pred_down} ({n_pred_down/len(y_test)*100:.1f}%)")
    logger.info(f"  Actual UP: {n_actual_up} ({n_actual_up/len(y_test)*100:.1f}%)")
    logger.info(f"  Actual DOWN: {n_actual_down} ({n_actual_down/len(y_test)*100:.1f}%)")
    
    # Check for bias
    pred_up_pct = n_pred_up / len(y_test) * 100
    if pred_up_pct < 30 or pred_up_pct > 70:
        logger.warning(f"⚠️  WARNING: Model shows BIAS! Predicting {pred_up_pct:.1f}% UP")
        logger.warning("   A balanced model should predict ~50% UP, ~50% DOWN")
    else:
        logger.info(f"✅ Predictions are BALANCED ({pred_up_pct:.1f}% UP, {100-pred_up_pct:.1f}% DOWN)")
    
    # Detailed classification report
    logger.info("\n" + "="*80)
    logger.info("PER-CLASS METRICS (Precision, Recall, F1-Score):")
    logger.info("="*80)
    report = classification_report(y_test, pred_labels, target_names=['DOWN', 'UP'], digits=4)
    logger.info("\n" + report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, pred_labels)
    logger.info("Confusion Matrix:")
    logger.info(f"              Predicted")
    logger.info(f"              DOWN    UP")
    logger.info(f"Actual DOWN   {cm[0,0]:4d}  {cm[0,1]:4d}")
    logger.info(f"Actual UP     {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Confidence analysis
    high_conf_mask = (pred_probs > 0.6) | (pred_probs < 0.4)
    if high_conf_mask.sum() > 0:
        high_conf_acc = (pred_labels[high_conf_mask] == y_test[high_conf_mask]).mean()
        logger.info(f"\nHigh Confidence Trades (>60% or <40%):")
        logger.info(f"  Count: {high_conf_mask.sum()} / {len(y_test)} ({high_conf_mask.mean()*100:.1f}%)")
        logger.info(f"  Accuracy: {high_conf_acc*100:.2f}%")
    
    return predictions


def run_backtest(trainer, test_df, preprocessor, predictions):
    """
    Run backtest on test data
    """
    logger.info("\n" + "="*80)
    logger.info("RUNNING BACKTEST")
    logger.info("="*80)
    
    # Prepare data for backtesting
    # Need to align predictions with test data (accounting for sequence length)
    sequence_length = preprocessor.sequence_length
    test_df_aligned = test_df.iloc[sequence_length:].reset_index(drop=True)
    
    # Add predictions to dataframe
    test_df_aligned['prediction'] = predictions
    test_df_aligned['signal'] = 0
    
    # Generate signals based on confidence threshold
    conf_threshold = 0.6
    test_df_aligned.loc[test_df_aligned['prediction'] > conf_threshold, 'signal'] = 1  # BUY
    test_df_aligned.loc[test_df_aligned['prediction'] < (1 - conf_threshold), 'signal'] = -1  # SELL
    
    logger.info(f"Confidence threshold: {conf_threshold}")
    logger.info(f"Buy signals: {(test_df_aligned['signal'] == 1).sum()}")
    logger.info(f"Sell signals: {(test_df_aligned['signal'] == -1).sum()}")
    logger.info(f"No trade: {(test_df_aligned['signal'] == 0).sum()}")
    
    # Run backtest
    backtest = BacktestEngine(
        initial_balance=10000,
        trade_amount=100,
        payout_rate=0.80,
        trade_duration_minutes=1
    )
    
    results = backtest.run(test_df_aligned)
    
    # Print results
    logger.info("\nBacktest Results:")
    logger.info(f"  Total Trades: {results['total_trades']}")
    logger.info(f"  Win Rate: {results['win_rate']*100:.2f}%")
    logger.info(f"  Total Profit: ${results['total_profit']:.2f}")
    logger.info(f"  Final Balance: ${results['final_balance']:.2f}")
    logger.info(f"  Return: {results['return_pct']:.2f}%")
    logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    
    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    backtest.save_results(f"{results_dir}/usdjpy_backtest.json")
    backtest.plot_results(f"{results_dir}/usdjpy_backtest.png")
    
    logger.info(f"\nResults saved to {results_dir}/")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train TCN on USDJPY data')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to USDJPY CSV file')
    parser.add_argument('--train_size', type=int, default=60000,
                        help='Number of bars for training (default: 60000)')
    parser.add_argument('--val_size', type=int, default=30000,
                        help='Number of bars for validation (default: 30000)')
    parser.add_argument('--sequence_length', type=int, default=50,
                        help='Sequence length for TCN (default: 50)')
    parser.add_argument('--model_path', type=str, default='models/tcn_usdjpy.pt',
                        help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Setup
    global logger
    trading_logger = TradingLogger(log_dir='logs')
    logger = trading_logger.logger  # Get the underlying logger
    
    logger.info("="*80)
    logger.info("TCN USDJPY TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Train size: {args.train_size}")
    logger.info(f"Validation size: {args.val_size}")
    logger.info(f"Sequence length: {args.sequence_length}")
    logger.info(f"Model path: {args.model_path}")
    
    try:
        # Step 1: Load data
        df = load_and_validate_data(args.data_path)
        
        # Step 2: Split data
        train_df, val_df, test_df = split_data(
            df, 
            train_size=args.train_size,
            val_size=args.val_size
        )
        
        # Step 3: Preprocess
        preprocessor, (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data(
            train_df, val_df, test_df,
            sequence_length=args.sequence_length
        )
        
        # Save preprocessor
        preprocessor_path = args.model_path.replace('.pt', '_preprocessor.pkl')
        import pickle
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Preprocessor saved to: {preprocessor_path}")
        
        # Step 4: Train model
        trainer, history = train_model(
            X_train, y_train, X_val, y_val,
            n_features=X_train.shape[1],  # Number of feature channels after transpose
            model_path=args.model_path
        )
        
        # Step 5: Evaluate
        predictions = evaluate_model(trainer, X_test, y_test, test_df)
        
        # Step 6: Backtest
        results = run_backtest(trainer, test_df, preprocessor, predictions)
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Preprocessor: {preprocessor_path}")
        logger.info(f"Test Accuracy: {results['win_rate']*100:.2f}%")
        logger.info(f"Final Return: {results['return_pct']:.2f}%")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
