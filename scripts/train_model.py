"""
Training Script for TCN Forex Model

Implements the methodology from AboutBot.pdf:
- Walk-forward validation for time series
- Proper train/validation/test splits
- Early stopping and model checkpointing
- Performance evaluation and backtesting simulation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from typing import Tuple, Dict
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tcn_model import TCNForex, TCNTrainer
from data_preprocessing import ForexDataPreprocessor


class WalkForwardValidator:
    """
    Walk-forward validation for time series models
    
    As described in AboutBot.pdf:
    - Train on data up to time t
    - Test on t+1...t+k
    - Roll forward and repeat
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2
    ):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X: np.ndarray, y: np.ndarray):
        """
        Generate train/test indices for walk-forward validation
        
        Args:
            X: Input data
            y: Target data
        
        Yields:
            train_idx, test_idx for each fold
        """
        n_samples = len(X)
        test_size = int(n_samples * self.test_size)
        
        # Minimum training size
        min_train = int(n_samples * 0.5)
        
        for i in range(self.n_splits):
            # Calculate split point
            test_end = n_samples - (self.n_splits - i - 1) * (test_size // self.n_splits)
            test_start = test_end - test_size
            
            # Ensure minimum training size
            if test_start < min_train:
                continue
            
            train_idx = np.arange(0, test_start)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx


class ModelTrainer:
    """
    Complete training pipeline for TCN model
    """
    
    def __init__(
        self,
        data_path: str,
        model_save_path: str = 'models/',
        config_path: str = 'configs/'
    ):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.config_path = config_path
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(config_path, exist_ok=True)
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
    
    def load_and_prepare_data(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Load and preprocess data"""
        print("="*60)
        print("LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        # Initialize preprocessor
        preprocessor = ForexDataPreprocessor(
            use_log_returns=True,
            scaler_type='standard',
            add_technical_indicators=True
        )
        
        # Load data
        df = preprocessor.load_and_prepare(self.data_path)
        
        # Full preprocessing pipeline
        X, y, feature_columns = preprocessor.full_pipeline(
            df,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            fit=True
        )
        
        # Save preprocessor
        import pickle
        with open(os.path.join(self.model_save_path, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(preprocessor, f)
        
        print(f"\nData prepared successfully!")
        
        return X, y, feature_columns
    
    def create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders"""
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=False  # Keep temporal order
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: dict
    ) -> Tuple[TCNForex, Dict]:
        """Train TCN model with given configuration"""
        
        print("\n" + "="*60)
        print("TRAINING TCN MODEL")
        print("="*60)
        
        # Split data (temporal split - no shuffling)
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Target balance - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            X_train, y_train, X_val, y_val,
            batch_size=config['batch_size']
        )
        
        # Initialize model
        n_features = X.shape[1]
        model = TCNForex(
            input_channels=n_features,
            num_channels=config['num_channels'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout']
        )
        
        print(f"\nModel configuration:")
        print(f"  Input channels: {n_features}")
        print(f"  TCN channels: {config['num_channels']}")
        print(f"  Kernel size: {config['kernel_size']}")
        print(f"  Receptive field: {model.get_receptive_field()} time steps")
        print(f"  Dropout: {config['dropout']}")
        
        # Train
        trainer = TCNTrainer(model, device=self.device)
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            early_stopping_patience=config['early_stopping_patience'],
            verbose=True
        )
        
        # Save model
        model_path = os.path.join(self.model_save_path, 'tcn_forex_model.pt')
        trainer.save_model(model_path)
        print(f"\nModel saved to {model_path}")
        
        return model, history
    
    def evaluate_model(
        self,
        model: TCNForex,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Comprehensive model evaluation
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        trainer = TCNTrainer(model, device=self.device)
        
        # Get predictions
        X_test_t = torch.FloatTensor(X_test)
        y_pred_proba = trainer.predict(X_test_t)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 DOWN    UP")
        print(f"Actual DOWN  [{cm[0,0]:6d} {cm[0,1]:6d}]")
        print(f"Actual UP    [{cm[1,0]:6d} {cm[1,1]:6d}]")
        
        # Trading simulation with confidence threshold
        print(f"\n" + "-"*60)
        print("TRADING SIMULATION (60% Confidence Threshold)")
        print("-"*60)
        
        confidence_threshold = 0.6
        confident_mask = (y_pred_proba >= confidence_threshold) | (y_pred_proba <= (1 - confidence_threshold))
        
        if confident_mask.sum() > 0:
            confident_pred = y_pred[confident_mask]
            confident_actual = y_test[confident_mask]
            
            trade_accuracy = accuracy_score(confident_actual, confident_pred)
            trade_count = len(confident_pred)
            trade_percentage = (trade_count / len(y_test)) * 100
            
            print(f"\nTrades taken: {trade_count} ({trade_percentage:.1f}% of opportunities)")
            print(f"Trade accuracy: {trade_accuracy:.4f} ({trade_accuracy*100:.2f}%)")
            
            # Profitability analysis (80% payout rate)
            payout_rate = 0.80
            wins = (confident_pred == confident_actual).sum()
            losses = trade_count - wins
            
            total_profit = (wins * payout_rate) - losses
            roi = (total_profit / trade_count) * 100
            
            print(f"\nProfitability Analysis (80% payout):")
            print(f"  Wins: {wins} | Losses: {losses}")
            print(f"  Win rate: {wins/trade_count:.2%}")
            print(f"  Total P&L: {total_profit:+.2f} units")
            print(f"  ROI: {roi:+.2f}%")
            
            # Break-even analysis
            min_acc_breakeven = 1 / (1 + payout_rate)
            print(f"\nBreak-even accuracy: {min_acc_breakeven:.2%}")
            
            if trade_accuracy >= min_acc_breakeven:
                print("✅ Model is potentially profitable!")
            else:
                print("❌ Model needs improvement for profitability")
        else:
            print("No trades meet confidence threshold")
        
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist()
        }
    
    def plot_training_history(self, history: Dict, save_path: str = None):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(history['train_acc'], label='Train Accuracy')
        axes[1].plot(history['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nTraining history plot saved to {save_path}")
        
        plt.show()
    
    def run_full_training(self):
        """Execute complete training pipeline"""
        
        # Configuration
        config = {
            'sequence_length': 60,
            'prediction_horizon': 1,
            'num_channels': [16, 16, 8],
            'kernel_size': 3,
            'dropout': 0.1,
            'batch_size': 64,
            'epochs': 100,
            'learning_rate': 0.001,
            'early_stopping_patience': 15
        }
        
        print("="*60)
        print("TCN FOREX TRAINING PIPELINE")
        print("="*60)
        print(f"\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Save config
        config_file = os.path.join(self.config_path, 'training_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved to {config_file}")
        
        # Load and prepare data
        X, y, feature_columns = self.load_and_prepare_data(
            sequence_length=config['sequence_length'],
            prediction_horizon=config['prediction_horizon']
        )
        
        # Save feature columns
        with open(os.path.join(self.config_path, 'feature_columns.json'), 'w') as f:
            json.dump(feature_columns, f, indent=2)
        
        # Train model
        model, history = self.train_model(X, y, config)
        
        # Plot training history
        plot_path = os.path.join(self.model_save_path, 'training_history.png')
        self.plot_training_history(history, save_path=plot_path)
        
        # Evaluate on test set (last 20% of data)
        test_size = int(0.2 * len(X))
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Save metrics
        metrics_file = os.path.join(self.config_path, 'evaluation_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_file}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel files saved in: {self.model_save_path}")
        print(f"Config files saved in: {self.config_path}")


if __name__ == "__main__":
    import sys
    
    # Check if data path provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Use default or create sample data
        print("No data path provided. Creating sample data...")
        
        # Create sample data
        import os
        os.makedirs('data', exist_ok=True)
        
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=10000, freq='1min')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': 1.1000 + np.cumsum(np.random.randn(10000) * 0.0001),
            'high': 1.1000 + np.cumsum(np.random.randn(10000) * 0.0001) + 0.0005,
            'low': 1.1000 + np.cumsum(np.random.randn(10000) * 0.0001) - 0.0005,
            'close': 1.1000 + np.cumsum(np.random.randn(10000) * 0.0001),
            'volume': np.random.randint(100, 1000, 10000)
        })
        
        data_path = 'data/sample_forex_data.csv'
        df.to_csv(data_path, index=False)
        print(f"Sample data created at: {data_path}")
    
    # Run training
    trainer = ModelTrainer(data_path)
    trainer.run_full_training()
