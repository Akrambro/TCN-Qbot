
# Create a comprehensive training script for the TCN model

training_script = '''"""
TCN Model Training Script for Binary Options
============================================

This script demonstrates how to train the TCN model with historical data.
It includes data collection, feature engineering, model training, and evaluation.

Author: TCN Trading Bot
Date: October 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Import from main bot
from tcn_quotex_bot import TCNModel, TechnicalIndicators


class ModelTrainer:
    """Train and evaluate TCN model"""
    
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.model = None
        
    def load_data(self, filepath):
        """
        Load historical data from CSV
        
        Expected columns: timestamp, open, high, low, close, volume
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Convert timestamp if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Loaded {len(df)} candles")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("\\nPreparing features...")
        
        # Add technical indicators
        df_features = TechnicalIndicators.add_indicators(df)
        
        # Select feature columns
        feature_columns = [col for col in df_features.columns 
                          if col not in ['timestamp', 'target', 'open', 'high', 
                                       'low', 'close', 'volume']]
        
        print(f"Total features: {len(feature_columns)}")
        print(f"Feature columns: {feature_columns[:5]}... (showing first 5)")
        
        return df_features, feature_columns
    
    def train_model(self, df_features, feature_columns, 
                   train_split=0.8, epochs=100, batch_size=32):
        """Train TCN model"""
        
        # Create model
        n_features = len(feature_columns)
        self.model = TCNModel(
            sequence_length=self.sequence_length,
            n_features=n_features
        )
        
        print(f"\\nCreating TCN model...")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Number of features: {n_features}")
        
        # Prepare sequences
        print("\\nPreparing sequences...")
        X, y = self.model.prepare_sequences(df_features, feature_columns)
        
        print(f"Total sequences: {len(X)}")
        print(f"Input shape: {X.shape}")
        print(f"Target distribution - Up: {y.sum()}, Down: {len(y) - y.sum()}")
        print(f"Target balance: {y.mean():.2%} up candles")
        
        # Split data (maintain temporal order)
        split_idx = int(train_split * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"\\nTraining set: {len(X_train)} sequences")
        print(f"Validation set: {len(X_val)} sequences")
        
        # Build and train model
        print("\\nBuilding model...")
        self.model.build_model()
        
        print("\\nTraining model...")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print("-" * 60)
        
        history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history, X_val, y_val
    
    def evaluate_model(self, X_val, y_val):
        """Evaluate model performance"""
        print("\\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        # Make predictions
        y_pred_proba = self.model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        print(f"\\nOverall Metrics:")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        print(f"\\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                DOWN    UP")
        print(f"Actual DOWN  [{cm[0,0]:5d}  {cm[0,1]:5d}]")
        print(f"Actual UP    [{cm[1,0]:5d}  {cm[1,1]:5d}]")
        
        # Trading simulation
        print(f"\\n" + "-"*60)
        print("Trading Simulation (60% Confidence Threshold)")
        print("-"*60)
        
        # Only trade when confidence >= 60%
        confident_mask = (y_pred_proba >= 0.6) | (y_pred_proba <= 0.4)
        confident_pred = y_pred[confident_mask.flatten()]
        confident_actual = y_val[confident_mask.flatten()]
        
        if len(confident_pred) > 0:
            trade_accuracy = accuracy_score(confident_actual, confident_pred)
            trade_count = len(confident_pred)
            trade_percentage = (trade_count / len(y_val)) * 100
            
            print(f"Trades taken: {trade_count} ({trade_percentage:.1f}% of opportunities)")
            print(f"Trade accuracy: {trade_accuracy:.4f} ({trade_accuracy*100:.2f}%)")
            
            # Estimate profitability (assuming 80% payout)
            payout_rate = 0.80
            wins = np.sum(confident_pred == confident_actual)
            losses = trade_count - wins
            
            profit_per_win = 1 * payout_rate
            loss_per_trade = -1
            
            total_profit = (wins * profit_per_win) + (losses * loss_per_trade)
            roi = (total_profit / trade_count) * 100
            
            print(f"\\nProfitability Analysis (80% payout):")
            print(f"Wins: {wins} | Losses: {losses}")
            print(f"Total P&L: {total_profit:+.2f} units")
            print(f"ROI: {roi:+.2f}%")
            
            # Minimum accuracy for profitability
            min_acc_breakeven = 1 / (1 + payout_rate)
            print(f"\\nMinimum accuracy for breakeven: {min_acc_breakeven:.2%}")
            
            if trade_accuracy >= min_acc_breakeven:
                print("✅ Model is potentially profitable!")
            else:
                print("❌ Model needs improvement for profitability")
        else:
            print("No trades meet the confidence threshold")
        
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'trade_accuracy': trade_accuracy if len(confident_pred) > 0 else 0,
            'trade_count': len(confident_pred) if len(confident_pred) > 0 else 0
        }
    
    def plot_training_history(self, history):
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Accuracy
            axes[0, 0].plot(history.history['accuracy'], label='Train')
            axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Loss
            axes[0, 1].plot(history.history['loss'], label='Train')
            axes[0, 1].plot(history.history['val_loss'], label='Validation')
            axes[0, 1].set_title('Model Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # AUC
            if 'auc' in history.history:
                axes[1, 0].plot(history.history['auc'], label='Train')
                axes[1, 0].plot(history.history['val_auc'], label='Validation')
                axes[1, 0].set_title('Model AUC')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('AUC')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning rate
            axes[1, 1].axis('off')
            textstr = f'Final Metrics:\\n'
            textstr += f'Train Accuracy: {history.history["accuracy"][-1]:.4f}\\n'
            textstr += f'Val Accuracy: {history.history["val_accuracy"][-1]:.4f}\\n'
            textstr += f'Train Loss: {history.history["loss"][-1]:.4f}\\n'
            textstr += f'Val Loss: {history.history["val_loss"][-1]:.4f}'
            axes[1, 1].text(0.1, 0.5, textstr, fontsize=12, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=150)
            print("\\nTraining history plot saved as 'training_history.png'")
            
        except Exception as e:
            print(f"Could not plot training history: {e}")
    
    def save_model(self, filepath='tcn_quotex_model.h5'):
        """Save trained model"""
        self.model.save(filepath)
        print(f"\\nModel saved successfully!")


def create_sample_data():
    """
    Create sample historical data for demonstration
    In real usage, replace this with actual market data
    """
    print("Creating sample data for demonstration...")
    print("WARNING: This is synthetic data. Use real market data for actual trading!")
    
    np.random.seed(42)
    n_candles = 10000
    
    # Generate realistic price data using random walk
    returns = np.random.normal(0.0001, 0.01, n_candles)
    price = 1.1000  # Starting price (like EURUSD)
    prices = [price]
    
    for ret in returns:
        price = price * (1 + ret)
        prices.append(price)
    
    prices = np.array(prices[1:])
    
    # Generate OHLC
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_candles, freq='1min'),
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_candles))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_candles))),
        'close': prices * (1 + np.random.normal(0, 0.001, n_candles)),
        'volume': np.random.randint(100, 1000, n_candles)
    })
    
    # Ensure high >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Save to CSV
    df.to_csv('sample_historical_data.csv', index=False)
    print(f"Sample data saved to 'sample_historical_data.csv'")
    print(f"Candles: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def main():
    """Main training function"""
    print("="*60)
    print("TCN Model Training Script")
    print("="*60)
    
    # Option 1: Use sample data for demonstration
    print("\\n[1] Using sample data for demonstration")
    df = create_sample_data()
    
    # Option 2: Load your own data (uncomment and modify)
    # print("\\n[2] Loading your historical data")
    # df = pd.read_csv('your_historical_data.csv')
    
    # Initialize trainer
    trainer = ModelTrainer(sequence_length=50)
    
    # Prepare data
    df_features, feature_columns = trainer.prepare_data(df)
    
    # Train model
    history, X_val, y_val = trainer.train_model(
        df_features,
        feature_columns,
        train_split=0.8,
        epochs=50,  # Increase for better results
        batch_size=32
    )
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Evaluate model
    metrics = trainer.evaluate_model(X_val, y_val)
    
    # Save model
    trainer.save_model('tcn_quotex_model.h5')
    
    print("\\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\\nNext steps:")
    print("1. Review the model performance metrics above")
    print("2. If accuracy is >= 55-60%, the model may be profitable")
    print("3. Load the model in tcn_quotex_bot.py for live trading")
    print("4. Start with PRACTICE mode to test without risk")
    print("="*60)


if __name__ == "__main__":
    main()
'''

# Save the training script
with open('train_tcn_model.py', 'w', encoding='utf-8') as f:
    f.write(training_script)

print("✅ Training script created successfully!")
print(f"📄 File saved as: train_tcn_model.py")
print(f"📊 Total lines of code: {len(training_script.splitlines())}")
