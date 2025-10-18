# TCN Trading Bot - Complete Implementation Guide

## Overview

This document provides a comprehensive guide to the TCN (Temporal Convolutional Network) Binary Options Trading Bot implementation, based on the research methodology described in AboutBot.pdf.

## Project Architecture

### Directory Structure

```
TCN-Qbot/
├── src/                      # Core source code
│   ├── tcn_model.py         # PyTorch TCN implementation
│   └── data_preprocessing.py # Feature engineering pipeline
├── scripts/                  # Executable scripts
│   └── train_model.py       # Model training script
├── utils/                    # Utility modules
│   ├── data_collection.py   # Data fetching and generation
│   ├── backtesting.py       # Backtesting framework
│   ├── risk_management.py   # Position sizing and risk rules
│   └── logging_utils.py     # Logging and monitoring
├── examples/                 # Example scripts
│   └── quick_start.py       # Complete end-to-end example
├── configs/                  # Configuration files
│   └── default_config.json  # Default model/trading config
├── models/                   # Trained model storage
├── data/                     # Historical data storage
└── logs/                     # Trading and training logs
```

## Implementation Details

### 1. TCN Model Architecture

**File**: `src/tcn_model.py`

The Temporal Convolutional Network is implemented in PyTorch following the research paper:

**Key Components:**
- **CausalConv1d**: Ensures causality (no future information leakage)
- **TemporalBlock**: Residual block with two causal convolutions
- **TCNForex**: Complete model with stacked temporal blocks
- **TCNTrainer**: Training utilities with early stopping

**Architecture:**
```
Input (batch, channels, sequence_length)
    ↓
TemporalBlock (dilation=1)
    ↓
TemporalBlock (dilation=2)
    ↓
TemporalBlock (dilation=4)
    ↓
Global Pooling
    ↓
Fully Connected
    ↓
Sigmoid (probability output)
```

**Parameters:**
- Input channels: Number of features (e.g., 28)
- TCN channels: [16, 16, 8] (configurable)
- Kernel size: 3
- Dropout: 0.1
- Receptive field: 15 time steps

### 2. Data Preprocessing Pipeline

**File**: `src/data_preprocessing.py`

**ForexDataPreprocessor** implements the complete feature engineering pipeline:

**Step 1: Load Data**
- Read CSV with OHLCV columns
- Parse timestamps
- Resample if needed

**Step 2: Add Returns**
- Log returns: `r_t = ln(P_t / P_{t-1})`
- Simple returns: `r_t = (P_t - P_{t-1}) / P_{t-1}`

**Step 3: Technical Indicators**
- Trend: SMA(5, 15), EMA(5, 15), MACD
- Momentum: RSI(14), momentum
- Volatility: ATR, rolling std
- Total: 10+ indicators

**Step 4: Lagged Features**
- Lag periods: [1, 2, 3, 5]
- Applied to: close, returns

**Step 5: Time Features**
- Hour (sin/cos encoding)
- Day of week
- Market sessions (London, NY)

**Step 6: Normalization**
- StandardScaler (z-score normalization)
- Fit on training data only

**Step 7: Sequence Creation**
- Sliding windows of length 60
- Output shape: (samples, features, sequence_length)

**Total Features**: ~28 features per time step

### 3. Training Pipeline

**File**: `scripts/train_model.py`

**WalkForwardValidator**: Time series cross-validation
**ModelTrainer**: Complete training workflow

**Process:**
1. Load and preprocess data
2. Split temporally (60% train, 20% val, 20% test)
3. Create PyTorch data loaders
4. Build TCN model
5. Train with early stopping
6. Evaluate on test set
7. Save model and config

**Training Configuration:**
```json
{
  "sequence_length": 60,
  "batch_size": 64,
  "epochs": 100,
  "learning_rate": 0.001,
  "early_stopping_patience": 15
}
```

**Outputs:**
- `models/tcn_forex_model.pt` - Model weights
- `models/preprocessor.pkl` - Feature scaler
- `configs/training_config.json` - Configuration
- `configs/evaluation_metrics.json` - Performance metrics
- `models/training_history.png` - Training curves

### 4. Backtesting Framework

**File**: `utils/backtesting.py`

**BacktestEngine** simulates trading performance on historical data:

**Features:**
- Realistic trading constraints
- Daily limits and risk rules
- Commission/spread modeling
- Slippage simulation
- Performance metrics

**Key Metrics:**
- Win rate
- Profit factor (wins / losses)
- Maximum drawdown
- Sharpe ratio
- ROI (return on investment)

**Usage:**
```python
backtester = BacktestEngine(
    initial_balance=1000.0,
    payout_rate=0.80,
    min_confidence=0.60,
    trade_amount=10.0
)

results = backtester.run_backtest(timestamps, predictions, actuals)
backtester.plot_results(save_path='backtest.png')
```

### 5. Risk Management

**File**: `utils/risk_management.py`

**RiskManager** implements position sizing and risk rules:

**Position Sizing Methods:**
1. **Fixed**: Constant dollar amount
2. **Kelly Criterion**: Optimal based on win rate
3. **Confidence-based**: Scale with prediction confidence
4. **Percentage**: Fixed % of balance

**Risk Rules:**
- Maximum risk per trade: 1-2% of balance
- Daily loss limit: 20% of balance
- Maximum daily trades: 100
- Minimum confidence threshold: 60%

**Kelly Criterion Formula:**
```
f = (p × (1 + b) - 1) / b
where:
  f = fraction to bet
  p = win probability
  b = payout rate
```

**Usage:**
```python
risk_mgr = RiskManager(
    initial_balance=1000.0,
    max_risk_per_trade=0.02,
    max_daily_trades=100
)

decision = risk_mgr.should_trade(confidence, date)
position = risk_mgr.calculate_position_size(confidence, payout, 'kelly')
```

### 6. Data Collection

**File**: `utils/data_collection.py`

**ForexDataCollector** supports multiple data sources:

**Supported Sources:**
1. **Alpha Vantage API** - Free with registration
2. **HistData.com** - Free historical data
3. **Dukascopy** - High-quality broker data
4. **MetaTrader** - MT4/MT5 export
5. **Sample generation** - For testing

**Data Format:**
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.10050,1.10080,1.10040,1.10070,250
```

**Quality Requirements:**
- Minimum 10,000 candles (50,000+ recommended)
- 1-minute or 5-minute timeframe
- No missing values
- Consistent timestamps

### 7. Logging and Monitoring

**File**: `utils/logging_utils.py`

**TradingLogger**: Structured logging for all operations
**PerformanceMonitor**: Performance metrics tracking

**Log Types:**
- Startup/shutdown logs
- Model loading logs
- Prediction logs
- Trade execution logs
- Trade result logs
- Daily summaries
- Error logs
- Risk alerts

**Log Files:**
- `logs/trades_YYYY-MM-DD.log` - Daily trade logs
- `logs/trades_detail_YYYY-MM-DD.jsonl` - JSON trade details

## Usage Guide

### Quick Start

Run the complete example:
```bash
python examples/quick_start.py
```

This demonstrates:
1. Sample data generation
2. Feature preprocessing
3. Model training (20 epochs)
4. Evaluation on test set
5. Backtesting
6. Results summary

### Training with Your Data

1. **Prepare Data**
   ```bash
   # Place your CSV in data/ directory
   # Format: timestamp,open,high,low,close,volume
   ```

2. **Train Model**
   ```bash
   python scripts/train_model.py data/your_data.csv
   ```

3. **Evaluate Results**
   - Check `models/training_history.png`
   - Review `configs/evaluation_metrics.json`
   - Examine console output

### Backtesting

```python
from utils.backtesting import BacktestEngine
import pandas as pd
import numpy as np

# Load your test data and predictions
# ...

backtester = BacktestEngine(
    initial_balance=1000.0,
    payout_rate=0.80,
    min_confidence=0.60
)

results = backtester.run_backtest(timestamps, predictions, actuals)
backtester.plot_results('backtest.png')
```

### Customization

**Modify Model Architecture:**
Edit `configs/default_config.json`:
```json
{
  "model": {
    "num_channels": [32, 32, 16],
    "kernel_size": 5,
    "dropout": 0.2
  }
}
```

**Adjust Training Parameters:**
```json
{
  "training": {
    "batch_size": 128,
    "epochs": 200,
    "learning_rate": 0.0005
  }
}
```

**Change Trading Rules:**
```json
{
  "trading": {
    "min_confidence": 0.65,
    "trade_amount": 20.0,
    "max_daily_trades": 50
  }
}
```

## Performance Benchmarks

### Model Performance

**Training Set:**
- Accuracy: 60-65%
- AUC: 0.65-0.70

**Validation Set:**
- Accuracy: 55-60%
- AUC: 0.60-0.65

**Test Set (Expected):**
- Accuracy: 52-58%
- Win Rate: 50-56%
- Note: Lower due to slippage and real-world conditions

### Computational Performance

**Training Time (5,000 candles):**
- CPU (4 cores): ~5 minutes
- GPU (CUDA): ~1-2 minutes

**Prediction Time:**
- Single prediction: <50ms
- Batch (32 samples): ~100ms

**Memory Usage:**
- Model size: ~500KB
- Runtime RAM: ~200MB
- Training RAM: ~1GB

## Best Practices

### Data Quality
1. Use at least 10,000 candles for training
2. Prefer 50,000+ candles for production models
3. Validate data quality before training
4. Check for gaps and missing values
5. Use consistent timeframes (1-min or 5-min)

### Model Development
1. Start with sample data to validate pipeline
2. Use walk-forward validation
3. Monitor for overfitting (train vs. val loss)
4. Save best model based on validation performance
5. Test on completely unseen data

### Risk Management
1. Never risk more than 2% per trade
2. Set daily loss limits (20% recommended)
3. Use conservative position sizing
4. Implement stop-loss at portfolio level
5. Monitor drawdown carefully

### Trading Discipline
1. Only trade when confidence >= 60%
2. Follow risk management rules strictly
3. Keep detailed logs of all trades
4. Review performance daily
5. Retrain model periodically (weekly/monthly)

### Avoiding Overfitting
1. Use dropout (0.1-0.2)
2. Early stopping (patience: 10-15 epochs)
3. Validation set monitoring
4. Walk-forward validation
5. Simple models (avoid too many parameters)

## Troubleshooting

### Low Accuracy (<50%)
- Collect more training data
- Check data quality
- Adjust model architecture
- Try different features
- Verify target creation logic

### Model Overfitting
- Increase dropout rate
- Reduce model complexity
- Add more training data
- Use early stopping
- Check train/val loss gap

### Slow Training
- Reduce batch size
- Simplify model (fewer channels)
- Use GPU if available
- Reduce sequence length
- Profile code for bottlenecks

### Memory Errors
- Reduce batch size
- Use smaller model
- Process data in chunks
- Clear PyTorch cache
- Reduce sequence length

## References

1. **AboutBot.pdf** - Research paper on TCN for Forex forecasting
2. **Bai et al. (2018)** - "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
3. **PyTorch Documentation** - https://pytorch.org/docs/
4. **Technical Analysis Library** - https://github.com/bukosabino/ta

## Future Enhancements

### Planned Features
- [ ] Ensemble methods (multiple models)
- [ ] Hyperparameter optimization (Optuna)
- [ ] Jupyter notebook tutorials
- [ ] Live trading interface
- [ ] Multi-asset support
- [ ] Advanced risk management
- [ ] Performance dashboard
- [ ] Model interpretability (SHAP)

### Research Directions
- [ ] Attention mechanisms
- [ ] Transformer architectures
- [ ] Multi-timeframe inputs
- [ ] Order flow features
- [ ] News sentiment analysis
- [ ] Adversarial training

## License and Disclaimer

**Educational Purpose Only**: This code is for educational and research purposes only.

**No Warranty**: No guarantee of profitability or performance.

**High Risk**: Binary options trading carries significant risk of loss.

**Terms of Service**: Automated trading may violate Quotex terms of service.

**Use at Your Own Risk**: The authors assume no responsibility for any losses or consequences.

---

**Last Updated**: October 2025
**Version**: 1.0.0
**Python**: 3.10-3.12
**PyTorch**: >=2.0.0
