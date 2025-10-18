# TCN-Based Binary Options Trading Bot for Quotex

A sophisticated trading bot that uses **PyTorch-based Temporal Convolutional Networks (TCN)** deep learning model combined with technical indicators to predict Forex candle direction for binary options trading.

## 📚 Research-Based Implementation

This implementation is based on the research paper methodology described in `AboutBot.pdf`:
- **Temporal Convolutional Networks** with dilated causal convolutions
- **Long-range dependencies** through exponentially increasing dilation factors
- **Multivariate input channels** (OHLC + technical indicators)
- **Binary classification** for directional prediction (up/down)
- **Walk-forward validation** for robust time series evaluation

**Reference**: Bai et al. (2018) - "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"

## ⚠️ IMPORTANT WARNING

**Quotex explicitly prohibits the use of automated trading bots.** According to their terms:

> "The use of automated bots for trading is prohibited according to account rules. If you use them, your account will be blocked without the possibility of withdrawing funds and re-registration."

**This code is provided for EDUCATIONAL PURPOSES ONLY.** The author is not responsible for any losses, account suspensions, or other consequences resulting from using this bot.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Strategy Explanation](#strategy-explanation)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

## ✨ Features

### Deep Learning Model (PyTorch Implementation)
- **TCN Architecture**: State-of-the-art temporal convolutional network with causal dilated convolutions
- **Long Memory**: Exponentially increasing dilation [1, 2, 4, 8, 16, 32] captures patterns from 60+ candles
- **Parallel Processing**: Faster training and inference compared to LSTM/RNN models
- **Residual Connections**: Skip connections for better gradient flow
- **Batch Normalization**: Stable and faster training

### Technical Indicators
The bot uses 30+ technical indicators for feature engineering:

**Trend Indicators:**
- MACD (Moving Average Convergence Divergence)
- EMA (9, 21 periods)
- SMA (50 period)

**Momentum Indicators:**
- RSI (Relative Strength Index)
- Stochastic Oscillator

**Volatility Indicators:**
- Bollinger Bands
- ATR (Average True Range)

**Price Features:**
- Log returns and percentage changes
- High-Low range
- Open-Close differences
- Lagged features (1, 2, 3, 5 periods)

**Time Features:**
- Hour of day (cyclical encoding)
- Day of week
- Market session indicators (London, NY)

### Trading Features
- **Confidence-Based Trading**: Only trades when prediction confidence >= 60%
- **Real-Time Prediction**: Live candle analysis and prediction
- **Risk Management**: Configurable trade amount, max daily trades, and loss limits
- **Performance Tracking**: Win rate, total trades, and P&L monitoring
- **Walk-Forward Validation**: Proper time series cross-validation
- **Backtesting**: Historical performance simulation

## 🔧 Installation

### Requirements
- Python >= 3.10, <= 3.12
- pip package manager
- At least 4GB RAM (8GB recommended for training)
- GPU recommended for faster training (optional)
- Internet connection

### Step 1: Clone Repository

```bash
git clone https://github.com/Akrambro/TCN-Qbot.git
cd TCN-Qbot
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Detailed Package Installation

```bash
# Core deep learning
pip install torch>=2.0.0              # PyTorch for TCN
pip install torchvision>=0.15.0       # PyTorch vision utilities

# Data processing
pip install pandas>=2.0.0             # Data manipulation
pip install numpy>=1.24.0             # Numerical computing

# Technical analysis
pip install ta>=0.11.0                # Technical indicators library

# Machine learning utilities
pip install scikit-learn>=1.3.0       # ML utilities and metrics

# Visualization
pip install matplotlib>=3.7.0         # Plotting
pip install seaborn>=0.12.0          # Statistical visualization

# API clients (optional for data collection)
pip install requests>=2.31.0          # HTTP requests

# For live trading (optional)
pip install quotexpy==1.40.7         # Quotex API wrapper
```

## 📁 Project Structure

```
TCN-Qbot/
│
├── src/                              # Core source code
│   ├── __init__.py
│   ├── tcn_model.py                 # PyTorch TCN implementation
│   └── data_preprocessing.py        # Feature engineering pipeline
│
├── scripts/                          # Executable scripts
│   └── train_model.py               # Model training script
│
├── utils/                            # Utility functions
│   ├── __init__.py
│   └── data_collection.py           # Data collection utilities
│
├── configs/                          # Configuration files
│   └── default_config.json          # Default model/trading config
│
├── models/                           # Trained models (created after training)
│   ├── tcn_forex_model.pt           # Trained PyTorch model
│   ├── preprocessor.pkl             # Feature scaler
│   └── training_history.png         # Training curves
│
├── data/                             # Historical data
│   ├── README.md                    # Data sources guide
│   └── *.csv                        # Your market data files
│
├── logs/                             # Trading logs (created automatically)
│   └── trades_YYYY-MM-DD.log        # Daily trade logs
│
├── tests/                            # Unit tests (optional)
│
├── AboutBot.pdf                      # Research paper reference
├── PROJECT_SUMMARY.txt               # Project overview
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── tcn_quotex_bot.py                # Legacy Keras implementation
└── train_tcn_model.py               # Legacy training script
```

## 🚀 Quick Start

### 1. Collect or Prepare Historical Data

You have several options:

**Option A: Generate Sample Data (for testing)**
```bash
python -c "
from utils.data_collection import ForexDataCollector
collector = ForexDataCollector()
df = collector.generate_sample_data(pair='EURUSD', n_candles=10000, save=True)
"
```

**Option B: Use Alpha Vantage API**
```python
from utils.data_collection import ForexDataCollector

collector = ForexDataCollector()
df = collector.collect_from_alpha_vantage(
    symbol='EURUSD',
    api_key='YOUR_API_KEY',  # Get free key at alphavantage.co
    interval='1min'
)
```

**Option C: Use Your Own CSV Data**
Place your CSV file in the `data/` directory with format:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.10050,1.10080,1.10040,1.10070,250
2024-01-01 00:01:00,1.10070,1.10090,1.10060,1.10085,300
```

See `data/README.md` for more data source options (HistData.com, Dukascopy, MetaTrader, etc.)

### 2. Train the Model

```bash
# Train with your data
python scripts/train_model.py data/your_data.csv

# Or use sample data (will be generated automatically)
python scripts/train_model.py
```

**Training Output:**
- `models/tcn_forex_model.pt` - Trained model weights
- `models/preprocessor.pkl` - Feature scaler
- `models/training_history.png` - Training metrics plot
- `configs/training_config.json` - Model configuration
- `configs/evaluation_metrics.json` - Performance metrics

**Expected Training Time:**
- 10,000 candles: ~3-5 minutes (CPU) / 1-2 minutes (GPU)
- 50,000 candles: ~15-20 minutes (CPU) / 5-10 minutes (GPU)

### 3. Evaluate Model Performance

After training, the script automatically evaluates the model and shows:
- **Classification Metrics**: Accuracy, Precision, Recall, F1, AUC
- **Confusion Matrix**: True positives, false positives, etc.
- **Trading Simulation**: Profitability with 60% confidence threshold
- **Break-even Analysis**: Minimum accuracy needed for profit

Look for:
- ✅ Validation accuracy >= 55% (break-even)
- ✅ Trading accuracy >= 60% (profitable)
- ✅ AUC >= 0.60 (decent discrimination)

### 4. Backtest Strategy (Optional)

```python
import torch
from src.tcn_model import TCNForex, TCNTrainer
from src.data_preprocessing import ForexDataPreprocessor

# Load model
model = TCNForex(input_channels=30, num_channels=[16, 16, 8])
model.load_state_dict(torch.load('models/tcn_forex_model.pt'))

# Load and preprocess new data
preprocessor = ForexDataPreprocessor()
# ... process data ...

# Make predictions
trainer = TCNTrainer(model)
predictions = trainer.predict(X_test)
```

## 🏗️ Model Architecture

### TCN Design (from AboutBot.pdf)

The Temporal Convolutional Network architecture consists of:

1. **Causal Convolutions**: Ensures predictions at time t only use data from t and earlier
2. **Dilated Convolutions**: Exponentially increasing receptive field (1, 2, 4, 8, 16, 32)
3. **Residual Blocks**: Skip connections for better gradient flow
4. **Batch Normalization**: Stable training and faster convergence

```
Input (batch, channels, sequence_length=60)
    ↓
TemporalBlock1 (dilation=1, filters=16)
    ↓
TemporalBlock2 (dilation=2, filters=16)
    ↓
TemporalBlock3 (dilation=4, filters=8)
    ↓
Global Pooling (last time step)
    ↓
Fully Connected (output=1)
    ↓
Sigmoid Activation
    ↓
Output: Probability [0, 1]
```

**Receptive Field Calculation:**
- With 3 blocks and kernel_size=3: RF = (3-1) × (2³ - 1) + 1 = 15 time steps
- Can see up to 15 candles back effectively

### Why TCN Works Better Than LSTM

| Feature | TCN | LSTM |
|---------|-----|------|
| Training Speed | **Parallel** (faster) | Sequential (slower) |
| Memory | **Stable long-term** | Vanishing gradients |
| Complexity | **Simple** | Complex gates |
| Inference | **Fast** | Moderate |
| Receptive Field | **Controllable** (dilation) | Fixed |

### Model Parameters

Default configuration (`configs/default_config.json`):

```json
{
  "model": {
    "sequence_length": 60,
    "num_channels": [16, 16, 8],
    "kernel_size": 3,
    "dropout": 0.1
  },
  "training": {
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.001,
    "early_stopping_patience": 15
  }
}
```

You can customize these parameters based on your data and computational resources.

## 📊 Data Collection

### Supported Data Sources

The `utils/data_collection.py` module supports multiple data sources:

1. **HistData.com** - Free historical tick and 1-minute data
2. **Dukascopy** - High-quality Swiss broker data
3. **Alpha Vantage API** - Free with registration (5 calls/min)
4. **OANDA API** - Practice account access
5. **MetaTrader** - Export from MT4/MT5

### Data Quality Requirements

**Minimum Requirements:**
- At least 10,000 candles (50,000+ recommended)
- 1-minute or 5-minute timeframe
- Consistent timestamp format
- No missing values or large gaps

**Data Validation:**
```python
from utils.data_collection import ForexDataCollector

collector = ForexDataCollector()
df = collector.load_from_csv('data/your_data.csv')
validation = collector.validate_data(df)

if validation['valid']:
    print("✅ Data is ready for training")
else:
    print("❌ Data quality issues found")
```

## 🎓 Model Training

### Training Process Explained

The training script performs these steps:

1. **Data Loading**: Loads historical OHLCV data
2. **Feature Engineering**: Adds 30+ technical indicators
3. **Sequence Creation**: Creates 60-candle sequences for TCN input
4. **Model Building**: Constructs TCN architecture with:
   - Multiple temporal blocks (dilated convolutions)
   - Residual connections for gradient flow
   - Batch normalization for stability
   - Dropout for regularization
5. **Training**: Trains with Adam optimizer and early stopping
6. **Evaluation**: Tests on validation set
7. **Saving**: Saves model, scaler, and config

### Using the Training Script

```bash
# Train with custom data
python scripts/train_model.py data/EURUSD_1min.csv

# Or use the example which generates sample data
python examples/quick_start.py
```

### Understanding Training Metrics

**Accuracy**: Percentage of correct predictions
- Target: >= 55% for break-even
- Good: >= 60% for profitability
- Excellent: >= 65% for consistent profits

**AUC (Area Under Curve)**: Model discrimination ability
- 0.5 = Random guessing
- 0.6-0.7 = Acceptable
- 0.7-0.8 = Good
- 0.8+ = Excellent

**Loss**: Model error (lower is better)
- Should decrease during training
- Validation loss should follow training loss
- Large gap = overfitting

### Walk-Forward Validation

For production models, the training script supports walk-forward validation for time series:

```python
from scripts.train_model import WalkForwardValidator

validator = WalkForwardValidator(n_splits=5)
for train_idx, test_idx in validator.split(X, y):
    # Train and evaluate on each fold
    pass
```

## 📊 Backtesting

### Running Backtests

After training, evaluate strategy performance on historical data:

```python
from utils.backtesting import BacktestEngine

backtester = BacktestEngine(
    initial_balance=1000.0,
    payout_rate=0.80,
    min_confidence=0.60,
    trade_amount=10.0
)

results = backtester.run_backtest(timestamps, predictions, actuals)
backtester.plot_results(save_path='backtest_results.png')
```

### Key Metrics

- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Total wins / Total losses
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns
- **ROI**: Return on investment

### Realistic Expectations

- Training accuracy: 60-65%
- Validation accuracy: 55-60%
- **Live trading: 52-58%** (accounting for slippage)
- Requires continuous monitoring
- Regular model retraining needed
- Market conditions affect performance

## 🛡️ Risk Management

### Position Sizing

The bot supports multiple position sizing methods:

```python
from utils.risk_management import RiskManager

risk_mgr = RiskManager(
    initial_balance=1000.0,
    max_risk_per_trade=0.02,  # 2% per trade
    max_daily_trades=100,
    max_daily_loss_pct=0.20   # 20% max daily loss
)

# Check if should trade
decision = risk_mgr.should_trade(prediction_confidence, date)

# Calculate position size
position = risk_mgr.calculate_position_size(
    prediction_confidence,
    payout_rate=0.80,
    sizing_method='kelly'  # 'fixed', 'kelly', 'confidence'
)
```

### Risk Rules

1. **Maximum Risk Per Trade**: 1-2% of balance
2. **Daily Loss Limit**: Stop trading after 20% daily loss
3. **Maximum Daily Trades**: Limit number of trades per day
4. **Confidence Threshold**: Only trade when confidence >= 60%
5. **Position Sizing**: Use Kelly criterion or fixed fractional

### Money Management

```python
# Conservative approach (recommended)
trade_amount = 0.01 * balance  # 1% per trade

# Aggressive approach (higher risk)
trade_amount = 0.02 * balance  # 2% per trade

# Kelly criterion (optimal but risky)
kelly_fraction = (win_rate * (1 + payout) - 1) / payout
trade_amount = kelly_fraction * 0.25 * balance  # Quarter Kelly
```

## 📊 Live Trading

### Trading Workflow

1. **Initialization**
   - Connect to Quotex
   - Load trained model
   - Set account type (Practice/Real)

2. **Data Collection**
   - Fetch last 100 candles every 60s
   - Calculate technical indicators
   - Normalize features

3. **Prediction**
   - Feed last 50 candles to TCN
   - Get probability (0-1)
   - Determine signal:
     - probability >= 0.6 → CALL
     - probability <= 0.4 → PUT
     - 0.4 < probability < 0.6 → NO TRADE

4. **Trade Execution**
   - Place trade if signal present
   - Track trade ID
   - Monitor result

5. **Performance Tracking**
   - Update win/loss statistics
   - Calculate win rate
   - Log all trades

### Risk Management

**Position Sizing:**
- Start with minimum ($1-10)
- Never risk > 2% of balance per trade
- Use fixed fractional sizing

**Stop Loss:**
Binary options have built-in stop loss (entire trade amount)

**Daily Limits:**
```python
# Add to bot class
self.max_daily_trades = 50
self.max_daily_loss = 100  # USD
self.daily_trades = 0
self.daily_pnl = 0
```

**Confidence Thresholds:**
- 60%: Moderate risk, more trades
- 65%: Conservative, fewer trades
- 70%: Very conservative, rare trades

## 🎯 Strategy Explanation

### Why This Strategy Works

**1. Multi-Timeframe Analysis**
- TCN sees patterns across 50 candles
- Dilated convolutions capture both short and long-term trends
- Receptive field of 559 time steps

**2. Technical Indicator Confirmation**
- MACD confirms trend direction
- RSI identifies overbought/oversold
- Bollinger Bands show volatility
- Multiple indicators reduce false signals

**3. Machine Learning Advantages**
- Learns non-linear relationships
- Adapts to market conditions
- Finds patterns humans miss
- Processes information instantly

### Theoretical Win Rate

**Break-Even Analysis:**
With 80% payout:
- Win: +$0.80
- Loss: -$1.00
- Break-even win rate: 55.6%

**Profitable Scenarios:**
- 56% accuracy: +$0.96 per 100 trades
- 60% accuracy: +$8 per 100 trades
- 65% accuracy: +$17 per 100 trades

**Reality Check:**
- Models typically achieve 52-58% in live trading
- Slippage, latency reduce real performance
- Market regime changes affect accuracy
- Continuous retraining essential

## ⚙️ Performance Optimization

### Model Optimization

**1. Hyperparameter Tuning:**
```python
# Try different configurations
configs = [
    {'nb_filters': 32, 'kernel_size': 2},
    {'nb_filters': 64, 'kernel_size': 3},
    {'nb_filters': 128, 'kernel_size': 4},
]
```

**2. Feature Selection:**
```python
# Use feature importance
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_2d, y)  # Reshape X to 2D
importances = rf.feature_importances_
# Select top features
```

**3. Ensemble Methods:**
```python
# Combine multiple models
predictions = []
for model in models:
    pred = model.predict(X)
    predictions.append(pred)
final_pred = np.mean(predictions, axis=0)
```

### System Optimization

**1. Reduce Latency:**
- Use WebSocket for real-time data
- Pre-compute indicators
- Cache predictions

**2. Improve Throughput:**
- Batch predictions
- Async I/O operations
- GPU inference

**3. Error Handling:**
```python
try:
    # Trading logic
except ConnectionError:
    # Reconnect
except Exception as e:
    # Log and continue
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'quotexpy'
```
**Solution:** Install missing package
```bash
pip install quotexpy
```

**2. Connection Failed**
```
Connection failed: Invalid credentials
```
**Solution:** Check email/password, ensure account is active

**3. Model Loading Error**
```
FileNotFoundError: tcn_quotex_model.h5 not found
```
**Solution:** Train model first using `train_tcn_model.py`

**4. Insufficient Data**
```
ValueError: Not enough data for sequence creation
```
**Solution:** Fetch more historical candles (minimum 50 + sequence_length)

**5. Low Accuracy**
```
Model accuracy: 48%
```
**Solution:**
- Collect more training data
- Try different features
- Adjust model architecture
- Check data quality

### Performance Issues

**Slow Predictions:**
- Use GPU for inference
- Reduce sequence length
- Simplify model architecture

**Memory Errors:**
- Reduce batch size
- Use smaller model
- Process data in chunks

**Connection Drops:**
- Implement reconnection logic
- Use stable internet
- Add timeout handling

## 📚 Additional Resources

### Learning Resources

**Deep Learning:**
- [Temporal Convolutional Networks Paper](https://arxiv.org/abs/1803.01271)
- [Keras Documentation](https://keras.io)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

**Trading:**
- [Binary Options Strategies](https://www.binaryoptions.com/strategies/)
- [Technical Analysis](https://www.investopedia.com/technical-analysis-4689657)
- [Risk Management](https://www.investopedia.com/terms/r/riskmanagement.asp)

**Python:**
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Asyncio Guide](https://docs.python.org/3/library/asyncio.html)

### Similar Projects

- [quotexpy GitHub](https://github.com/SantiiRepair/quotexpy)
- [keras-tcn GitHub](https://github.com/philipperemy/keras-tcn)
- [ta Python Library](https://github.com/bukosabino/ta)

## 📄 License

This project is provided "as is" for educational purposes only. No warranty or guarantee of profitability is provided. Use at your own risk.

## 🤝 Contributing

This is an educational project. Contributions, suggestions, and improvements are welcome!

## 📞 Support

For technical issues:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify data format and quality

## 🎓 Educational Disclaimer

This bot is designed for educational purposes to demonstrate:
- Deep learning applications in finance
- Time series prediction with TCN
- Technical analysis programming
- Automated trading system design

**It is NOT:**
- A guaranteed profit system
- Professional trading advice
- Suitable for live trading without extensive testing
- Compliant with Quotex terms of service

## ⚖️ Final Warning

**Trading binary options carries significant risk. You can lose all invested capital. Never trade with money you cannot afford to lose. This bot is for educational purposes only and should not be used for actual trading on Quotex or any other platform.**

**The author assumes no responsibility for financial losses, account suspensions, or any other consequences resulting from the use of this code.**

---

**Version:** 1.0.0  
**Last Updated:** October 2025  
**Author:** TCN Trading Bot Project  
**Python Version:** 3.10-3.12
