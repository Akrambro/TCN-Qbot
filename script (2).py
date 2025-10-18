
# Create a comprehensive README file with setup instructions and usage guide

readme_content = '''# TCN-Based Binary Options Trading Bot for Quotex

A sophisticated trading bot that uses **Temporal Convolutional Networks (TCN)** deep learning model combined with technical indicators to predict next candle direction for binary options trading on the Quotex platform.

## ⚠️ IMPORTANT WARNING

**Quotex explicitly prohibits the use of automated trading bots.** According to their terms:

> "The use of automated bots for trading is prohibited according to account rules. If you use them, your account will be blocked without the possibility of withdrawing funds and re-registration."

**This code is provided for EDUCATIONAL PURPOSES ONLY.** The author is not responsible for any losses, account suspensions, or other consequences resulting from using this bot.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Model Training](#model-training)
- [Live Trading](#live-trading)
- [Strategy Explanation](#strategy-explanation)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

## ✨ Features

### Deep Learning Model
- **TCN Architecture**: State-of-the-art temporal convolutional network with dilated causal convolutions
- **Long Memory**: Captures patterns from 50+ previous candles using exponential receptive field
- **Parallel Processing**: Faster training and inference compared to LSTM/RNN models
- **Dropout Regularization**: Prevents overfitting with 20-30% dropout layers

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
- Price changes and returns
- High-Low range
- Open-Close differences
- Lagged features (1, 2, 3, 5 periods)

### Trading Features
- **Confidence-Based Trading**: Only trades when prediction confidence >= 60%
- **Real-Time Prediction**: Live candle analysis and prediction
- **Risk Management**: Configurable trade amount and expiry time
- **Performance Tracking**: Win rate, total trades, and P&L monitoring
- **Practice Mode**: Test strategies without risking real money

## 🔧 Installation

### Requirements
- Python >= 3.10, <= 3.12
- pip package manager
- At least 4GB RAM
- Internet connection

### Step 1: Clone or Download

Download the following files to your project directory:
- `tcn_quotex_bot.py` - Main trading bot
- `train_tcn_model.py` - Model training script
- `requirements.txt` - Dependencies

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install quotexpy tensorflow keras-tcn pandas numpy ta scikit-learn matplotlib
```

### Detailed Package Installation

```bash
# Core packages
pip install quotexpy==1.40.7        # Quotex API wrapper
pip install tensorflow>=2.13.0      # Deep learning framework
pip install keras-tcn>=2.9.3        # TCN implementation

# Data processing
pip install pandas>=2.0.0           # Data manipulation
pip install numpy>=1.24.0           # Numerical computing

# Technical analysis
pip install ta>=0.11.0              # Technical indicators library

# Machine learning utilities
pip install scikit-learn>=1.3.0     # ML utilities and metrics

# Visualization (optional, for training)
pip install matplotlib>=3.7.0       # Plotting
```

## 📁 Project Structure

```
quotex-tcn-bot/
│
├── tcn_quotex_bot.py              # Main trading bot
├── train_tcn_model.py             # Model training script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── models/                        # Trained models (created after training)
│   ├── tcn_quotex_model.h5       # Trained TCN model
│   ├── scaler.pkl                # Feature scaler
│   └── model_config.json         # Model configuration
│
├── data/                          # Historical data (user-provided)
│   └── historical_data.csv       # Your market data
│
└── logs/                          # Trading logs (created automatically)
    └── trades_YYYY-MM-DD.log     # Daily trade logs
```

## 🚀 Quick Start

### 1. Prepare Historical Data

Create a CSV file with historical candle data:

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.10050,1.10080,1.10040,1.10070,250
2024-01-01 00:01:00,1.10070,1.10090,1.10060,1.10085,300
...
```

**Minimum Requirements:**
- At least 10,000 candles (preferably 50,000+)
- 1-minute or 5-minute timeframe
- Consistent data format
- No missing values

**Data Sources:**
- MetaTrader 4/5 export
- TradingView export
- Historical data APIs (Alpha Vantage, Yahoo Finance)
- Quotex historical data (via API)

### 2. Train the Model

```bash
# Option A: Use sample data for testing
python train_tcn_model.py

# Option B: Use your own data (modify script first)
# Edit train_tcn_model.py, line ~300:
# df = pd.read_csv('data/your_historical_data.csv')
python train_tcn_model.py
```

**Training Output:**
- `tcn_quotex_model.h5` - Trained model weights
- `scaler.pkl` - Feature scaler for normalization
- `model_config.json` - Model configuration
- `training_history.png` - Training metrics plot

**Expected Training Time:**
- 10,000 candles: ~5-10 minutes (CPU) / 1-2 minutes (GPU)
- 50,000 candles: ~20-30 minutes (CPU) / 5-10 minutes (GPU)

### 3. Configure the Bot

Edit `tcn_quotex_bot.py` at the bottom of the file:

```python
# Configuration
EMAIL = "your_email@example.com"     # Your Quotex email
PASSWORD = "your_password"           # Your Quotex password
ASSET = "EURUSD_otc"                # Asset to trade
PRACTICE_MODE = True                # True = Practice, False = Real

# Trading parameters (in the class)
self.min_confidence = 0.60          # Minimum 60% confidence
self.trade_amount = 10              # $10 per trade
self.expiry_time = 60               # 60 seconds (1 minute)
```

### 4. Run the Bot

```bash
# Make sure model is trained first
python tcn_quotex_bot.py
```

**Bot will:**
1. Connect to Quotex platform
2. Load the trained TCN model
3. Fetch real-time candle data
4. Calculate technical indicators
5. Make predictions every 60 seconds
6. Place trades when confidence >= 60%
7. Track performance metrics

## 🎓 Model Training

### Training Process Explained

The training script performs these steps:

1. **Data Loading**: Loads historical OHLCV data
2. **Feature Engineering**: Adds 30+ technical indicators
3. **Sequence Creation**: Creates 50-candle sequences for TCN input
4. **Model Building**: Constructs TCN architecture with:
   - 64 filters
   - Kernel size 3
   - Dilations [1, 2, 4, 8, 16, 32]
   - Dropout 0.2-0.3
5. **Training**: Trains with Adam optimizer and early stopping
6. **Evaluation**: Tests on validation set
7. **Saving**: Saves model, scaler, and config

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

For production models, implement walk-forward validation:

```python
# In train_tcn_model.py
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train and evaluate
    model.train(X_train, y_train, X_val, y_val)
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
'''

# Save the README
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("✅ README.md created successfully!")
print(f"📄 File saved as: README.md")
print(f"📊 Total lines: {len(readme_content.splitlines())}")
