# USDJPY Training Guide

## Your Data
- **Pair**: USDJPY
- **Timeframe**: 1-minute candles
- **Total Bars**: 100,000
- **Period**: Last 3 months

## Training Split
- **Training**: 60,000 bars (60%)
- **Validation**: 30,000 bars (30%)
- **Testing**: 10,000 bars (10%)

---

## Quick Start

### Step 1: Place Your CSV File

Put your USDJPY CSV file in the `data/` directory:
```bash
cp /path/to/your/usdjpy.csv /workspaces/TCN-Qbot/data/usdjpy_100k.csv
```

### Step 2: Verify CSV Format

Your CSV should have these columns (any of these variations work):
```
timestamp,open,high,low,close,volume
# OR
date,Open,High,Low,Close,Volume
# OR
time,o,h,l,c,v
```

**Minimum required**: `open`, `high`, `low`, `close`

Example row:
```
2024-07-01 00:00:00,161.234,161.245,161.220,161.238,1000
```

### Step 3: Run Training

```bash
python train_usdjpy.py --data_path data/usdjpy_100k.csv
```

That's it! The script will:
1. ✅ Load and validate your data
2. ✅ Split into 60k/30k/10k
3. ✅ Generate 30+ technical indicators
4. ✅ Train TCN model for up to 50 epochs
5. ✅ Evaluate on test set
6. ✅ Run backtest simulation
7. ✅ Save model to `models/tcn_usdjpy.pt`

---

## Expected Output

### Console Output
```
================================================================================
TCN USDJPY TRAINING PIPELINE
================================================================================
Loading data from: data/usdjpy_100k.csv
Loaded 100000 rows

Data Summary:
  Total bars: 100000
  Date range: 2024-07-01 to 2024-10-01

================================================================================
SPLITTING DATA
================================================================================
Training: 60000 bars (60.0%)
Validation: 30000 bars (30.0%)
Testing: 10000 bars (10.0%)

================================================================================
PREPROCESSING DATA
================================================================================
Computing technical indicators...
Training features shape: (59950, 50, 28)
Number of features: 28

================================================================================
TRAINING TCN MODEL
================================================================================
Model architecture:
  Input channels: 28
  Hidden channels: [64, 64, 32, 32]
  Receptive field: 31
  Total parameters: 123,456

Epoch 1/50 - Train Loss: 0.6932, Val Loss: 0.6928
Epoch 2/50 - Train Loss: 0.6920, Val Loss: 0.6915
...

================================================================================
EVALUATING MODEL
================================================================================
Test Set Performance:
  Overall Accuracy: 54.32%
  Up Accuracy: 56.12%
  Down Accuracy: 52.45%

High Confidence Trades (>60% or <40%):
  Count: 4523 / 9950 (45.5%)
  Accuracy: 58.76%

================================================================================
RUNNING BACKTEST
================================================================================
Confidence threshold: 0.6
Buy signals: 2245
Sell signals: 2278
No trade: 5427

Backtest Results:
  Total Trades: 4523
  Win Rate: 58.76%
  Total Profit: $1,234.56
  Final Balance: $11,234.56
  Return: 12.35%
  Profit Factor: 1.45
  Max Drawdown: -8.23%

Results saved to results/
```

### Files Created
```
models/
  ├── tcn_usdjpy.pt              # Trained model
  └── tcn_usdjpy_preprocessor.pkl # Feature scaler

results/
  ├── usdjpy_backtest.json        # Detailed results
  └── usdjpy_backtest.png         # Performance charts

logs/
  └── training_YYYYMMDD_HHMMSS.log
```

---

## Advanced Options

### Custom Split Sizes
```bash
python train_usdjpy.py \
    --data_path data/usdjpy_100k.csv \
    --train_size 70000 \
    --val_size 20000
```

### Different Sequence Length
```bash
python train_usdjpy.py \
    --data_path data/usdjpy_100k.csv \
    --sequence_length 100  # Look back 100 candles instead of 50
```

### Custom Model Path
```bash
python train_usdjpy.py \
    --data_path data/usdjpy_100k.csv \
    --model_path models/my_custom_model.pt
```

---

## What Happens Under the Hood

### 1. Data Loading
- Auto-detects column names (case-insensitive)
- Handles missing volume column
- Sorts by timestamp
- Validates data quality

### 2. Feature Engineering (28 features)
**Trend Indicators:**
- EMA (9, 21 periods)
- SMA (50 periods)
- MACD (12, 26, 9)

**Momentum:**
- RSI (14 periods)
- Stochastic Oscillator (14 periods)

**Volatility:**
- Bollinger Bands (20, 2σ)
- ATR (14 periods)

**Price Features:**
- Log returns
- High-low range
- Lagged features (1, 2, 3, 5 periods)

**Time Features:**
- Hour of day
- Day of week
- Trading session (Asian/European/US)

### 3. TCN Model Architecture
```
Input: (batch, 50, 28)  # 50 candles, 28 features
  ↓
Temporal Block 1 (dilation=1, channels=64)
  ↓
Temporal Block 2 (dilation=2, channels=64)
  ↓
Temporal Block 3 (dilation=4, channels=32)
  ↓
Temporal Block 4 (dilation=8, channels=32)
  ↓
Global Average Pool
  ↓
Dense Layer → Sigmoid
  ↓
Output: Probability [0, 1]
```

### 4. Training Process
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Cross-Entropy
- **Batch Size**: 128
- **Early Stopping**: Patience 10 epochs
- **Device**: Auto-detect GPU/CPU

### 5. Evaluation Metrics
- Overall accuracy
- Directional accuracy (up/down separately)
- Confidence-filtered accuracy
- Backtest performance (win rate, profit factor, drawdown)

---

## Troubleshooting

### Issue: "Missing required columns"
**Solution**: Check your CSV format. Rename columns to: `timestamp, open, high, low, close, volume`

### Issue: "Not enough data after preprocessing"
**Solution**: Your CSV has less than 100k rows, or reduce `--train_size` and `--val_size`

### Issue: Training is slow
**Solution**: 
- Reduce batch size: Add `batch_size=64` in code
- Use GPU if available
- Reduce sequence length: `--sequence_length 30`

### Issue: Accuracy is ~50% (random)
**Solutions**:
1. **More data**: Collect 6+ months of data
2. **Different features**: Experiment with indicators
3. **Hyperparameter tuning**: Adjust channels, dropout
4. **Ensemble**: Train multiple models and average predictions

---

## Next Steps

### 1. Analyze Results
```bash
# View training logs
cat logs/training_*.log

# Check backtest visualizations
open results/usdjpy_backtest.png
```

### 2. Improve Model
Try these experiments:
- Add more layers: `num_channels=[128, 128, 64, 64, 32]`
- Increase dropout: `dropout=0.3`
- Longer sequences: `--sequence_length 100`
- Different train/val split: `--train_size 80000 --val_size 15000`

### 3. Deploy for Live Trading
```bash
# Load trained model
from src.tcn_model import TCNForex, TCNTrainer
import torch
import pickle

# Load model
model = TCNForex(input_channels=28, num_channels=[64, 64, 32, 32])
model.load_state_dict(torch.load('models/tcn_usdjpy.pt'))

# Load preprocessor
with open('models/tcn_usdjpy_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Make predictions on new data
# ... (see tcn_quotex_bot.py for full integration)
```

---

## Performance Expectations

### With 100k bars of quality data:

**Realistic Training Results:**
- Training accuracy: 58-65%
- Validation accuracy: 54-60%
- Test accuracy: 52-58%

**Live Trading (expect lower):**
- Real accuracy: 50-56%
- With 80% payout, break-even at 55.6%
- Target 56-58% for profitability

**Key Factors:**
- ✅ Data quality matters most
- ✅ Market regime changes affect performance
- ✅ Retrain monthly with fresh data
- ✅ Use confidence thresholds (>60%)
- ✅ Risk management is essential

---

## Questions?

- Check `HOW_TO_PROCEED.md` for general guidance
- See `examples/quick_start.py` for code examples
- Read `src/tcn_model.py` for model details
- Review `src/data_preprocessing.py` for features

**Ready to train?** 
```bash
python train_usdjpy.py --data_path data/usdjpy_100k.csv
```
