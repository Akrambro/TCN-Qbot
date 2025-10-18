# 🚀 USDJPY TCN Training - Quick Start Guide

## ✅ Environment Status: READY!

All dependencies are installed and project structure is verified.

---

## 📂 Step 1: Place Your CSV File

Put your USDJPY 1-minute candle CSV file here:

```
/workspaces/TCN-Qbot/data/usdjpy_100k.csv
```

### CSV Format Requirements

Your CSV should have these columns (column names can vary):

| Required | Column Names (any variation works) |
|----------|-------------------------------------|
| ✅ Time  | `timestamp`, `date`, `time`, `datetime` |
| ✅ Open  | `open`, `o`, `Open` |
| ✅ High  | `high`, `h`, `High` |
| ✅ Low   | `low`, `l`, `Low` |
| ✅ Close | `close`, `c`, `Close` |
| ⚪ Volume | `volume`, `v`, `Volume` (optional) |

**Example CSV:**
```csv
timestamp,open,high,low,close,volume
2024-07-18 00:00:00,157.891,157.912,157.885,157.904,1234
2024-07-18 00:01:00,157.904,157.915,157.898,157.910,2345
...
```

---

## 🎯 Step 2: Train the Model

### Basic Training (recommended):
```bash
python train_usdjpy.py --data_path data/usdjpy_100k.csv
```

### Advanced Options:
```bash
# Custom split (default: 60k train, 30k val, 10k test)
python train_usdjpy.py --data_path data/usdjpy_100k.csv --train_size 70000 --val_size 20000

# More epochs for better accuracy
python train_usdjpy.py --data_path data/usdjpy_100k.csv --epochs 100

# Larger model for complex patterns
python train_usdjpy.py --data_path data/usdjpy_100k.csv --num_channels 32,32,16,16

# Combine options
python train_usdjpy.py \
  --data_path data/usdjpy_100k.csv \
  --epochs 100 \
  --batch_size 64 \
  --learning_rate 0.0005 \
  --num_channels 32,32,16,16
```

---

## 📊 Step 3: Monitor Training

### What to Expect:

**Training Time:**
- CPU: 20-45 minutes
- GPU: 5-15 minutes

**Progress Output:**
```
=== Epoch 1/50 ===
Train Loss: 0.6921 | Train Acc: 51.2%
Val Loss: 0.6915 | Val Acc: 52.1%
✓ Model improved! Saved checkpoint

=== Epoch 2/50 ===
Train Loss: 0.6890 | Train Acc: 52.8%
Val Loss: 0.6885 | Val Acc: 53.4%
✓ Model improved! Saved checkpoint

...

=== Epoch 25/50 ===
Train Loss: 0.6512 | Train Acc: 61.2%
Val Loss: 0.6587 | Val Acc: 58.5%
No improvement for 15 epochs - Early stopping!
```

**Good Signs:**
- ✅ Training accuracy > 55%
- ✅ Validation accuracy > 53%
- ✅ Loss decreasing steadily
- ✅ Val accuracy within 2-5% of train accuracy

**Warning Signs:**
- ⚠️ Val accuracy stuck at ~50% (random guessing)
- ⚠️ Large gap between train/val accuracy (overfitting)
- ⚠️ Loss not decreasing after 20 epochs

---

## 📈 Step 4: Review Results

### Output Files:

```
models/
  └── tcn_usdjpy.pt                    # Trained model checkpoint
  └── tcn_usdjpy_scaler.pkl            # Feature scaler
  └── tcn_usdjpy_config.json           # Model configuration

results/
  └── usdjpy_training_history.png      # Loss/accuracy plots
  └── usdjpy_backtest_results.png      # Trading simulation
  └── usdjpy_backtest_summary.json     # Performance metrics

logs/
  └── training_YYYYMMDD_HHMMSS.log     # Detailed training log
```

### Performance Metrics:

The backtest will show:
- **Win Rate**: Target 55-60% (>50% is profitable)
- **Profit Factor**: Target >1.5 (1.0 = break-even)
- **Sharpe Ratio**: Target >1.0 (risk-adjusted returns)
- **Max Drawdown**: Keep <20%

---

## 🎓 Understanding Your Results

### Example Good Results:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Backtest Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Trades:     423
Winning Trades:   245 (57.9%)  ← Good!
Losing Trades:    178 (42.1%)

Total P&L:        $2,150      ← Profitable!
Profit Factor:    1.68        ← Excellent!
Sharpe Ratio:     1.24        ← Good risk-adjusted return
Max Drawdown:     -15.2%      ← Acceptable risk
```

### What Accuracy Means:
- **50%** = Random (coin flip) - No edge
- **52-54%** = Small edge - Potentially profitable
- **55-58%** = Good edge - Should be profitable
- **>60%** = Excellent - Rare, verify no overfitting

**Remember**: With 80% payout binary options, you need >55.6% win rate to break even!

---

## 🔧 Troubleshooting

### Issue: "File not found"
```bash
# Check file exists
ls -lh data/usdjpy_100k.csv

# Check file format
head data/usdjpy_100k.csv
```

### Issue: "Not enough data"
```bash
# Count rows
wc -l data/usdjpy_100k.csv

# Should show ~100,000+ rows
```

### Issue: Low accuracy (<52%)
Try these adjustments:
```bash
# 1. More training data
python train_usdjpy.py --train_size 70000 --val_size 20000

# 2. Larger model
python train_usdjpy.py --num_channels 32,32,16,16

# 3. More epochs
python train_usdjpy.py --epochs 100

# 4. Lower learning rate
python train_usdjpy.py --learning_rate 0.0001
```

### Issue: Overfitting (train >> val accuracy)
```bash
# Increase dropout
python train_usdjpy.py --dropout 0.3

# Smaller model
python train_usdjpy.py --num_channels 8,8

# More validation data
python train_usdjpy.py --train_size 50000 --val_size 40000
```

---

## 🎯 Next Steps After Training

### 1. Analyze Feature Importance
Check which technical indicators are most useful

### 2. Optimize Confidence Threshold
Find the sweet spot between trade frequency and accuracy

### 3. Test Different Timeframes
Try 5-minute or 15-minute candles

### 4. Walk-Forward Validation
Test on completely unseen future data

### 5. Paper Trading
Test on demo account before going live

---

## ⚠️ Important Warnings

### Before Live Trading:
- ✅ Achieve >55% accuracy on test set
- ✅ Backtest shows consistent profit factor >1.5
- ✅ Test on paper/demo account for 1-2 weeks
- ✅ Understand that past performance ≠ future results
- ✅ Never risk more than 1-2% per trade
- ✅ Check Quotex terms of service regarding bots

### Risk Disclosure:
- Binary options trading is HIGH RISK
- Most traders lose money
- This bot is for EDUCATIONAL purposes
- No guarantee of profits
- Use at your own risk

---

## 📞 Need Help?

If you encounter issues:

1. Check the detailed logs: `logs/training_*.log`
2. Review error messages carefully
3. Verify CSV format matches requirements
4. Try with sample data first: `python examples/quick_start.py`

---

## 🚀 Ready to Start?

Copy your CSV file and run:

```bash
# Copy your file
cp /path/to/your/usdjpy_data.csv data/usdjpy_100k.csv

# Start training
python train_usdjpy.py --data_path data/usdjpy_100k.csv

# Grab coffee ☕ and wait 20-45 minutes
```

Good luck! 🎲📈
