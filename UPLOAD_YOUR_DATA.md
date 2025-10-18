# 📤 How to Upload Your USDJPY CSV Data

## I'm NOT stuck - Environment is READY! ✅

All dependencies are installed and the project is fully configured.

---

## 📍 **WHERE TO PUT YOUR CSV FILE**

```
/workspaces/TCN-Qbot/data/usdjpy_100k.csv
```

### Step-by-Step:

1. **Locate your USDJPY CSV file** (100,000 bars of 1-minute data)

2. **Upload it to this location**:
   - If using VS Code: Drag & drop into the `data/` folder
   - If using command line: `cp /path/to/your/file.csv data/usdjpy_100k.csv`
   - If using file browser: Place in `/workspaces/TCN-Qbot/data/` directory

3. **Verify upload**:
   ```bash
   ls -lh data/usdjpy_100k.csv
   ```

---

## 📋 **CSV FORMAT REQUIREMENTS**

Your CSV must have these columns (names can vary):

### Required Columns:
| Column | Accepted Names |
|--------|----------------|
| Time | `timestamp`, `date`, `time`, `datetime` |
| Open | `open`, `o`, `Open` |
| High | `high`, `h`, `High` |
| Low | `low`, `l`, `Low` |
| Close | `close`, `c`, `Close` |

### Optional:
- Volume: `volume`, `v`, `Volume`

### Example Format:
```csv
timestamp,open,high,low,close,volume
2024-07-18 00:00:00,157.891,157.912,157.885,157.904,1234
2024-07-18 00:01:00,157.904,157.915,157.898,157.910,2345
2024-07-18 00:02:00,157.910,157.925,157.908,157.920,3456
...
```

---

## ✅ **VALIDATE YOUR CSV** (Optional but Recommended)

Before training, check if your CSV is properly formatted:

```bash
python scripts/check_csv.py data/usdjpy_100k.csv
```

This will show:
- ✅ File size and row count
- ✅ Column names detected
- ✅ Data quality checks
- ✅ Recommended data split
- ✅ Sample rows

---

## 🚀 **TRAIN THE MODEL**

Once your CSV is uploaded and validated:

### Basic Training:
```bash
python train_usdjpy.py --data_path data/usdjpy_100k.csv
```

### With Custom Settings:
```bash
# More epochs for better accuracy
python train_usdjpy.py --data_path data/usdjpy_100k.csv --epochs 100

# Custom data split
python train_usdjpy.py --data_path data/usdjpy_100k.csv \
  --train_size 70000 --val_size 20000
```

---

## ⏱️ **TRAINING TIME**

- **On CPU**: 20-45 minutes
- **On GPU**: 5-15 minutes

You'll see progress updates like:
```
Epoch 1/50: Train Loss=0.6921, Val Loss=0.6915, Val Acc=52.1%
Epoch 2/50: Train Loss=0.6890, Val Loss=0.6885, Val Acc=53.4%
...
```

---

## 📊 **RESULTS LOCATION**

After training completes, check:

```
models/
  └── tcn_usdjpy.pt              # Your trained model

results/
  └── usdjpy_training_history.png
  └── usdjpy_backtest_results.png
  └── usdjpy_backtest_summary.json

logs/
  └── training_*.log
```

---

## 🎯 **WHAT YOU'LL GET**

1. **Trained TCN Model** - Ready to make predictions
2. **Performance Metrics** - Win rate, profit factor, Sharpe ratio
3. **Visual Charts** - Training curves and backtest results
4. **Detailed Logs** - Complete training history

---

## 📞 **QUICK REFERENCE**

```bash
# 1. Upload CSV
cp /path/to/usdjpy.csv data/usdjpy_100k.csv

# 2. Validate (optional)
python scripts/check_csv.py data/usdjpy_100k.csv

# 3. Train
python train_usdjpy.py --data_path data/usdjpy_100k.csv

# 4. View results
ls -lh models/
ls -lh results/
```

---

## 🆘 **NEED HELP?**

- **Detailed Guide**: See `QUICK_START.md`
- **Training Guide**: See `USDJPY_TRAINING_GUIDE.md`
- **Checklist**: See `CHECKLIST.md`

---

**I'm ready and waiting for your CSV file!** 🚀📊

