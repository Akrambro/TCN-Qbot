#!/usr/bin/env python3
"""
Evaluate the trained TCN model on test data
"""

import torch
import numpy as np
import pickle
import pandas as pd
from src.tcn_model import TCNForex, TCNTrainer
from src.data_preprocessing import ForexDataPreprocessor

print("="*80)
print("TCN MODEL EVALUATION")
print("="*80)

# Load data
print("\n📊 Loading test data...")
df = pd.read_csv('data/usdjpy_100k.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Split data
train_size = 60000
val_size = 30000
test_df = df.iloc[train_size + val_size:].copy()

print(f"Test data: {len(test_df)} bars")

# Load preprocessor
print("\n🔧 Loading preprocessor...")
with open('models/tcn_usdjpy_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Preprocess test data
print("⚙️  Preprocessing test data...")
X_test, y_test, feature_cols = preprocessor.full_pipeline(test_df, sequence_length=50, fit=False)

print(f"Test samples: {len(X_test)}")
print(f"Features: {X_test.shape[1]}")
print(f"Sequence length: {X_test.shape[2]}")

# Load model
print("\n🤖 Loading trained model...")
model = TCNForex(
    input_channels=X_test.shape[1],
    num_channels=[128, 128, 64, 64],  # Match the improved model!
    kernel_size=3,
    dropout=0.5  # Match training dropout
)

model.load_state_dict(torch.load('models/tcn_usdjpy.pt'))
trainer = TCNTrainer(model=model, device='cpu')

# Make predictions
print("\n📈 Making predictions...")
X_test_tensor = torch.FloatTensor(X_test)
predictions = trainer.predict(X_test_tensor)

pred_probs = predictions
pred_labels = (predictions > 0.5).astype(int)

# Calculate metrics
from sklearn.metrics import classification_report, confusion_matrix

accuracy = (pred_labels == y_test).mean()
up_accuracy = (pred_labels[y_test == 1] == 1).mean() if (y_test == 1).sum() > 0 else 0
down_accuracy = (pred_labels[y_test == 0] == 0).mean() if (y_test == 0).sum() > 0 else 0

print("\n" + "="*80)
print("📊 TEST SET RESULTS")
print("="*80)
print(f"\n✅ Overall Accuracy: {accuracy*100:.2f}%")
print(f"📈 Up Prediction Accuracy: {up_accuracy*100:.2f}%")
print(f"📉 Down Prediction Accuracy: {down_accuracy*100:.2f}%")

# Distribution of predictions (BIAS CHECK)
n_pred_up = (pred_labels == 1).sum()
n_pred_down = (pred_labels == 0).sum()
n_actual_up = (y_test == 1).sum()
n_actual_down = (y_test == 0).sum()

print(f"\n📊 Prediction Distribution (BIAS CHECK):")
print(f"   Predicted UP: {n_pred_up} ({n_pred_up/len(y_test)*100:.1f}%)")
print(f"   Predicted DOWN: {n_pred_down} ({n_pred_down/len(y_test)*100:.1f}%)")
print(f"   Actual UP: {n_actual_up} ({n_actual_up/len(y_test)*100:.1f}%)")
print(f"   Actual DOWN: {n_actual_down} ({n_actual_down/len(y_test)*100:.1f}%)")

# Check for bias
pred_up_pct = n_pred_up / len(y_test) * 100
if pred_up_pct < 30 or pred_up_pct > 70:
    print(f"\n⚠️  WARNING: Model shows BIAS! Predicting {pred_up_pct:.1f}% UP")
else:
    print(f"\n✅ Predictions are BALANCED ({pred_up_pct:.1f}% UP, {100-pred_up_pct:.1f}% DOWN)")

# Detailed classification report
print("\n" + "="*80)
print("PER-CLASS METRICS:")
print("="*80)
report = classification_report(y_test, pred_labels, target_names=['DOWN', 'UP'], digits=4)
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, pred_labels)
print("\nConfusion Matrix:")
print(f"              Predicted")
print(f"              DOWN    UP")
print(f"Actual DOWN   {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"Actual UP     {cm[1,0]:4d}  {cm[1,1]:4d}")

# Confidence analysis
high_conf_mask = (pred_probs > 0.6) | (pred_probs < 0.4)
if high_conf_mask.sum() > 0:
    high_conf_acc = (pred_labels[high_conf_mask] == y_test[high_conf_mask]).mean()
    high_conf_count = high_conf_mask.sum()

    print(f"\n🎯 High Confidence Predictions (>60% or <40%):")
    print(f"   Count: {high_conf_count}/{len(predictions)} ({high_conf_count/len(predictions)*100:.1f}%)")
    print(f"   Accuracy: {high_conf_acc*100:.2f}%")

# Very high confidence
very_high_conf_mask = (pred_probs > 0.7) | (pred_probs < 0.3)
if very_high_conf_mask.sum() > 0:
    very_high_conf_acc = (pred_labels[very_high_conf_mask] == y_test[very_high_conf_mask]).mean()
    very_high_conf_count = very_high_conf_mask.sum()
    print(f"\n🔥 Very High Confidence (>70% or <30%):")
    print(f"   Count: {very_high_conf_count}/{len(predictions)} ({very_high_conf_count/len(predictions)*100:.1f}%)")
    print(f"   Accuracy: {very_high_conf_acc*100:.2f}%")

# Trading simulation
print("\n" + "="*80)
print("💰 SIMPLE TRADING SIMULATION")
print("="*80)

balance = 1000
trade_amount = 10
payout_rate = 0.80
wins = 0
losses = 0

for i, (pred, actual) in enumerate(zip(pred_labels, y_test)):
    if pred == actual:
        balance += trade_amount * payout_rate
        wins += 1
    else:
        balance -= trade_amount
        losses += 1

win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
profit = balance - 1000

print(f"\n📈 Trading Results:")
print(f"   Starting Balance: $1000")
print(f"   Final Balance: ${balance:.2f}")
print(f"   Profit/Loss: ${profit:.2f} ({profit/1000*100:.1f}%)")
print(f"\n   Total Trades: {wins + losses}")
print(f"   Wins: {wins} ({win_rate*100:.1f}%)")
print(f"   Losses: {losses} ({(1-win_rate)*100:.1f}%)")
print(f"   Win Rate: {win_rate*100:.2f}%")

# Break-even analysis
breakeven_rate = 1 / (1 + payout_rate)
print(f"\n💡 Analysis:")
print(f"   Break-even win rate: {breakeven_rate*100:.2f}%")
if win_rate > breakeven_rate:
    print(f"   ✅ Model is PROFITABLE! ({win_rate*100:.2f}% > {breakeven_rate*100:.2f}%)")
else:
    print(f"   ❌ Model needs improvement ({win_rate*100:.2f}% ≤ {breakeven_rate*100:.2f}%)")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)
