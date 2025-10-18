#!/bin/bash
# Monitor improved training progress

echo "╔════════════════════════════════════════════════════════╗"
echo "║      🚀 IMPROVED TCN TRAINING MONITOR                 ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Check if running
if ps aux | grep -q "[p]ython train_usdjpy"; then
    echo "✅ Status: TRAINING IN PROGRESS"
    echo ""
    
    # Process info
    PID=$(ps aux | grep "[p]ython train_usdjpy" | awk '{print $2}')
    ELAPSED=$(ps -p $PID -o etime= 2>/dev/null | tr -d ' ')
    CPU=$(ps aux | grep "[p]ython train_usdjpy" | awk '{print $3}')
    MEM=$(ps aux | grep "[p]ython train_usdjpy" | awk '{print $4}')
    
    echo "📊 Process Info:"
    echo "   PID: $PID"
    echo "   CPU: ${CPU}%"
    echo "   Memory: ${MEM}%"
    echo "   Running time: $ELAPSED"
else
    echo "❌ Status: NOT RUNNING"
    echo ""
    echo "Training completed or stopped. Check log:"
    echo "   tail -100 training_log_improved.txt"
    exit 1
fi

echo ""
echo "📈 Latest Epochs:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
grep "Epoch" training_log_improved.txt | tail -10
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
EPOCHS=$(grep -c "^Epoch" training_log_improved.txt)
echo "✓ Completed epochs: $EPOCHS"

echo ""
echo "💡 Commands:"
echo "   Live monitor: tail -f training_log_improved.txt"
echo "   Stop training: kill $PID"
echo "   Check status: ./monitor_improved_training.sh"
