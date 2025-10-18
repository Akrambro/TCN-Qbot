#!/bin/bash
# Quick training status check script

echo "╔════════════════════════════════════════════════════════╗"
echo "║         TCN TRAINING STATUS CHECK                     ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Check if process is running
if ps aux | grep -q "[p]ython train_usdjpy"; then
    echo "✅ Status: RUNNING"
    
    # Get process info
    echo ""
    echo "📊 Process Info:"
    ps aux | grep "[p]ython train_usdjpy" | awk '{printf "   PID: %s\n   CPU: %s%%\n   MEM: %s%%\n   TIME: %s\n", $2, $3, $4, $10}'
    
    # Get elapsed time
    PID=$(ps aux | grep "[p]ython train_usdjpy" | awk '{print $2}')
    ELAPSED=$(ps -p $PID -o etime= 2>/dev/null)
    echo "   Elapsed: $ELAPSED"
    
else
    echo "❌ Status: NOT RUNNING"
    echo ""
    echo "Training may have completed or stopped."
    echo "Check the log file for results:"
    echo "   tail -50 training_log.txt"
    exit 1
fi

echo ""
echo "📋 Latest Log Output:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -n 10 training_log.txt
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "🔍 Log file size: $(wc -l < training_log.txt) lines"

echo ""
echo "💡 To monitor live:"
echo "   tail -f training_log.txt"
echo ""
echo "⏰ Check again in 5-10 minutes for epoch progress"
