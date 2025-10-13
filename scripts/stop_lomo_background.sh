#!/bin/bash
# scripts/stop_lomo_background.sh  
# Stop LOMO experiments running in background

LOG_DIR="logs/lomo"
PID_FILE="$LOG_DIR/lomo_background.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "❌ PID file not found: $PID_FILE"
    echo "🔍 LOMO experiments may not be running in background"
    exit 1
fi

LOMO_PID=$(cat "$PID_FILE")

echo "🛑 Stopping LOMO experiments..."
echo "🔢 Process ID: $LOMO_PID"

# Check if process is still running
if ps -p "$LOMO_PID" > /dev/null 2>&1; then
    echo "⏹️  Killing process $LOMO_PID..."
    kill "$LOMO_PID"
    
    # Wait a bit and check if it's really stopped
    sleep 2
    if ps -p "$LOMO_PID" > /dev/null 2>&1; then
        echo "💥 Force killing process $LOMO_PID..."
        kill -9 "$LOMO_PID"
    fi
    
    echo "✅ LOMO experiments stopped"
else
    echo "ℹ️  Process $LOMO_PID is not running"
fi

# Clean up PID file
rm "$PID_FILE"
echo "🧹 Cleaned up PID file"
echo ""
echo "📊 Check partial results in: experiments/lomo/"