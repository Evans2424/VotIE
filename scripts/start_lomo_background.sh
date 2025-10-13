#!/bin/bash
# scripts/start_lomo_background.sh
# Wrapper script to run LOMO experiments in background with nohup

set -e

cd "$(dirname "$0")/.."  # Change to project root

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/lomo"
mkdir -p "$LOG_DIR"

BACKGROUND_LOG="$LOG_DIR/lomo_background_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/lomo_background.pid"

echo "🚀 Starting LOMO experiments in background..."
echo "📝 Background log: $BACKGROUND_LOG"
echo "🔢 PID file: $PID_FILE"
echo ""

# Remove old PID file if exists
[ -f "$PID_FILE" ] && rm "$PID_FILE"

# Start LOMO experiments in background
nohup bash scripts/run_lomo.sh > "$BACKGROUND_LOG" 2>&1 &

# Save PID
echo $! > "$PID_FILE"
LOMO_PID=$!

echo "✅ LOMO experiments started in background"
echo "🔢 Process ID: $LOMO_PID"
echo "📝 Monitor progress with: tail -f $BACKGROUND_LOG"
echo "🛑 Stop with: kill $LOMO_PID"
echo ""
echo "📋 Useful commands:"
echo "  Check status:    ps -p $LOMO_PID"
echo "  Follow logs:     tail -f $BACKGROUND_LOG"
echo "  Check results:   ls -la experiments/lomo/"
echo "  Kill process:    kill $LOMO_PID && rm $PID_FILE"
echo ""
echo "🎯 The process will continue running even if you close this terminal."
echo "📊 Results will be saved to: experiments/lomo/"