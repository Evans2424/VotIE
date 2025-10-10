#!/bin/bash
# Run grid search in background with nohup

# Activate virtual environment (adjust path if needed)
source ../../../venv/bin/activate

# Run grid search in background
nohup python grid_search.py > grid_search.log 2>&1 &

# Get process ID
PID=$!

echo "Grid search started with PID: $PID"
echo "Monitor progress with: tail -f grid_search.log"
echo "Check detailed logs: tail -f grid_search_detailed.log"
echo "View results: cat grid_search_results.json"
echo "Kill process with: kill $PID"

# Save PID to file for later reference
echo $PID > grid_search.pid
echo "PID saved to grid_search.pid"
