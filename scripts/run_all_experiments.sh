#!/bin/bash

# VotIE Main Experiment Pipeline Runner
# Runs all configurations in configs/main_experiment across 4 GPUs
# Each job runs detached with nohup

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOTIE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$VOTIE_DIR/configs/main_experiment"
LOG_DIR="$VOTIE_DIR/logs/pipeline_runs"
NUM_GPUS=4

# Create log directory
mkdir -p "$LOG_DIR"

# Get current timestamp for this batch
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "🚀 Starting VotIE pipeline batch run at $BATCH_TIMESTAMP"
echo "📁 Working directory: $VOTIE_DIR"
echo "📋 Config directory: $CONFIG_DIR"
echo "📄 Logs directory: $LOG_DIR"
echo "🔢 Using $NUM_GPUS GPUs (0-3)"

# List of all configurations
CONFIGS=(
    "bert_crf.yaml"
    "bert_linear.yaml" 
    "bilstm_fasttext.yaml"
    "crf.yaml"
    "deberta_crf.yaml"
    "deberta_linear.yaml"
    "xlmr_crf.yaml"
    "xlmr_linear.yaml"
)

# Function to get GPU assignment based on job index
get_gpu_id() {
    local job_index=$1
    echo $((job_index % NUM_GPUS))
}

# Function to get experiment name from config file
get_experiment_name() {
    local config_file=$1
    local basename=$(basename "$config_file" .yaml)
    echo "${basename}_${BATCH_TIMESTAMP}"
}

# Function to run a single pipeline job
run_pipeline_job() {
    local config_file=$1
    local gpu_id=$2
    local experiment_name=$3
    local job_index=$4
    
    local config_path="$CONFIG_DIR/$config_file"
    local log_file="$LOG_DIR/${experiment_name}.log"
    local error_file="$LOG_DIR/${experiment_name}.error"
    
    echo "🎯 Starting job $((job_index+1))/${#CONFIGS[@]}: $experiment_name on GPU $gpu_id"
    echo "   Config: $config_file"
    echo "   Log: $log_file"
    
    # Set CUDA_VISIBLE_DEVICES and run pipeline
    nohup bash -c "
        cd '$VOTIE_DIR'
        export CUDA_VISIBLE_DEVICES=$gpu_id
        echo 'Job started at: \$(date)' > '$log_file'
        echo 'GPU ID: $gpu_id' >> '$log_file'
        echo 'Config: $config_file' >> '$log_file'
        echo 'Experiment: $experiment_name' >> '$log_file'
        echo '===========================================' >> '$log_file'
        
        python scripts/run_pipeline.py \\
            --config '$config_path' \\
            --name '$experiment_name' \\
            --force \\
            >> '$log_file' 2>> '$error_file'
        
        exit_code=\$?
        echo '===========================================' >> '$log_file'
        echo 'Job finished at: \$(date)' >> '$log_file'
        echo 'Exit code: '\$exit_code >> '$log_file'
        
        if [ \$exit_code -eq 0 ]; then
            echo '✅ SUCCESS: $experiment_name' >> '$log_file'
        else
            echo '❌ FAILED: $experiment_name' >> '$log_file'
        fi
    " &
    
    # Store the background job PID
    local job_pid=$!
    echo "   PID: $job_pid"
    echo "$job_pid" >> "$LOG_DIR/job_pids_${BATCH_TIMESTAMP}.txt"
    
    return 0
}

# Main execution
echo ""
echo "📋 Configuration Summary:"
for i in "${!CONFIGS[@]}"; do
    config_file="${CONFIGS[$i]}"
    gpu_id=$(get_gpu_id $i)
    experiment_name=$(get_experiment_name "$config_file")
    echo "   Job $((i+1)): $config_file -> GPU $gpu_id -> $experiment_name"
done

echo ""
read -p "🤔 Continue with pipeline execution? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled by user"
    exit 1
fi

echo ""
echo "🚀 Launching pipeline jobs..."

# Launch all jobs
for i in "${!CONFIGS[@]}"; do
    config_file="${CONFIGS[$i]}"
    gpu_id=$(get_gpu_id $i)
    experiment_name=$(get_experiment_name "$config_file")
    
    run_pipeline_job "$config_file" "$gpu_id" "$experiment_name" "$i"
    
    # Small delay between job launches to avoid race conditions
    sleep 2
done

echo ""
echo "🎉 All jobs launched successfully!"
echo "📄 Job PIDs saved to: $LOG_DIR/job_pids_${BATCH_TIMESTAMP}.txt"
echo "📋 Monitor logs in: $LOG_DIR/"
echo ""
echo "🔍 Useful commands:"
echo "   # Check job status:"
echo "   jobs"
echo "   ps aux | grep run_pipeline"
echo ""
echo "   # Monitor logs:"
echo "   tail -f $LOG_DIR/*.log"
echo "   ls -la $LOG_DIR/"
echo ""
echo "   # Kill all jobs if needed:"
echo "   while read pid; do kill \$pid 2>/dev/null || true; done < $LOG_DIR/job_pids_${BATCH_TIMESTAMP}.txt"
echo ""
echo "   # Check GPU usage:"
echo "   nvidia-smi"
echo "   watch -n 1 nvidia-smi"
echo ""

# Create a summary script for this batch
SUMMARY_SCRIPT="$LOG_DIR/check_batch_${BATCH_TIMESTAMP}.sh"
cat > "$SUMMARY_SCRIPT" << 'EOF'
#!/bin/bash

# Batch summary checker
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_TIMESTAMP="$(basename "$0" .sh | sed 's/check_batch_//')"
PID_FILE="$SCRIPT_DIR/job_pids_${BATCH_TIMESTAMP}.txt"

echo "🔍 VotIE Pipeline Batch Status Check"
echo "📅 Batch: $BATCH_TIMESTAMP"
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo "❌ PID file not found: $PID_FILE"
    exit 1
fi

echo "📋 Job Status:"
job_count=0
running_count=0
finished_count=0

while read pid; do
    ((job_count++))
    if kill -0 "$pid" 2>/dev/null; then
        echo "   Job $job_count (PID $pid): 🟡 RUNNING"
        ((running_count++))
    else
        echo "   Job $job_count (PID $pid): ✅ FINISHED"
        ((finished_count++))
    fi
done < "$PID_FILE"

echo ""
echo "📊 Summary: $running_count running, $finished_count finished, $job_count total"

echo ""
echo "📄 Recent log activity:"
find "$SCRIPT_DIR" -name "*.log" -newer "$PID_FILE" -exec tail -n 3 {} +

echo ""
echo "🔍 To check individual logs:"
echo "   ls -la $SCRIPT_DIR/*.log"
echo "   tail -f $SCRIPT_DIR/<experiment_name>.log"

EOF

chmod +x "$SUMMARY_SCRIPT"
echo "📊 Batch summary script created: $SUMMARY_SCRIPT"
echo "   Run it anytime to check job status: $SUMMARY_SCRIPT"

exit 0