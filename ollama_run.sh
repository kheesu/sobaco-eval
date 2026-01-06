#!/bin/bash

# Full evaluation run for SOBACO-EVAL
# Runs all models on all datasets

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "SOBACO-EVAL: Full Evaluation Run"
echo "=========================================="

# Define all models
LOCAL_MODELS=(
    "qwen-3-30b-inst"
)

API_MODELS=(
)

# Define all datasets
DATASETS=(
    "csv/ja-en_dataset.csv"
    "csv/zh-en_dataset.csv"
    "csv/ko-en-v2_dataset.csv"
    "csv/zh-ko_dataset.csv"
)

# Configuration
BATCH_SIZE=16  # Adjust based on your GPU memory
USE_ASYNC_API=true  # Use async for API models
MAX_CONCURRENT=10  # Max concurrent API requests

# Total count
TOTAL_MODELS=$((${#LOCAL_MODELS[@]} + ${#API_MODELS[@]}))
TOTAL_DATASETS=${#DATASETS[@]}
TOTAL_RUNS=$((TOTAL_MODELS * TOTAL_DATASETS))

echo "Total Models: $TOTAL_MODELS (${#LOCAL_MODELS[@]} local + ${#API_MODELS[@]} API)"
echo "Total Datasets: $TOTAL_DATASETS"
echo "Total Runs: $TOTAL_RUNS"
echo "=========================================="

# Counter for progress
CURRENT_RUN=0

# Function to run evaluation
run_evaluation() {
    local model=$1
    local is_api=$2
    local dataset=$3
    
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    echo ""
    echo -e "${GREEN}[${CURRENT_RUN}/${TOTAL_RUNS}] Evaluating: $model${NC} on $dataset"
    echo "=========================================="
    
    # Build command
    CMD="python3 evaluate.py --model $model --dataset $dataset --use-ollama"
    
    if [ "$is_api" = true ]; then
        if [ "$USE_ASYNC_API" = true ]; then
            CMD="$CMD --async-api --max-concurrent $MAX_CONCURRENT"
        fi
    else
        CMD="$CMD --batch-size $BATCH_SIZE"
    fi
    
    echo "Running: $CMD"
    
    # Run evaluation
    if eval $CMD; then
        echo -e "${GREEN}✓ Success: $model${NC}"
    else
        echo -e "${RED}✗ Failed: $model${NC}"
        echo "Continuing with next model..."
    fi
}

# Start time
START_TIME=$(date +%s)

# Run local models
echo ""
echo "=========================================="
echo "Starting Local Model Evaluations"
echo "=========================================="

for model in "${LOCAL_MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        run_evaluation "$model" false "$dataset"
    done
done

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Summary
echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Total runs: $TOTAL_RUNS"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved in: results/"
echo "=========================================="
