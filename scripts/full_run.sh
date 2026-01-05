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
    "llama-3.1-8b-inst"
    "llama-3.1-70b-inst"
    "qwen-3-4b-inst"
    "hyperclovax"
    "hyperclovax-omni"
    "swallow-3.1-8b-inst"
    "swallow-3.1-70b-inst"
)

API_MODELS=(
    "gpt-5.1"
)

# Define all datasets
DATASETS=(
    "csv/ja-ja_dataset.csv"
    "csv/ja-ko_dataset.csv"
    "csv/ja-zh_dataset.csv"
    "csv/ko-ja-v2_dataset.csv"
    "csv/ko-ko-v2_dataset.csv"
    "csv/ko-zh-v2_dataset.csv"
    "csv/zh-ja_dataset.csv"
    "csv/zh-ko_dataset.csv"
    "csv/zh-zh_dataset.csv"
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
    
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    echo ""
    echo -e "${GREEN}[${CURRENT_RUN}/${TOTAL_RUNS}] Evaluating: $model${NC}"
    echo "=========================================="
    
    # Build command
    CMD="python evaluate.py --model $model --all-datasets"
    
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
    run_evaluation "$model" false
done

# Run API models
echo ""
echo "=========================================="
echo "Starting API Model Evaluations"
echo "=========================================="

for model in "${API_MODELS[@]}"; do
    run_evaluation "$model" true
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
