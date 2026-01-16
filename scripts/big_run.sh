#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "SOBACO-EVAL: 70B Model Evaluation Run (vLLM)"
echo "=========================================="

LOCAL_MODELS=(
    "llama-3.1-70b-inst"
    "swallow-3.1-70b-inst"
)

API_MODELS=(
)

DATASETS=(
    "csv/ko-en-v2_dataset.csv"
    "csv/ko-zh-v2_dataset.csv"
    "csv/ko-ja-v2_dataset.csv"
    "csv/ko-ko-v2_dataset.csv"
)

BATCH_SIZE=1000  
USE_ASYNC_API=true  # Use async for API models
MAX_CONCURRENT=10   # Max concurrent API requests

# Total count
TOTAL_MODELS=$((${#LOCAL_MODELS[@]} + ${#API_MODELS[@]}))
TOTAL_DATASETS=${#DATASETS[@]}
TOTAL_RUNS=$((TOTAL_MODELS * TOTAL_DATASETS))

echo "Total Models: $TOTAL_MODELS (${#LOCAL_MODELS[@]} local + ${#API_MODELS[@]} API)"
echo "Total Datasets: $TOTAL_DATASETS"
echo "Total Runs: $TOTAL_RUNS"
echo "=========================================="

CURRENT_RUN=0

run_evaluation() {
    local model=$1
    local is_api=$2
    local dataset=$3
    
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    echo ""
    echo -e "${GREEN}[${CURRENT_RUN}/${TOTAL_RUNS}] Evaluating: $model${NC} on $dataset"
    echo "=========================================="
    
    CMD="python3 vllm_eval.py --model $model --dataset $dataset --all-templates"
    
    if [ "$is_api" = true ]; then
        if [ "$USE_ASYNC_API" = true ]; then
            CMD="$CMD --async-api --max-concurrent $MAX_CONCURRENT"
        fi
    else
        CMD="$CMD --batch-size $BATCH_SIZE"
        CMD="$CMD --tensor-parallel-size 2 --dtype float16"
    fi
    
    echo "Running: $CMD"
    
    if eval $CMD; then
        echo -e "${GREEN}✓ Success: $model${NC}"
    else
        echo -e "${RED}✗ Failed: $model${NC}"
        echo "Continuing with next model..."
    fi
}

START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "Starting Local Model Evaluations"
echo "=========================================="

for model in "${LOCAL_MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        run_evaluation "$model" false "$dataset"
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Total runs: $TOTAL_RUNS"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved in: results/"
echo "=========================================="