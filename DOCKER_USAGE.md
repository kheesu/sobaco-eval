# Docker Usage Guide

This guide explains how to use Docker and Docker Compose to run SOBACO-EVAL evaluations.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- For GPU support: NVIDIA Docker runtime ([NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- Docker Compose installed (usually included with Docker Desktop)

## Quick Start

### 1. Setup Environment Variables

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys and GPU configuration:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# GPU Configuration (optional)
# Examples:
# Use GPU 0 only: CUDA_VISIBLE_DEVICES=0
# Use GPUs 0 and 1: CUDA_VISIBLE_DEVICES=0,1
# Use all GPUs: CUDA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0
```

### 2. Build Docker Images

```bash
# Build GPU-enabled image (for local models like Llama)
docker-compose build sobaco-eval-gpu

# Build CPU-only image (for API models)
docker-compose build sobaco-eval-cpu

# Build Jupyter notebook image
docker-compose build jupyter
```

## Running Evaluations

### GPU-Enabled Container (Local Models)

Run evaluations with local models like Llama 3.1:

```bash
# Evaluate on a single dataset with specific GPU
CUDA_VISIBLE_DEVICES=0 docker-compose run --rm sobaco-eval-gpu \
  python3 evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv --subset 0.1

# Evaluate on all datasets using GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 docker-compose run --rm sobaco-eval-gpu \
  python3 evaluate.py --model llama-3.1-8b --all-datasets

# Use all available GPUs
CUDA_VISIBLE_DEVICES=all docker-compose run --rm sobaco-eval-gpu \
  python3 evaluate.py --model llama-3.1-70b-inst --all-datasets
```

### CPU-Only Container (API Models)

Run evaluations with API-based models (GPT-4, Claude, Gemini):

```bash
# Evaluate GPT-4 on Japanese dataset
docker-compose run --rm sobaco-eval-cpu \
  python3 evaluate.py --model gpt-4 --dataset csv/ja_dataset.csv

# Evaluate multiple API models
docker-compose run --rm sobaco-eval-cpu \
  python3 evaluate.py --model gpt-4 claude-3-opus gemini-pro --all-datasets
```

### Jupyter Notebook

Start Jupyter for analysis:

```bash
# Start Jupyter server
docker-compose up jupyter

# Access at http://localhost:8888
```

## GPU Configuration Options

You can control which GPUs to use in three ways:

### 1. Using Environment Variable (Recommended)

Set `CUDA_VISIBLE_DEVICES` in your `.env` file:

```bash
# .env
CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

Then run:
```bash
docker-compose run --rm sobaco-eval-gpu python3 evaluate.py --model llama-3.1-8b --all-datasets
```

### 2. Using Command Line Override

Override the environment variable when running:

```bash
# Use only GPU 2
CUDA_VISIBLE_DEVICES=2 docker-compose run --rm sobaco-eval-gpu \
  python3 evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv

# Use GPUs 1, 2, and 3
CUDA_VISIBLE_DEVICES=1,2,3 docker-compose run --rm sobaco-eval-gpu \
  python3 evaluate.py --model llama-3.1-70b-inst --all-datasets
```

### 3. Using Docker Run Directly

Run container directly with specific GPU:

```bash
docker run --rm --gpus '"device=0"' \
  -v $(pwd)/csv:/app/csv:ro \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -e CUDA_VISIBLE_DEVICES=0 \
  sobaco-eval:gpu \
  python3 evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv
```

## Common Commands

### Analyze Results

```bash
# Analyze evaluation results
docker-compose run --rm sobaco-eval-cpu \
  python3 analyze_results.py --results results/llama-3.1-8b_ja_dataset.json
```

### Interactive Shell

```bash
# Open shell in GPU container
docker-compose run --rm sobaco-eval-gpu bash

# Open shell in CPU container
docker-compose run --rm sobaco-eval-cpu bash
```

### Clean Up

```bash
# Stop all running containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove images
docker rmi sobaco-eval:gpu sobaco-eval:cpu sobaco-eval:jupyter
```

## Volume Mounts

The following directories are mounted:

- `./csv` → `/app/csv` (read-only) - Dataset files
- `./results` → `/app/results` - Evaluation results
- `./config.yaml` → `/app/config.yaml` - Configuration
- `./notebooks` → `/app/notebooks` - Jupyter notebooks
- `huggingface-cache` → `/root/.cache/huggingface` - Model cache (GPU only)

Results are automatically saved to your local `./results` directory.

## Troubleshooting

### GPU Not Available

If you get "CUDA not available" error:

1. Verify NVIDIA Docker runtime is installed:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

2. Check GPU visibility:
   ```bash
   CUDA_VISIBLE_DEVICES=0 docker-compose run --rm sobaco-eval-gpu \
     python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Out of Memory

For large models, use 8-bit quantization or select specific GPUs:

```bash
# Use multiple GPUs for large models
CUDA_VISIBLE_DEVICES=0,1,2,3 docker-compose run --rm sobaco-eval-gpu \
  python3 evaluate.py --model llama-3.1-70b-inst --dataset csv/ja_dataset.csv
```

### Permission Issues

If you encounter permission issues with results directory:

```bash
# Fix permissions (Linux/Mac)
sudo chown -R $USER:$USER ./results

# Or run with user ID
docker-compose run --rm --user $(id -u):$(id -g) sobaco-eval-cpu \
  python3 evaluate.py --model gpt-4 --dataset csv/ja_dataset.csv
```

## Examples

### Example 1: Quick Test on 10% of Data

```bash
CUDA_VISIBLE_DEVICES=0 docker-compose run --rm sobaco-eval-gpu \
  python3 evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv --subset 0.1
```

### Example 2: Full Evaluation on Multiple GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 docker-compose run --rm sobaco-eval-gpu \
  python3 evaluate.py --model llama-3.1-8b --all-datasets
```

### Example 3: API Model Evaluation

```bash
docker-compose run --rm sobaco-eval-cpu \
  python3 evaluate.py --model gpt-4 claude-3-opus --all-datasets
```

### Example 4: Using Specific GPU for Different Models

```bash
# Run on GPU 0
CUDA_VISIBLE_DEVICES=0 docker-compose run --rm sobaco-eval-gpu \
  python3 evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv &

# Run on GPU 1 simultaneously
CUDA_VISIBLE_DEVICES=1 docker-compose run --rm sobaco-eval-gpu \
  python3 evaluate.py --model llama-3.1-8b --dataset csv/ja-ko_dataset.csv &
```
