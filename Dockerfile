# Multi-stage Dockerfile for SOBACO-EVAL
# Supports both CPU and GPU (CUDA) execution

ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
# For CUDA support, use requirements-cuda.txt
ARG INSTALL_CUDA=true
RUN if [ "$INSTALL_CUDA" = "true" ]; then \
        pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu128 && \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p results csv notebooks

# Set default command
CMD ["python3", "evaluate.py", "--help"]

# Labels
LABEL maintainer="kheesu"
LABEL description="Evaluation for the CultureLLM Project"
LABEL version="1.0"
