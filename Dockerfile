ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base


ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade pip packaging wheel setuptools

COPY requirements.txt ./


RUN export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//') && \
    pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu128-cp38-abi3-manylinux_2_31_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p results csv notebooks

CMD ["python3", "evaluate.py", "--help"]

LABEL maintainer="kheesu"
LABEL description="Evaluation for the CJK-CUBE"