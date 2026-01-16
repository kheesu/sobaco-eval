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


RUN pip install --no-cache-dir \
    --extra-index-url https://wheels.vllm.ai/nightly \
    vllm

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p results csv notebooks

CMD ["python3", "evaluate.py", "--help"]

LABEL maintainer="kheesu"
LABEL description="Evaluation for the CJK-CUBE"