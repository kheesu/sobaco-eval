#!/bin/bash

# Quick start script for SOBACO-EVAL
# Run a quick evaluation on a sample of the data

echo "🚀 SOBACO-EVAL Quick Start"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if CUDA is needed
echo ""
echo "⚠️  IMPORTANT: For local models (Llama), you need CUDA-enabled PyTorch!"
echo "Check if CUDA is available:"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
    echo ""
    echo "Installing CUDA-enabled PyTorch..."
    pip install -q --upgrade pip
    pip install -q torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
    pip install -q -r requirements.txt
else
    echo "❌ No NVIDIA GPU detected - installing CPU-only version"
    echo "   (You can only use API models like GPT-4, Claude, Gemini)"
    echo ""
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
fi

echo ""
echo "Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Available commands:"
echo "  1. Quick test on 10% of data (recommended first step):"
echo "     python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv --subset 0.1"
echo ""
echo "  2. Evaluate Llama 3.1 8B on Japanese dataset:"
echo "     python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv"
echo ""
echo "  3. Evaluate on all datasets:"
echo "     python evaluate.py --model llama-3.1-8b --all-datasets"
echo ""
echo "  4. Analyze results:"
echo "     python analyze_results.py --results results/*.csv"
echo ""
echo "  5. Open Jupyter notebook:"
echo "     jupyter notebook notebooks/evaluation_demo.ipynb"
echo ""
echo "📝 Don't forget to set up your API keys in .env file if using API models!"
echo "   Copy .env.example to .env and add your keys."
echo ""
