# SOBACO-EVAL: Social Bias and Cultural Awareness Evaluation

A comprehensive framework for evaluating Large Language Models (LLMs) on social bias and cultural awareness across multiple languages. This repository tests models like Llama 3.1 8B, GPT-4, and others on culturally nuanced questions in Japanese, Korean, and Chinese.

## 📊 Overview

This repository evaluates LLMs on their ability to:
- **Avoid social biases** (e.g., stereotypes based on appearance)
- **Understand cultural context** (e.g., honorific language in hierarchical relationships)
- **Handle multilingual scenarios** across Japanese, Korean, and Chinese

## 🗂️ Dataset Structure

The evaluation datasets are located in `./csv/`:
- `ja_dataset.csv` - Japanese language evaluation (11,954 samples)
- `ja-ko_dataset.csv` - Korean language evaluation (11,954 samples)
- `ja-zh_dataset.csv` - Chinese language evaluation (11,954 samples)

Each dataset contains:
- `context`: The main scenario
- `additional_context`: Additional information
- `type`: Question type (`bias` or `culture`)
- `question`: The evaluation question
- `options`: Multiple choice options
- `answer`: Correct answer
- `biased_option`: The stereotypical/biased answer
- `category`: Classification category

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kheesu/sobaco-eval.git
cd sobaco-eval

# For GPU support (recommended for local models like Llama)
# Install CUDA-enabled PyTorch first (choose your CUDA version):
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# OR
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# Then install remaining dependencies
pip install -r requirements.txt

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# For CPU-only (API models only, no local model support):
pip install -r requirements.txt
```

### Running Evaluations

#### Evaluate a single model
```bash
python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv
```

#### Quick test on 10% of data (recommended for testing)
```bash
python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv --subset 0.1
```

#### Evaluate all datasets
```bash
python evaluate.py --model llama-3.1-8b --all-datasets
```

#### Evaluate multiple models
```bash
python evaluate.py --model llama-3.1-8b gpt-4 --all-datasets
```

## 🤖 Supported Models

- **Meta Llama 3.1** (8B, 70B, 405B)
- **GPT-4** / **GPT-3.5** (via OpenAI API)
- **Claude** (via Anthropic API)
- **Gemini** (via Google API)
- **Custom models** (configure in `config.yaml`)

## 📈 Evaluation Metrics

- **Overall Accuracy**: Percentage of correct answers
- **Bias Rate**: Percentage of biased option selections
- **Culture Score**: Accuracy on culture-specific questions
- **Per-category Performance**: Breakdown by question type

## 🛠️ Configuration

Edit `config.yaml` to configure:
- Model parameters (temperature, max tokens, etc.)
- API keys for commercial models
- Evaluation settings
- Output paths

## 📊 Results Analysis

After running evaluations, analyze results:

```bash
python analyze_results.py --results results/llama-3.1-8b_ja_dataset.json
```

This generates:
- Performance summary tables
- Bias analysis charts
- Per-category breakdowns
- Comparison plots (when multiple models evaluated)

## 📁 Project Structure

```
sobaco-eval/
├── csv/                      # Evaluation datasets
│   ├── ja_dataset.csv
│   ├── ja-ko_dataset.csv
│   └── ja-zh_dataset.csv
├── evaluate.py               # Main evaluation script
├── analyze_results.py        # Results analysis
├── utils.py                  # Utility functions
├── config.yaml              # Model configuration
├── requirements.txt         # Python dependencies
├── results/                 # Evaluation results (generated)
└── notebooks/               # Example notebooks
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
