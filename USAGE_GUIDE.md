# SOBACO-EVAL Usage Guide

This guide provides detailed instructions for using the SOBACO-EVAL framework.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Running Evaluations](#running-evaluations)
5. [Analyzing Results](#analyzing-results)
6. [Using Jupyter Notebooks](#using-jupyter-notebooks)
7. [Adding Custom Models](#adding-custom-models)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kheesu/sobaco-eval.git
cd sobaco-eval
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

**IMPORTANT: For GPU Support (Local Models)**

The default PyTorch from PyPI is CPU-only. For local models like Llama 3.1, you need CUDA-enabled PyTorch:

```bash
pip install --upgrade pip

# Step 1: Install CUDA-enabled PyTorch (choose your CUDA version)
# Check your CUDA version: nvidia-smi

# For CUDA 11.8:
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install remaining dependencies
pip install -r requirements.txt

# Step 3: Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**For API Models Only (No GPU Needed)**

If you only plan to use API models (GPT, Claude, Gemini):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up API Keys (Optional)

If using API-based models (GPT, Claude, Gemini):

```bash
cp .env.example .env
# Edit .env and add your API keys
```

---

## Quick Start

### Run a Quick Test

Test the setup without GPU or API keys:

```bash
python quick_test.py --sample-size 50
```

This runs a mock evaluation on 50 samples to verify everything is working.

### Run Real Evaluation

For local models (requires GPU):

```bash
# Evaluate Llama 3.1 8B on Japanese dataset
python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv
```

For API models:

```bash
# Evaluate GPT-4 (requires OPENAI_API_KEY in .env)
python evaluate.py --model gpt-4 --dataset csv/ja_dataset.csv
```

---

## Configuration

The `config.yaml` file contains all configuration settings.

### Model Configuration

#### Local Models (Hugging Face)

```yaml
local_models:
  llama-3.1-8b:
    model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
    load_in_8bit: true
    device_map: "auto"
    max_tokens: 100
    temperature: 0.1
```

**Key Parameters:**
- `model_name`: Hugging Face model identifier
- `load_in_8bit`: Enable 8-bit quantization (saves memory)
- `device_map`: "auto" for automatic device placement
- `temperature`: Controls randomness (0.0-1.0)

#### API Models

```yaml
api_models:
  gpt-4:
    provider: "openai"
    model_name: "gpt-4-turbo-preview"
    max_tokens: 100
    temperature: 0.1
```

**Supported Providers:**
- `openai`: GPT models
- `anthropic`: Claude models
- `google`: Gemini models

### Prompt Templates

Customize prompts for different languages:

```yaml
prompts:
  system_prompt_ja: |
    あなたは質問に正確に答える有益なアシスタントです。
```

---

## Running Evaluations

### Basic Usage

```bash
python evaluate.py --model MODEL_NAME --dataset DATASET_PATH
```

### Examples

#### Single Model, Single Dataset

```bash
python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv
```

#### Single Model, All Datasets

```bash
python evaluate.py --model llama-3.1-8b --all-datasets
```

#### Test on Subset (10% of data)

```bash
# Quick test to verify setup - evaluates only first 10% of dataset
python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv --subset 0.1

# Or 20% for slightly longer test
python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv --subset 0.2
```

#### Multiple Models, All Datasets

```bash
python evaluate.py --model llama-3.1-8b gpt-4 claude-3-sonnet --all-datasets
```

#### Custom Output Directory

```bash
python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv --output-dir my_results
```

### Command-Line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--model` | Model name(s) to evaluate | Yes |
| `--dataset` | Path to specific dataset | No* |
| `--all-datasets` | Evaluate on all datasets in csv/ | No* |
| `--config` | Path to config file (default: config.yaml) | No |
| `--output-dir` | Override output directory | No |
| `--subset` | Evaluate on subset (e.g., 0.1 for 10%, 0.2 for 20%) | No |

*Either `--dataset` or `--all-datasets` must be specified.

---

## Analyzing Results

After running evaluations, analyze the results:

### Basic Analysis

```bash
python analyze_results.py --results results/llama-3.1-8b_ja_dataset.csv
```

### Compare Multiple Models

```bash
python analyze_results.py --results results/*.csv --output-dir results/analysis
```

### Generated Outputs

The analysis script generates:

1. **summary.csv**: Overall metrics table
2. **overall_performance.png**: Accuracy comparison across models
3. **bias_analysis.png**: Bias selection rates
4. **category_performance_*.png**: Per-category breakdown for each model
5. **confusion_matrix_*.png**: Confusion matrices (if applicable)
6. **type_comparison_*.png**: Bias vs Culture question performance

---

## Using Jupyter Notebooks

### Start Jupyter

```bash
jupyter notebook notebooks/evaluation_demo.ipynb
```

### Notebook Contents

The demo notebook includes:

1. **Data Exploration**: Examine dataset structure
2. **Sample Evaluation**: Run a small-scale test
3. **Results Analysis**: Calculate and visualize metrics
4. **Bias Detection**: Identify biased predictions

### Key Features

- Interactive data visualization
- Step-by-step evaluation process
- Easy experimentation with parameters
- Loading and analyzing real results

---

## Adding Custom Models

### Local Models

1. Add model configuration to `config.yaml`:

```yaml
local_models:
  my-custom-model:
    model_name: "organization/model-name"
    load_in_8bit: true
    device_map: "auto"
    max_tokens: 100
    temperature: 0.1
```

2. Run evaluation:

```bash
python evaluate.py --model my-custom-model --dataset csv/ja_dataset.csv
```

### API Models

1. Add model configuration:

```yaml
api_models:
  my-api-model:
    provider: "openai"  # or "anthropic", "google"
    model_name: "model-identifier"
    max_tokens: 100
    temperature: 0.1
```

2. Set API key in `.env`:

```bash
OPENAI_API_KEY=your_key_here
```

3. Run evaluation:

```bash
python evaluate.py --model my-api-model --dataset csv/ja_dataset.csv
```

---

## Troubleshooting

### Out of Memory (OOM) Error

**Problem**: GPU runs out of memory when loading model.

**Solutions**:
1. Enable 8-bit quantization:
   ```yaml
   load_in_8bit: true
   ```

2. Use a smaller model:
   ```bash
   python evaluate.py --model llama-3.2-3b --dataset csv/ja_dataset.csv
   ```

3. Reduce batch size in `config.yaml`:
   ```yaml
   evaluation:
     batch_size: 4  # Reduce from default 8
   ```

### API Rate Limits

**Problem**: API requests are rate limited.

**Solutions**:
1. Add delays between requests (modify `evaluate.py`)
2. Use a lower-tier model
3. Split evaluation across multiple sessions

### Model Not Found

**Problem**: Model name not recognized.

**Solutions**:
1. Check model name in `config.yaml`
2. Ensure model is in either `local_models` or `api_models`
3. For Hugging Face models, verify the model ID is correct

### Import Errors

**Problem**: Module not found errors.

**Solutions**:
1. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Check Python version (requires 3.8+):
   ```bash
   python --version
   ```

3. Ensure virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```

### Permission Errors

**Problem**: Cannot access Hugging Face gated models (like Llama).

**Solutions**:
1. Accept model license on Hugging Face website
2. Set HF token in `.env`:
   ```bash
   HF_TOKEN=your_huggingface_token
   ```
3. Login via CLI:
   ```bash
   huggingface-cli login
   ```

### Invalid Predictions

**Problem**: High rate of invalid/unparseable predictions.

**Solutions**:
1. Adjust prompt template in `config.yaml`
2. Increase temperature for more diverse outputs
3. Modify `extract_answer()` function in `utils.py` for better parsing

---

## Performance Tips

### For Faster Evaluation

1. **Use 8-bit quantization**: Reduces memory and speeds up inference
2. **Batch processing**: Increase batch size if memory allows
3. **GPU acceleration**: Ensure CUDA is properly installed
4. **Smaller sample**: Test on subset first with `quick_test.py`

### For Better Accuracy

1. **Lower temperature**: Use 0.1 for more deterministic outputs
2. **Better prompts**: Customize system prompts for each language
3. **Few-shot examples**: Add examples to prompts (modify templates)

---

## Advanced Usage

### Custom Evaluation Metrics

Modify `calculate_metrics()` in `utils.py` to add custom metrics:

```python
def calculate_metrics(df: pd.DataFrame) -> Dict:
    metrics = {}
    # Add your custom metrics here
    metrics['custom_metric'] = your_calculation(df)
    return metrics
```

### Custom Prompt Engineering

Edit prompt templates in `config.yaml`:

```yaml
prompts:
  format_template: |
    Your custom prompt format here...
    Context: {context}
    Question: {question}
```

### Parallel Evaluation

Modify `evaluate.py` to process multiple samples in parallel using threading or multiprocessing.

---

## Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/kheesu/sobaco-eval/issues)
- **Discussions**: Join discussions for questions and ideas
- **Documentation**: Check README.md and code comments

---

## Citation

If you use this framework in your research:

```bibtex
@software{sobaco_eval,
  title={SOBACO-EVAL: Social Bias and Cultural Awareness Evaluation},
  author={Your Name},
  year={2025},
  url={https://github.com/kheesu/sobaco-eval}
}
```
