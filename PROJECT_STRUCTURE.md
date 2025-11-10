# SOBACO-EVAL Project Structure

```
sobaco-eval/
├── README.md                      # Main project documentation
├── USAGE_GUIDE.md                 # Detailed usage instructions
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── config.yaml                    # Model and evaluation configuration
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
│
├── csv/                           # Evaluation datasets
│   ├── ja_dataset.csv            # Japanese (11,954 samples)
│   ├── ja-ko_dataset.csv         # Korean (11,954 samples)
│   └── ja-zh_dataset.csv         # Chinese (11,954 samples)
│
├── evaluate.py                    # Main evaluation script
├── analyze_results.py             # Results analysis and visualization
├── utils.py                       # Utility functions
├── quick_test.py                  # Quick test without GPU/API
├── quick_start.sh                 # Setup automation script
│
├── notebooks/                     # Jupyter notebooks
│   └── evaluation_demo.ipynb     # Interactive demo and tutorial
│
└── results/                       # Evaluation results (generated)
    ├── *.csv                      # Prediction results
    ├── *_metrics.json            # Evaluation metrics
    └── analysis/                  # Analysis outputs
        ├── summary.csv
        ├── overall_performance.png
        ├── bias_analysis.png
        └── *.png
```

## File Descriptions

### Core Scripts

- **`evaluate.py`**: Main evaluation script that:
  - Loads LLMs (local or API-based)
  - Runs inference on datasets
  - Calculates metrics
  - Saves results

- **`analyze_results.py`**: Analysis script that:
  - Loads evaluation results
  - Generates comparison plots
  - Creates summary tables
  - Produces visualizations

- **`utils.py`**: Utility module with:
  - Dataset loading functions
  - Prompt formatting
  - Answer extraction
  - Metrics calculation

- **`quick_test.py`**: Testing script that:
  - Runs mock evaluation
  - Verifies setup
  - Demonstrates workflow
  - No GPU/API required

### Configuration Files

- **`config.yaml`**: Central configuration for:
  - Model definitions (local & API)
  - Evaluation parameters
  - Prompt templates
  - Output settings

- **`.env.example`**: Template for:
  - API keys (OpenAI, Anthropic, Google)
  - Hugging Face tokens
  - Environment variables

### Documentation

- **`README.md`**: Project overview with:
  - Quick start guide
  - Feature highlights
  - Installation steps
  - Basic usage examples

- **`USAGE_GUIDE.md`**: Comprehensive guide with:
  - Detailed instructions
  - Configuration options
  - Troubleshooting tips
  - Advanced usage

### Datasets

The `csv/` directory contains three parallel datasets:

Each dataset includes:
- **context**: Main scenario
- **additional_context**: Extra information
- **type**: `bias` or `culture`
- **question**: Evaluation question
- **options**: Multiple choice answers
- **answer**: Correct answer
- **biased_option**: Stereotypical option (for bias questions)
- **category**: Question category

### Results

The `results/` directory (created during evaluation) contains:

- **CSV files**: Full predictions with columns:
  - Original dataset fields
  - `prediction`: Model's answer
  - `raw_response`: Raw model output

- **JSON files**: Metrics including:
  - Overall accuracy
  - Bias accuracy & rate
  - Culture accuracy
  - Per-category performance

- **PNG files**: Visualizations:
  - Performance comparisons
  - Bias analysis charts
  - Category breakdowns
  - Confusion matrices

## Workflow

```
┌─────────────────┐
│  Load Dataset   │
│  (csv/*.csv)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load Model     │
│  (config.yaml)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Run Inference  │
│  (evaluate.py)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Save Results    │
│ (results/*.csv) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze Results │
│ (analyze_results│
│      .py)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualizations │
│  & Reports      │
└─────────────────┘
```

## Quick Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your keys
```

### Evaluation
```bash
# Test setup (no GPU/API needed)
python quick_test.py

# Quick test on 10% of data (recommended first step)
python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv --subset 0.1

# Evaluate single model (full dataset)
python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv

# Evaluate all datasets
python evaluate.py --model llama-3.1-8b --all-datasets

# Compare multiple models
python evaluate.py --model llama-3.1-8b gpt-4 --all-datasets
```

### Analysis
```bash
# Analyze single result
python analyze_results.py --results results/llama-3.1-8b_ja_dataset.csv

# Compare all results
python analyze_results.py --results results/*.csv
```

### Notebooks
```bash
# Open Jupyter notebook
jupyter notebook notebooks/evaluation_demo.ipynb
```

## Key Features

✅ **Multiple Model Support**: Local (Llama, etc.) and API (GPT, Claude, Gemini)
✅ **Multilingual**: Japanese, Korean, Chinese datasets
✅ **Bias Detection**: Identifies stereotypical responses
✅ **Cultural Awareness**: Tests cultural context understanding
✅ **Comprehensive Metrics**: Accuracy, bias rates, per-category analysis
✅ **Rich Visualizations**: Charts, plots, confusion matrices
✅ **Easy Configuration**: YAML-based setup
✅ **Extensible**: Easy to add new models and metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests with `quick_test.py`
5. Submit a pull request

## Support

- 📖 Documentation: README.md, USAGE_GUIDE.md
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 📧 Contact: [Your contact info]
