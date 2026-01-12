# All Templates Evaluation Feature

## Overview

The `--all-templates` flag enables evaluation of datasets on all three prompt templates, reporting results both separately for each template and averaged across all templates.

## Usage

```bash
python evaluate.py --model <model_name> --dataset <dataset_path> --all-templates
```

### Example

```bash
python evaluate.py --model gpt-5.1 --dataset csv/ja-en_dataset.csv --all-templates
```

### With Multiple Datasets

```bash
python evaluate.py --model gpt-5.1 --all-datasets --all-templates
```

### With Async API

```bash
python evaluate.py --model gpt-5.1 --dataset csv/ja-en_dataset.csv --all-templates --async-api --max-concurrent 10
```

## How It Works

1. **Template Evaluation**: When `--all-templates` is enabled, the system:
   - Detects the target language from the dataset filename
   - Finds all available templates for that language (typically 3 templates: `format_template_{lang}_1`, `format_template_{lang}_2`, `format_template_{lang}_3`)
   - Evaluates the dataset on each template sequentially
   
2. **Individual Results**: For each template, the system:
   - Calculates metrics (accuracy, etc.)
   - Prints results to console
   - Saves detailed CSV with predictions (e.g., `model_dataset_template_1_timestamp.csv`)
   - Saves metrics JSON (e.g., `model_dataset_template_1_timestamp_metrics.json`)

3. **Averaged Results**: After all templates are evaluated:
   - Calculates the average of all metrics across templates
   - Prints averaged results to console
   - Saves averaged metrics JSON (e.g., `model_dataset_averaged_timestamp_metrics.json`)

## Output Files

When using `--all-templates`, you will get the following output files:

### CSV Files (One per Template):
- `{model}_{dataset}_template_1_{timestamp}.csv` - Full predictions for template 1
- `{model}_{dataset}_template_2_{timestamp}.csv` - Full predictions for template 2
- `{model}_{dataset}_template_3_{timestamp}.csv` - Full predictions for template 3

### Single Comprehensive Metrics JSON:
- `{model}_{dataset}_all_templates_{timestamp}_metrics.json` - Contains all results in one file with sections:
  - `results.template_1` - Metrics for template 1
  - `results.template_2` - Metrics for template 2
  - `results.template_3` - Metrics for template 3
  - `results.averaged` - Averaged metrics across all templates

### Example JSON Structure:
```json
{
  "model": "gpt-5.1",
  "dataset": "csv/ja-en_dataset.csv",
  "timestamp": "2026-01-12T10:30:00",
  "evaluation_type": "all_templates",
  "results": {
    "template_1": {
      "accuracy": 0.85,
      "...": "..."
    },
    "template_2": {
      "accuracy": 0.87,
      "...": "..."
    },
    "template_3": {
      "accuracy": 0.86,
      "...": "..."
    },
    "averaged": {
      "accuracy": 0.86,
      "...": "..."
    }
  }
}
```

## Console Output

The console will show:

1. Progress for each template evaluation
2. Individual metrics for each template
3. A summary section showing **AVERAGED RESULTS ACROSS ALL TEMPLATES**

Example output:
```
============================================================
Evaluating on all 3 templates for csv/ja-en_dataset.csv
============================================================

--- Template 1/3 ---
Using ja format template 1
Evaluating gpt-5.1: 100%|████████████| 100/100 [00:30<00:00]

Metrics for gpt-5.1 on ja-en_dataset (Template 1):
  Accuracy: 0.85
  ...

--- Template 2/3 ---
Using ja format template 2
...

--- Template 3/3 ---
Using ja format template 3
...

============================================================
AVERAGED RESULTS ACROSS ALL TEMPLATES
============================================================
Metrics for gpt-5.1 on ja-en_dataset (AVERAGED):
  Accuracy: 0.87
  ...
```

## Benefits

- **Robustness**: Evaluates model performance across different prompt formulations
- **Variance Analysis**: Helps identify if model performance is sensitive to prompt formatting
- **Comprehensive Reporting**: Provides both granular (per-template) and aggregate (averaged) metrics
- **Fair Comparison**: Reduces bias from any single template formulation

## Compatibility

The `--all-templates` flag works with:
- Local models (HuggingFace)
- API models (OpenAI, Anthropic, Google)
- Async API evaluation (`--async-api`)
- Ollama models (`--use-ollama`)
- Batch processing (`--batch-size`)
- Subset evaluation (`--subset`)

## Notes

- Each dataset must have all three templates defined in `config.yaml` for its target language
- Evaluation time will be approximately 3x longer than single-template evaluation
- All intermediate results are saved, so you can analyze individual template performance later
