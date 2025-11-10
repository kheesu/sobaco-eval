"""
Results analysis script for SOBACO-EVAL
Generates comprehensive reports and visualizations from evaluation results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(results_path: str) -> tuple:
    """Load results CSV and metrics JSON"""
    results_path = Path(results_path)
    
    # Load CSV
    df = pd.read_csv(results_path)
    
    # Try to load corresponding metrics JSON
    metrics_path = results_path.parent / f"{results_path.stem}_metrics.json"
    metrics = None
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    
    return df, metrics


def plot_overall_performance(metrics_list: List[Dict], output_dir: Path):
    """Plot overall performance comparison across models"""
    if not metrics_list:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract data
    models = [m['model'] for m in metrics_list]
    overall_acc = [m['metrics']['overall_accuracy'] * 100 for m in metrics_list]
    bias_acc = [m['metrics']['bias_accuracy'] * 100 for m in metrics_list]
    culture_acc = [m['metrics']['culture_accuracy'] * 100 for m in metrics_list]
    
    # Plot 1: Overall Accuracy
    axes[0].bar(models, overall_acc, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Overall Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(overall_acc):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Bias Accuracy
    axes[1].bar(models, bias_acc, color='coral', alpha=0.8)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Bias Questions Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(bias_acc):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Culture Accuracy
    axes[2].bar(models, culture_acc, color='mediumseagreen', alpha=0.8)
    axes[2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[2].set_title('Culture Questions Accuracy', fontsize=14, fontweight='bold')
    axes[2].set_ylim(0, 100)
    for i, v in enumerate(culture_acc):
        axes[2].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = output_dir / 'overall_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_bias_analysis(metrics_list: List[Dict], output_dir: Path):
    """Plot bias selection rates"""
    if not metrics_list:
        return
    
    models = [m['model'] for m in metrics_list]
    bias_rates = [m['metrics']['bias_rate'] * 100 for m in metrics_list]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, bias_rates, color='crimson', alpha=0.7)
    
    ax.set_xlabel('Biased Option Selection Rate (%)', fontsize=12)
    ax.set_title('Bias Analysis: How Often Models Choose Biased Options', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Add value labels
    for i, (bar, v) in enumerate(zip(bars, bias_rates)):
        ax.text(v + 2, i, f'{v:.1f}%', va='center', fontsize=10)
    
    # Add reference line at 33% (random chance for 3 options)
    ax.axvline(x=33.33, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random (33%)')
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / 'bias_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_category_performance(metrics: Dict, output_dir: Path, model_name: str):
    """Plot per-category performance for a single model"""
    if 'per_category' not in metrics['metrics']:
        return
    
    categories = list(metrics['metrics']['per_category'].keys())
    accuracies = [metrics['metrics']['per_category'][cat]['accuracy'] * 100 for cat in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(categories, accuracies, color='teal', alpha=0.7)
    
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Per-Category Performance: {model_name}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Add value labels
    for bar, v in zip(bars, accuracies):
        ax.text(v + 2, bar.get_y() + bar.get_height()/2, f'{v:.1f}%', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / f'category_performance_{model_name.replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_confusion_heatmap(df: pd.DataFrame, output_dir: Path, model_name: str):
    """Create confusion matrix for answer vs prediction"""
    # Get unique answers
    unique_answers = sorted(df['answer'].dropna().unique())
    
    # Create confusion matrix
    confusion = pd.crosstab(
        df['answer'], 
        df['prediction'], 
        rownames=['Actual'], 
        colnames=['Predicted'],
        dropna=False
    )
    
    # Ensure all answers are in the matrix
    for ans in unique_answers:
        if ans not in confusion.index:
            confusion.loc[ans] = 0
        if ans not in confusion.columns:
            confusion[ans] = 0
    
    # Reorder
    confusion = confusion.loc[unique_answers, :]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'confusion_matrix_{model_name.replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_type_comparison(df: pd.DataFrame, output_dir: Path, model_name: str):
    """Compare performance on bias vs culture questions"""
    type_accuracy = df.groupby('type').apply(
        lambda x: (x['prediction'] == x['answer']).sum() / len(x) * 100
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(type_accuracy.index, type_accuracy.values, 
                   color=['coral', 'mediumseagreen'], alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Performance by Question Type: {model_name}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, v in zip(bars, type_accuracy.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 2, f'{v:.1f}%', 
                ha='center', fontsize=12)
    
    plt.tight_layout()
    output_path = output_dir / f'type_comparison_{model_name.replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def generate_report(results_paths: List[str], output_dir: str = None):
    """Generate comprehensive analysis report"""
    results_data = []
    dfs = []
    
    # Load all results
    for path in results_paths:
        df, metrics = load_results(path)
        if metrics:
            results_data.append(metrics)
        dfs.append((df, metrics, Path(path).stem))
    
    # Create output directory
    if output_dir is None:
        output_dir = Path('results/analysis')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating analysis report...")
    print(f"Output directory: {output_dir}")
    
    # Generate plots
    if len(results_data) > 1:
        # Multi-model comparison
        print("\nGenerating comparison plots...")
        plot_overall_performance(results_data, output_dir)
        plot_bias_analysis(results_data, output_dir)
    
    # Individual model plots
    for df, metrics, name in dfs:
        if metrics:
            print(f"\nGenerating plots for {metrics['model']}...")
            plot_category_performance(metrics, output_dir, metrics['model'])
            plot_type_comparison(df, output_dir, metrics['model'])
            
            # Only create confusion matrix if not too many unique answers
            if df['answer'].nunique() <= 10:
                plot_confusion_heatmap(df, output_dir, metrics['model'])
    
    # Generate summary table
    if results_data:
        summary_data = []
        for m in results_data:
            summary_data.append({
                'Model': m['model'],
                'Dataset': Path(m['dataset']).stem,
                'Overall Accuracy': f"{m['metrics']['overall_accuracy']:.2%}",
                'Bias Accuracy': f"{m['metrics']['bias_accuracy']:.2%}",
                'Bias Rate': f"{m['metrics']['bias_rate']:.2%}",
                'Culture Accuracy': f"{m['metrics']['culture_accuracy']:.2%}",
                'Invalid Rate': f"{m['metrics']['invalid_rate']:.2%}",
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / 'summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary table saved to: {summary_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SOBACO evaluation results")
    parser.add_argument('--results', nargs='+', required=True, 
                       help='Path(s) to result CSV file(s)')
    parser.add_argument('--output-dir', type=str, default='results/analysis',
                       help='Output directory for analysis')
    
    args = parser.parse_args()
    
    generate_report(args.results, args.output_dir)


if __name__ == "__main__":
    main()
