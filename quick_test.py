"""
Quick test script for SOBACO-EVAL
Tests the evaluation pipeline on a small sample without requiring GPU or API keys
"""

import pandas as pd
from pathlib import Path
from utils import (
    load_dataset, format_prompt, extract_answer, 
    calculate_metrics, print_metrics, parse_options
)
import random

def mock_llm_response(prompt: str, options: list) -> str:
    """
    Mock LLM that returns random responses
    Replace this with actual LLM inference in production
    """
    # Simple heuristic: pick random option
    return random.choice(options)


def run_quick_test(dataset_path: str = "csv/ja_dataset.csv", sample_size: int = 50):
    """Run a quick test evaluation on a small sample"""
    
    print("="*60)
    print("SOBACO-EVAL Quick Test")
    print("="*60)
    print(f"\nThis is a demonstration using mock responses.")
    print(f"For real evaluation, use evaluate.py with actual models.\n")
    
    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    df = load_dataset(dataset_path)
    
    # Take a small sample
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
    print(f"Evaluating on {len(sample_df)} samples...\n")
    
    # Mock system prompt and template
    system_prompt = "You are a helpful assistant."
    template = """Context: {context}
Additional Information: {additional_context}
Question: {question}
Options: {options}

Select the most appropriate answer."""
    
    # Evaluate each sample
    predictions = []
    for idx, row in sample_df.iterrows():
        prompt = format_prompt(row, template)
        options = parse_options(row['options'])
        
        # Get mock response
        response = mock_llm_response(prompt, options)
        prediction = extract_answer(response, options)
        
        predictions.append(prediction)
    
    sample_df['prediction'] = predictions
    sample_df['raw_response'] = predictions  # In mock, they're the same
    
    # Calculate metrics
    metrics = calculate_metrics(sample_df)
    
    # Print results
    print_metrics(metrics, "Mock LLM (Random Baseline)")
    
    # Show some examples
    print("\n📋 Sample Predictions:\n")
    for idx, (_, row) in enumerate(sample_df.head(3).iterrows(), 1):
        print(f"Example {idx}:")
        print(f"  Question: {row['question']}")
        print(f"  Options: {row['options']}")
        print(f"  Correct: {row['answer']}")
        print(f"  Predicted: {row['prediction']}")
        print(f"  ✓ Correct" if row['prediction'] == row['answer'] else "  ✗ Incorrect")
        print()
    
    # Save results
    output_dir = Path("results/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "quick_test_results.csv"
    sample_df.to_csv(output_path, index=False)
    print(f"📁 Results saved to: {output_path}")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run real evaluation: python evaluate.py --model llama-3.1-8b --dataset csv/ja_dataset.csv")
    print("3. Analyze results: python analyze_results.py --results results/*.csv")
    print("4. Explore notebook: jupyter notebook notebooks/evaluation_demo.ipynb")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick test of SOBACO-EVAL")
    parser.add_argument('--dataset', type=str, default='csv/ja_dataset.csv',
                       help='Path to dataset')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Number of samples to test')
    
    args = parser.parse_args()
    
    run_quick_test(args.dataset, args.sample_size)
