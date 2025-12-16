#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import utils


def main():
    parser = argparse.ArgumentParser(description='Calculate metrics from CSV files')
    parser.add_argument('--csv', required=True, help='First CSV file path')
    
    args = parser.parse_args()
    
    # Process first file
    df1 = pd.read_csv(args.file1)
    metrics1 = utils.calculate_metrics(df1)
    
    # Print results
    print(f"\n{args.csv}:")
    print(json.dumps(metrics1, indent=2))


if __name__ == '__main__':
    main()
