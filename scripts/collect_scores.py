#!/usr/bin/env python3
"""
Collect all scores from different models and global steps into a CSV file.
Uses record_new.txt for minerva_math and record.txt for other datasets.
"""

import os
import csv
from pathlib import Path
from collections import defaultdict

def extract_score(file_path):
    """Extract the score from a record file."""
    try:
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            # Format: path score
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[-1])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def collect_all_scores(base_dir):
    """Collect all scores from the gen_data directory."""
    base_path = Path(base_dir)
    
    # Store data as: (model_name, global_step) -> {dataset: score}
    data = defaultdict(dict)
    
    # Find all model directories
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Find all global_step directories
        for step_dir in model_dir.iterdir():
            if not step_dir.is_dir() or not step_dir.name.startswith('global_step_'):
                continue
            
            global_step = step_dir.name.replace('global_step_', '')
            
            # Navigate to merged/weqweasdas/
            merged_path = step_dir / 'merged' / 'weqweasdas'
            if not merged_path.exists():
                continue
            
            # Find all dataset directories
            for dataset_dir in merged_path.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                dataset_name = dataset_dir.name
                
                # Determine which record file to use
                if dataset_name == 'minerva_math':
                    record_file = dataset_dir / 'record_new.txt'
                else:
                    record_file = dataset_dir / 'record.txt'
                
                if record_file.exists():
                    score = extract_score(record_file)
                    if score is not None:
                        key = (model_name, global_step)
                        data[key][dataset_name] = score
                        print(f"Collected: {model_name} - step_{global_step} - {dataset_name}: {score}")
    
    return data

def write_csv(data, output_file):
    """Write the collected data to a CSV file."""
    if not data:
        print("No data collected!")
        return
    
    # Fixed dataset order
    all_datasets = ['math500', 'minerva_math', 'olympiadbench', 'aime_hmmt_brumo_cmimc_amc23']
    
    # Prepare CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header - combine model and step in first column
        header = ['model_name_step'] + all_datasets
        writer.writerow(header)
        
        # Write data rows, sorted by model name and global step
        sorted_keys = sorted(data.keys(), key=lambda x: (x[0], int(x[1])))
        for model_name, global_step in sorted_keys:
            # Combine model name and step in format: model_name_stepXXX
            model_step = f"{model_name}_step{global_step}"
            row = [model_step]
            scores = data[(model_name, global_step)]
            for dataset in all_datasets:
                row.append(scores.get(dataset, ''))
            writer.writerow(row)
    
    print(f"\nCSV file written to: {output_file}")
    print(f"Total rows: {len(data)}")
    print(f"Datasets: {', '.join(all_datasets)}")

def main():
    base_dir = '/home/chenluy/Reinforce-Ada/data/gen_data'
    output_file = '/home/chenluy/Reinforce-Ada/data/gen_data/scores_summary.csv'
    
    print("Collecting scores from all models and global steps...")
    data = collect_all_scores(base_dir)
    
    print(f"\nWriting to CSV...")
    write_csv(data, output_file)
    
    print("\nDone!")

if __name__ == '__main__':
    main()

