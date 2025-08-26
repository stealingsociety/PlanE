#!/usr/bin/env python3
# CSV to JSON with Train/Val/Test Split

import pandas as pd
import json
import ast
import gzip
import os
import sys
from sklearn.model_selection import train_test_split

def csv_to_train_val_test_split(csv_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Convert CSV to train/val/test JSON.gz files with 80/10/10 split"""
    
    # Verify ratios sum to 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Generate output filenames
    base_name = csv_file.replace('.csv', '')
    train_file = f"{base_name}_train.json.gz"
    val_file = f"{base_name}_val.json.gz"
    test_file = f"{base_name}_test.json.gz"
    
    # Read and process all data
    df = pd.read_csv(csv_file)
    data_list = []
    
    for _, row in df.iterrows():
        edges = ast.literal_eval(row['EDGES'])
        label = int(row['COEFFICIENTS'])
        
        nodes = set(sum(edges, ()))
        node_map = {n: i for i, n in enumerate(sorted(nodes))}
        
        edge_pairs = [[node_map[u], node_map[v]] for u, v in edges]
        edge_pairs += [[node_map[v], node_map[u]] for u, v in edges]
        
        if edge_pairs:
            sources = [pair[0] for pair in edge_pairs]
            targets = [pair[1] for pair in edge_pairs]
            edge_index = [sources, targets]
        else:
            edge_index = [[], []]
        
        data_dict = {
            'x': [0] * len(nodes) if nodes else [0],
            'edge_index': edge_index,
            'edge_attr': [0] * len(edge_pairs),
            'y': [float(label)]
        }
        data_list.append(data_dict)
    
    # First split: separate test set (10%)
    train_val_data, test_data = train_test_split(
        data_list,
        test_size=test_ratio,
        random_state=random_state,
        stratify=[d['y'][0] for d in data_list]
    )
    
    # Second split: divide remaining data into train (80%) and val (10%)
    # Since we already removed 10% for test, we need val_ratio / (train_ratio + val_ratio)
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=[d['y'][0] for d in train_val_data]
    )
    
    # Save all three sets
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    
    with gzip.open(train_file, 'wt', encoding='utf-8') as f:
        json.dump(train_data, f)
    
    with gzip.open(val_file, 'wt', encoding='utf-8') as f:
        json.dump(val_data, f)
    
    with gzip.open(test_file, 'wt', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    # Report results
    total = len(data_list)
    print(f"Split {total} graphs:")
    print(f"  Train: {len(train_data)} graphs ({len(train_data)/total*100:.1f}%) -> {train_file}")
    print(f"  Val:   {len(val_data)} graphs ({len(val_data)/total*100:.1f}%) -> {val_file}")
    print(f"  Test:  {len(test_data)} graphs ({len(test_data)/total*100:.1f}%) -> {test_file}")
    
    # Show label distribution
    train_labels = [d['y'][0] for d in train_data]
    val_labels = [d['y'][0] for d in val_data]
    test_labels = [d['y'][0] for d in test_data]
    
    print(f"  Train - Coeff 0: {train_labels.count(0.0)}, Coeff 1: {train_labels.count(1.0)}")
    print(f"  Val   - Coeff 0: {val_labels.count(0.0)}, Coeff 1: {val_labels.count(1.0)}")
    print(f"  Test  - Coeff 0: {test_labels.count(0.0)}, Coeff 1: {test_labels.count(1.0)}")
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("Enter CSV file path: ").strip()
    
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        sys.exit(1)
    
    try:
        train_data, val_data, test_data = csv_to_train_val_test_split(csv_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)