#!/usr/bin/env python3
# CSV to JSON with Train/Val/Test Split + Node Features

import pandas as pd
import json
import ast
import gzip
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Directory where .npy node features are stored
FEATURE_DIR = ".dataset_src/features_loop_9"
FEATURE_NAMES = ["closeness", "degree", "betweenness", "clustering", "pagerank", "face_count"]

def load_features(num_graphs):
    """Load all node feature .npy files into a list of arrays"""
    feature_arrays = {name: np.load(f"{FEATURE_DIR}/{name}.npy", allow_pickle=True) for name in FEATURE_NAMES}
    # Each entry in features_per_graph[i] will be (num_nodes_i, num_features)
    features_per_graph = []
    for i in range(num_graphs):
        feats = []
        for name in FEATURE_NAMES:
            f = feature_arrays[name][i]
            if f.ndim == 1:
                f = np.expand_dims(f, -1)
            feats.append(f)
        feats_concat = np.concatenate(feats, axis=1)
        features_per_graph.append(feats_concat)
    return features_per_graph

def csv_to_train_val_test_split(csv_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    df = pd.read_csv(csv_file)
    num_graphs = len(df)
    features_per_graph = load_features(num_graphs)

    data_list = []

    for idx, row in df.iterrows():
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

        # Use the loaded node features
        x_feat = features_per_graph[idx]
        # Ensure the number of nodes matches
        if x_feat.shape[0] != len(nodes):
            raise ValueError(f"Graph {idx}: Node feature length {x_feat.shape[0]} != number of nodes {len(nodes)}")

        data_dict = {
            'x': x_feat.tolist(),
            'edge_index': edge_index,
            'edge_attr': [0] * len(edge_pairs),
            'y': [float(label)]
        }
        data_list.append(data_dict)

    # Perform train/val/test split (same as original)
    train_val_data, test_data = train_test_split(
        data_list,
        test_size=test_ratio,
        random_state=random_state,
        stratify=[d['y'][0] for d in data_list]
    )

    val_size_adjusted = val_ratio / (train_ratio + val_ratio)

    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=[d['y'][0] for d in train_val_data]
    )

    # Save to JSON.gz
    base_name = csv_file.replace('.csv', '')
    os.makedirs(os.path.dirname(base_name), exist_ok=True)

    with gzip.open(f"{base_name}_feature_train.json.gz", 'wt', encoding='utf-8') as f:
        json.dump(train_data, f)
    with gzip.open(f"{base_name}_feature_val.json.gz", 'wt', encoding='utf-8') as f:
        json.dump(val_data, f)
    with gzip.open(f"{base_name}_feature_test.json.gz", 'wt', encoding='utf-8') as f:
        json.dump(test_data, f)

    print(f"✅ Converted {num_graphs} graphs with node features to JSON.gz files")
    return train_data, val_data, test_data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("Enter CSV file path: ").strip()

    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        sys.exit(1)

    csv_to_train_val_test_split(csv_file)
