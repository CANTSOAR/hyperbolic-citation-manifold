import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(data_dir="../data", max_features=128):
    print("Loading nodes and edges...")
    nodes_path = os.path.join(data_dir, "nodes.csv")
    edges_path = os.path.join(data_dir, "edges.csv")
    
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        raise FileNotFoundError("Parsed nodes or edges not found. Please run the rust parser first.")
        
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    # Map IDs to contiguous integer indices
    unique_ids = nodes_df['id'].unique()
    id_map = {id_str: idx for idx, id_str in enumerate(unique_ids)}
    
    print("Computing TF-IDF features from abstracts...")
    nodes_df['abstract'] = nodes_df['abstract'].fillna("")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    features = vectorizer.fit_transform(nodes_df['abstract']).toarray()
    
    x = torch.tensor(features, dtype=torch.float32)
    
    print("Constructing edge index...")
    # Filter edges where both source and target exist in our nodes
    valid_edges = edges_df[edges_df['source'].isin(id_map) & edges_df['target'].isin(id_map)]
    source_idx = valid_edges['source'].map(id_map).values
    target_idx = valid_edges['target'].map(id_map).values
    
    edge_index = torch.tensor(np.vstack((source_idx, target_idx)), dtype=torch.long)
    
    # Parse dates for train/test split (pre-2024 vs 2024+)
    # Update date format is generally YYYY-MM-DD
    # Some older formats might be different but usually start with year
    nodes_df['year'] = nodes_df['date'].str[:4].astype(float)
    
    # Pre-2024
    train_mask = torch.tensor((nodes_df['year'] < 2024).values, dtype=torch.bool)
    test_mask = torch.tensor((nodes_df['year'] >= 2024).values, dtype=torch.bool)
    
    # The actual date/time string is kept for reference
    data = Data(x=x, edge_index=edge_index)
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.node_ids = unique_ids # Keep original string IDs for evaluation lookup
    data.years = torch.tensor(nodes_df['year'].fillna(0).values, dtype=torch.float32)
    
    print(f"Graph constructed: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Train nodes (pre-2024): {train_mask.sum().item()}")
    print(f"Test nodes (2024+): {test_mask.sum().item()}")
    
    return data, id_map
