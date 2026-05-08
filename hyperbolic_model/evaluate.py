import torch
import geoopt
import pandas as pd
from dataset import load_data
from model import HGCN

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load data
    data, id_map = load_data()
    data = data.to(device)

    in_features = data.x.size(1)
    hidden_features = 64
    out_features = 32
    
    model = HGCN(in_features, hidden_features, out_features, curvature=1.0).to(device)
    try:
        model.load_state_dict(torch.load("hyperbolic_model.pt", map_location=device))
        print("Loaded trained model weights.")
    except FileNotFoundError:
        print("Warning: Model weights not found. Using untrained model for demonstration.")
        
    model.eval()
    
    print("Computing embeddings for the whole graph...")
    with torch.no_grad():
        # In a production environment with a 2M node graph, you would use 
        # NeighborLoader to compute this iteratively. 
        # For our local script, we attempt a full forward pass.
        z = model(data.x, data.edge_index)
        
    # The Time Machine Test: Evaluate only on test papers (2024+)
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    
    if len(test_idx) == 0:
        print("No test nodes (2024+) found in the dataset.")
        return
        
    print(f"Evaluating link prediction for {len(test_idx)} new (2024+) papers...")
    
    # We will test if the model can predict citations from new papers to old papers.
    # For a given new paper, its actual citations are the edges originating from it.
    
    row, col = data.edge_index
    hits_at_10 = 0
    total_queries = 0
    
    # To save time, we evaluate on a random sample of 100 test nodes
    sample_size = min(100, len(test_idx))
    eval_nodes = test_idx[torch.randperm(len(test_idx))[:sample_size]]
    
    for i, src_node in enumerate(eval_nodes):
        # Find true citations (targets) for this source node
        mask = row == src_node
        true_targets = col[mask]
        
        # Only evaluate if it cites at least one old paper (pre-2024)
        old_true_targets = [t.item() for t in true_targets if t.item() in train_idx]
        if not old_true_targets:
            continue
            
        # Calculate hyperbolic distance to all old (pre-2024) papers
        src_emb = z[src_node].unsqueeze(0) # [1, dim]
        old_emb = z[train_idx]             # [num_old, dim]
        
        # Calculate distances
        distances = model.manifold.dist(src_emb, old_emb)
        
        # Get top 10 closest old papers
        _, top10_indices = torch.topk(distances, k=10, largest=False)
        top10_old_nodes = train_idx[top10_indices].tolist()
        
        # Check if any true target is in the top 10
        hit = any(t in top10_old_nodes for t in old_true_targets)
        if hit:
            hits_at_10 += 1
            
        total_queries += 1
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{sample_size} nodes...")
            
    if total_queries > 0:
        accuracy = hits_at_10 / total_queries
        print(f"Hits@10 for new papers predicting old paradigms: {accuracy:.4f}")
        print("If the model successfully predicted citations purely based on the 'voids' it fell into, it has achieved its structural inductive bias.")
    else:
        print("No valid queries found (test nodes didn't cite any train nodes).")

if __name__ == "__main__":
    evaluate()
