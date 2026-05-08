import torch
import torch.nn.functional as F

def contrastive_graph_loss(manifold, z, edge_index, num_nodes, margin=1.0):
    """
    Minimizes hyperbolic distance between connected nodes (positive pairs)
    and maximizes distance between randomly sampled unconnected nodes (negative pairs).
    """
    row, col = edge_index
    
    # Positive pairs distance
    pos_dist = manifold.dist(z[row], z[col])
    pos_loss = torch.mean(pos_dist ** 2)
    
    # Negative pairs (randomly sample negative edges)
    # For a simple approximation, we randomly permute the targets
    neg_col = torch.randperm(num_nodes)[:len(col)]
    neg_dist = manifold.dist(z[row], z[neg_col])
    
    # The rubber band: Push apart up to a certain margin
    neg_loss = torch.mean(F.relu(margin - neg_dist) ** 2)
    
    return pos_loss + neg_loss

def entailment_cone_loss(manifold, z, edge_index, years):
    """
    Enforces hierarchical structure. 
    If Paper A (source) cites Paper B (target), and Paper A is newer,
    we penalize if the norm of B is greater than the norm of A.
    This forces older, highly cited papers inward (smaller norm).
    """
    row, col = edge_index
    
    # Get years for source and target
    year_A = years[row]
    year_B = years[col]
    
    # Mask for edges where A is strictly newer than B
    newer_mask = year_A > year_B
    
    if not newer_mask.any():
        return torch.tensor(0.0, device=z.device)
        
    z_A = z[row][newer_mask]
    z_B = z[col][newer_mask]
    
    # Norms in Poincaré ball can be approximated by Euclidean norms
    # since distance from origin increases monotonically with Euclidean norm
    norm_A = torch.norm(z_A, dim=-1)
    norm_B = torch.norm(z_B, dim=-1)
    
    # Penalize if ||B|| > ||A|| (norm of older target > norm of newer source)
    # Using ReLU to only apply loss when the constraint is violated
    cone_loss = torch.mean(F.relu(norm_B - norm_A + 1e-4) ** 2)
    
    return cone_loss

def calculate_total_loss(manifold, z, edge_index, years, num_nodes):
    loss_contrastive = contrastive_graph_loss(manifold, z, edge_index, num_nodes)
    loss_cone = entailment_cone_loss(manifold, z, edge_index, years)
    
    # Note: As specified, we explicitly omit any global uniformity loss 
    # to allow the hyperbolic space to naturally tear open voids.
    
    return loss_contrastive + loss_cone
