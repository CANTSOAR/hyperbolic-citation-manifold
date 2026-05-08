import torch
import torch.nn as nn
import geoopt
from torch_geometric.utils import scatter

class HGCN(nn.Module):
    """
    Hyperbolic Graph Convolutional Network.
    Approximates Fréchet Mean by mapping to the origin's tangent space,
    performing Euclidean neighborhood aggregation, and projecting back via expmap0.
    """
    def __init__(self, in_features, hidden_features, out_features, curvature=1.0):
        super(HGCN, self).__init__()
        self.manifold = geoopt.PoincareBall(c=curvature)
        
        # Feature transformation layers
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x, edge_index):
        """
        x: [num_nodes, in_features] - Euclidean features
        edge_index: [2, num_edges] - Graph connectivity
        """
        # 1. Transform features in Euclidean space
        h = self.linear1(x)
        h = torch.relu(h)
        
        # 2. Project to Hyperbolic Space (Poincaré Ball)
        h_hyp = self.manifold.expmap0(h)
        
        # 3. Message Passing (The Springs)
        # Map back to tangent space at origin for approximate Fréchet Mean
        h_tan = self.manifold.logmap0(h_hyp)
        
        row, col = edge_index # row: source, col: target
        # Aggregate features from neighbors (mean)
        agg = scatter(h_tan[row], col, dim=0, dim_size=h_tan.size(0), reduce='mean')
        
        # Combine node features with aggregated neighbor features
        h_tan_new = self.linear2(h_tan + agg)
        
        # 4. Project back to Hyperbolic Space
        h_hyp_new = self.manifold.expmap0(h_tan_new)
        
        return h_hyp_new
