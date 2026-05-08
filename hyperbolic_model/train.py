import torch
from torch_geometric.loader import NeighborLoader
import geoopt
from dataset import load_data
from model import HGCN
from losses import calculate_total_loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data, id_map = load_data()
    data = data.to(device)
    
    # We only train on papers published before 2024
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    
    # NeighborLoader for mini-batching on large graphs
    # Samples 15 neighbors in 1st hop, 10 in 2nd hop
    train_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        input_nodes=train_idx,
        batch_size=256,
        shuffle=True,
    )

    in_features = data.x.size(1)
    hidden_features = 64
    out_features = 32
    
    model = HGCN(in_features, hidden_features, out_features, curvature=1.0).to(device)
    
    # Standard Adam won't work in curved space. We must use RiemannianAdam.
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=0.01)

    epochs = 10
    model.train()
    
    print("Starting training on pre-2024 subgraph...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            z = model(batch.x, batch.edge_index)
            
            # Since NeighborLoader creates a bipartite subgraph, 
            # we calculate loss on the sampled edges
            loss = calculate_total_loss(
                model.manifold, 
                z, 
                batch.edge_index, 
                batch.years, 
                num_nodes=batch.num_nodes
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            
        print(f"Epoch {epoch+1:02d}, Loss: {total_loss / len(train_loader.dataset):.4f}")

    # Save the model and embeddings
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), "hyperbolic_model.pt")
    
    # Get full embeddings for validation
    model.eval()
    with torch.no_grad():
        # For simplicity in this script, if graph fits in memory we do a full forward pass.
        # Otherwise, we would use NeighborLoader iteratively to generate embeddings.
        try:
            full_z = model(data.x, data.edge_index)
            torch.save(full_z.cpu(), "embeddings.pt")
        except RuntimeError:
            print("Graph too large for full forward pass. Embeddings not saved.")

if __name__ == "__main__":
    train()
