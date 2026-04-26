import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data.dataset import STEADGraphDataset
from models.gnn import MultimodalGNN
from tqdm import tqdm

def train_model(epochs=5, metadata_path='metadata_clean.csv', hdf5_path='mock_waveforms.hdf5'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        dataset = STEADGraphDataset(metadata_path=metadata_path, hdf5_path=hdf5_path)
    except FileNotFoundError:
        print("Dataset not found. Please generate or download it first.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = MultimodalGNN(hidden_dim=64).to(device)

    clf_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data = data.to(device)
            optimizer.zero_grad()
            
            logits, mag_pred = model(data.x, data.edge_index, data.batch, data.pos)
            
            loss_clf = clf_criterion(logits, data.y.view(-1))
            
            mask = data.y.view(-1) == 1
            if mask.sum() > 0:
                loss_mag = reg_criterion(mag_pred[mask].squeeze(-1), data.mag[mask].squeeze(-1))
                loss = loss_clf + 0.1 * loss_mag
            else:
                loss = loss_clf
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f}")
            
    torch.save(model.state_dict(), 'earthquake_gnn.pth')
    print("Model saved to earthquake_gnn.pth")
