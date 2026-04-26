import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from dataset import STEADGraphDataset
from model import MultimodalGNN
from tqdm import tqdm

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        dataset = STEADGraphDataset(metadata_path='metadata_clean.csv', hdf5_path='mock_waveforms.hdf5')
    except FileNotFoundError:
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

    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for data in tqdm(train_loader):
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
            
    torch.save(model.state_dict(), 'earthquake_gnn.pth')

if __name__ == '__main__':
    train()
