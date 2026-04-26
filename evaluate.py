import torch
from torch_geometric.loader import DataLoader
from dataset import STEADGraphDataset
from model import MultimodalGNN
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
import numpy as np

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = STEADGraphDataset(metadata_path='metadata_clean.csv', hdf5_path='mock_waveforms.hdf5')
    
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = MultimodalGNN(hidden_dim=64).to(device)
    try:
        model.load_state_dict(torch.load('earthquake_gnn.pth', map_location=device, weights_only=True))
    except Exception as e:
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    all_mag_preds = []
    all_mag_true = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            logits, mag_pred = model(data.x, data.edge_index, data.batch, data.pos)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
            mask = data.y.view(-1) == 1
            if mask.sum() > 0:
                all_mag_preds.extend(mag_pred[mask].squeeze(-1).cpu().numpy())
                all_mag_true.extend(data.mag[mask].squeeze(-1).cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1-Score: {f1:.4f}")
    
    if len(all_mag_true) > 0:
        mse = mean_squared_error(all_mag_true, all_mag_preds)
        print(f"Magnitude MSE: {mse:.4f}")
        
if __name__ == '__main__':
    evaluate()
