import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data.dataset import STEADGraphDataset
from models.gnn import MultimodalGNN
from losses.custom_losses import EPGNNLoss
from tqdm import tqdm


def train_model(
    epochs=5,
    batch_size=4096,
    metadata_path='metadata_clean.csv',
    hdf5_path='mock_waveforms.hdf5'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------------- DATASET ----------------
    dataset = STEADGraphDataset(
        metadata_path=metadata_path,
        hdf5_path=hdf5_path
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    # ---------------- MODEL ----------------
    model = MultimodalGNN(hidden_dim=64).to(device)

    criterion = EPGNNLoss(mag_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ---------------- TRAIN ----------------
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            data = data.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(data.x, data.edge_index, data.batch, data.pos)

                logits, mag_pred = out[0], out[1]

                loss = criterion(logits, mag_pred, data.y, data.mag)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0

        correct = 0
        total = 0
        mag_error = 0.0

        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                data = data.to(device)

                out = model(data.x, data.edge_index, data.batch, data.pos)
                logits, mag_pred = out[0], out[1]

                loss = criterion(logits, mag_pred, data.y, data.mag)
                val_loss += loss.item()

                # ---------------- CLASSIFICATION ACCURACY ----------------
                preds = torch.argmax(logits, dim=1)
                correct += (preds == data.y).sum().item()
                total += data.y.size(0)

                # ---------------- MAGNITUDE ERROR ----------------
                mag_error += torch.abs(mag_pred - data.mag).sum().item()

        # ---------------- METRICS ----------------
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        accuracy = correct / total if total > 0 else 0
        mae_mag = mag_error / total if total > 0 else 0

        print(f"""
Epoch {epoch+1}
--------------------------------
Train Loss: {train_loss:.4f}
Val Loss:   {val_loss:.4f}
Val Acc:    {accuracy:.4f}
Mag MAE:    {mae_mag:.4f}
--------------------------------
""")

    torch.save(model.state_dict(), 'earthquake_gnn.pth')
    print("Model saved to earthquake_gnn.pth")
