import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data.dataset import STEADGraphDataset
from models.gnn import MultimodalGNN
from losses.custom_losses import EPGNNLoss
from tqdm import tqdm
import torch.backends.cudnn as cudnn


def train_model(
    epochs=5,
    batch_size=4096,
    metadata_path='metadata_clean.csv',
    hdf5_path='mock_waveforms.hdf5'
):
    # 🔥 HARD REQUIRE CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Fix your environment (check nvidia-smi).")

    device = torch.device('cuda')
    print(f"Using device: {device}")

    cudnn.benchmark = True
    print("Enabled cuDNN autotuning for high performance.")

    # Load dataset
    try:
        dataset = STEADGraphDataset(
            metadata_path=metadata_path,
            hdf5_path=hdf5_path
        )
    except FileNotFoundError:
        print("Dataset not found. Please generate or download it first.")
        return

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # DataLoader settings
    num_workers = 16
    pin_memory = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Model
    model = MultimodalGNN(hidden_dim=64).to(device)

    # Loss + optimizer
    criterion = EPGNNLoss(mag_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data = data.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                out = model(data.x, data.edge_index, data.batch, data.pos)

                # ✅ FIX: handle arbitrary outputs
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                    mag_pred = out[1]
                else:
                    raise RuntimeError(
                        "Model must return at least (logits, mag_pred)"
                    )

                loss = criterion(logits, mag_pred, data.y, data.mag)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'earthquake_gnn.pth')
    print("Model saved to earthquake_gnn.pth")
