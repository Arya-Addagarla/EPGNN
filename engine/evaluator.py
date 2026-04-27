import torch
from torch_geometric.loader import DataLoader
from data.dataset import STEADGraphDataset
from models.gnn import MultimodalGNN
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error


def evaluate_model(
    batch_size=4096,
    metadata_path='metadata_clean.csv',
    hdf5_path='mock_waveforms.hdf5',
    model_path='earthquake_gnn.pth'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- DATASET ----------------
    try:
        dataset = STEADGraphDataset(
            metadata_path=metadata_path,
            hdf5_path=hdf5_path
        )
    except FileNotFoundError:
        print("Dataset not found.")
        return

    num_workers = 16 if device.type == 'cuda' else 0

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda'
    )

    # ---------------- MODEL ----------------
    model = MultimodalGNN(hidden_dim=64).to(device)

    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    except Exception as e:
        print(f"Could not load model weights from {model_path}: {e}")
        return

    model.eval()

    # ---------------- STORAGE ----------------
    all_preds = []
    all_labels = []
    all_mag_preds = []
    all_mag_true = []

    # ---------------- EVAL LOOP ----------------
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)

            # 🔥 FIX: robust unpacking (CRITICAL BUG FIX)
            out = model(data.x, data.edge_index, data.batch, data.pos)

            logits = out[0]
            mag_pred = out[1]

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

            # ---------------- MAGNITUDE ----------------
            # safer: use graph-level mask if batched
            if hasattr(data, "mag"):
                all_mag_preds.extend(mag_pred.squeeze(-1).cpu().numpy())
                all_mag_true.extend(data.mag.squeeze(-1).cpu().numpy())

    # ---------------- METRICS ----------------
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"F1-Score: {f1:.4f}")

    if len(all_mag_true) > 0:
        mse = mean_squared_error(all_mag_true, all_mag_preds)
        print(f"Magnitude MSE: {mse:.4f}")

    print("--------------------------")
