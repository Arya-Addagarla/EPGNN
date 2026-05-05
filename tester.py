import torch
from torch_geometric.loader import DataLoader
import numpy as np

from data.dataset import STEADGraphDataset
from models.gnn import MultimodalGNN

def load_model(model_path, device):
    model = MultimodalGNN(hidden_dim=64).to(device)
    # weights_only=True is safer for modern PyTorch
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------- LOAD DATA ----------------
    dataset = STEADGraphDataset(
        metadata_path="metadata_clean.csv",
        hdf5_path="mock_waveforms.hdf5"
    )

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0 if device.type == 'cpu' else 8 # Safer for local Mac CPU
    )

    # ---------------- LOAD MODEL ----------------
    model = load_model("earthquake_gnn.pth", device)

    all_preds = []
    all_labels = []
    mag_errors = []
    
    # Precursor metrics
    pre_probs = []
    pre_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Unpack 3 values (Logits, Magnitude, and the new Precursor Prob)
            logits, mag_pred, pre_prob = model(data.x, data.edge_index, data.batch, data.pos)

            preds = torch.argmax(logits, dim=1)

            # ---------------- classification stats ----------------
            correct += (preds == data.y).sum().item()
            total += data.y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
            # ---------------- precursor stats ----------------
            if hasattr(data, "precursor"):
                pre_probs.extend(pre_prob.squeeze().cpu().numpy())
                pre_labels.extend(data.precursor.squeeze().cpu().numpy())

            # ---------------- magnitude error ----------------
            if hasattr(data, "mag"):
                err = (mag_pred.squeeze() - data.mag.squeeze()).abs()
                mag_errors.extend(err.cpu().numpy())

    # ---------------- FINAL STATS ----------------
    acc = correct / total

    print("\n================ TEST RESULTS ================")
    print(f"Total samples tested: {total}")
    print(f"Event Detection Accuracy: {acc:.4f}")

    if len(pre_probs) > 0:
        pre_probs_np = np.array(pre_probs)
        pre_labels_np = np.array(pre_labels)
        # Prediction counts as "Success" if prob > 0.5
        pre_acc = np.mean((pre_probs_np > 0.5) == pre_labels_np)
        print(f"Precursor Forecast Accuracy: {pre_acc:.4f} (Minutes to Hours warning)")

    if len(mag_errors) > 0:
        mag_errors = torch.tensor(mag_errors)
        print(f"Magnitude MAE: {mag_errors.mean().item():.4f}")
        print(f"Magnitude Std: {mag_errors.std().item():.4f}")

    # ---------------- COLLAPSE CHECK ----------------
    unique_preds = set(all_preds)
    print("\nUnique predicted classes:", unique_preds)

    if len(unique_preds) == 1:
        print("⚠️ WARNING: Model is collapsed (predicting single class only)")

    print("=============================================")

if __name__ == "__main__":
    run_test()
