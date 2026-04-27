import torch
from torch_geometric.loader import DataLoader

from data.dataset import STEADGraphDataset
from models.gnn import MultimodalGNN


def load_model(model_path, device):
    model = MultimodalGNN(hidden_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
        num_workers=8
    )

    # ---------------- LOAD MODEL ----------------
    model = load_model("earthquake_gnn.pth", device)

    all_preds = []
    all_labels = []
    mag_errors = []

    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch, data.pos)
            logits, mag_pred = out[0], out[1]

            preds = torch.argmax(logits, dim=1)

            # ---------------- classification stats ----------------
            correct += (preds == data.y).sum().item()
            total += data.y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

            # ---------------- magnitude error ----------------
            if hasattr(data, "mag"):
                err = (mag_pred.squeeze() - data.mag.squeeze()).abs()
                mag_errors.extend(err.cpu().numpy())

            # ---------------- DEBUG SAMPLE ----------------
            if total < 50:
                print("\nSample predictions:")
                print("Preds :", preds[:10].cpu().numpy())
                print("Labels:", data.y[:10].cpu().numpy())

    # ---------------- FINAL STATS ----------------
    acc = correct / total

    print("\n================ TEST RESULTS ================")
    print(f"Accuracy: {acc:.4f}")
    print(f"Total samples: {total}")

    if len(mag_errors) > 0:
        mag_errors = torch.tensor(mag_errors)
        print(f"Mag MAE: {mag_errors.mean().item():.4f}")
        print(f"Mag Std: {mag_errors.std().item():.4f}")
        print(f"Max Mag Error: {mag_errors.max().item():.4f}")

    # ---------------- COLLAPSE CHECK ----------------
    unique_preds = set(all_preds)
    print("\nUnique predicted classes:", unique_preds)

    if len(unique_preds) == 1:
        print("⚠️ WARNING: Model is collapsed (predicting single class only)")

    print("=============================================")


if __name__ == "__main__":
    run_test()
