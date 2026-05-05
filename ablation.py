"""
EPGNN Ablation Suite
====================
Tests the contribution of each architectural component by removing or
replacing them one at a time and comparing against the full model.

Ablation variants:
  1. Full model           — baseline (all components active)
  2. No CNN               — replace WaveformCNN with a linear projection
  3. No Transformer       — bypass TemporalTransformer (identity pass-through)
  4. No GCN               — skip both GCN layers (raw pooling of transformer output)
  5. No Dropout           — remove Dropout from the classifier head
  6. No CNN + No Transformer — raw linear features straight into GCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from data.dataset import STEADGraphDataset
from modules.waveform_cnn import WaveformCNN
from modules.transformer import TemporalTransformer

# ─────────────────────────────────────────────
# Ablated model variants
# ─────────────────────────────────────────────

class AblatedGNN(nn.Module):
    """
    Configurable EPGNN where each major component can be toggled off.
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        use_cnn: bool = True,
        use_transformer: bool = True,
        use_gcn: bool = True,
        use_dropout: bool = True,
    ):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_transformer = use_transformer
        self.use_gcn = use_gcn

        # --- Feature extractor ---
        if use_cnn:
            self.cnn_extractor = WaveformCNN(out_channels=hidden_dim)
        else:
            # Fallback: project raw flattened signal to hidden_dim
            self.raw_proj = nn.LazyLinear(hidden_dim)

        # --- Temporal module ---
        if use_transformer:
            self.temporal_transformer = TemporalTransformer(input_dim=hidden_dim)

        # --- Spatial module ---
        if use_gcn:
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # --- Heads ---
        dropout_p = 0.3 if use_dropout else 0.0
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.precursor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.mag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, batch, pos=None):
        # 1. Feature extraction
        if self.use_cnn:
            h = self.cnn_extractor(x)
        else:
            h = self.raw_proj(x)            # [N, hidden_dim]

        # 2. Temporal modelling
        if self.use_transformer:
            h = self.temporal_transformer(h.unsqueeze(1))  # [N, hidden_dim]

        # 3. Spatial / graph convolutions
        if self.use_gcn:
            h = F.relu(self.conv1(h, edge_index))
            h = F.relu(self.conv2(h, edge_index))

        graph_embed = global_mean_pool(h, batch)          # [B, hidden_dim]

        logits         = self.clf_head(graph_embed)
        mag_pred       = self.mag_head(graph_embed)
        precursor_prob = torch.sigmoid(self.precursor_head(graph_embed))

        return logits, mag_pred, precursor_prob


# ─────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────

@dataclass
class AblationResult:
    name: str
    accuracy: float
    precursor_acc: Optional[float]
    mag_mae: Optional[float]
    mag_std: Optional[float]
    collapsed: bool
    n_params: int

    def __str__(self):
        pre = f"{self.precursor_acc:.4f}" if self.precursor_acc is not None else "N/A"
        mae = f"{self.mag_mae:.4f}"       if self.mag_mae       is not None else "N/A"
        std = f"{self.mag_std:.4f}"       if self.mag_std       is not None else "N/A"
        col = "⚠️  YES" if self.collapsed else "No"
        return (
            f"  Name             : {self.name}\n"
            f"  Params           : {self.n_params:,}\n"
            f"  Detection Acc    : {self.accuracy:.4f}\n"
            f"  Precursor Acc    : {pre}\n"
            f"  Magnitude MAE    : {mae}\n"
            f"  Magnitude Std    : {std}\n"
            f"  Collapsed        : {col}"
        )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> AblationResult:
    model.eval()
    all_preds, all_labels = [], []
    mag_errors, pre_probs, pre_labels = [], [], []
    correct = total = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            try:
                logits, mag_pred, pre_prob = model(data.x, data.edge_index, data.batch, data.pos)
            except Exception as e:
                raise RuntimeError(f"Forward pass failed: {e}") from e

            preds = torch.argmax(logits, dim=1)
            correct += (preds == data.y).sum().item()
            total   += data.y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

            if hasattr(data, "precursor"):
                pre_probs.extend(pre_prob.squeeze().cpu().numpy())
                pre_labels.extend(data.precursor.squeeze().cpu().numpy())

            if hasattr(data, "mag"):
                err = (mag_pred.squeeze() - data.mag.squeeze()).abs()
                mag_errors.extend(err.cpu().numpy())

    acc = correct / total

    pre_acc = None
    if len(pre_probs) > 0:
        p = np.array(pre_probs)
        l = np.array(pre_labels)
        pre_acc = float(np.mean((p > 0.5) == l))

    mag_mae = mag_std = None
    if len(mag_errors) > 0:
        t = torch.tensor(mag_errors)
        mag_mae = t.mean().item()
        mag_std = t.std().item()

    unique = set(all_preds)
    collapsed = len(unique) == 1

    return AblationResult(
        name="",
        accuracy=acc,
        precursor_acc=pre_acc,
        mag_mae=mag_mae,
        mag_std=mag_std,
        collapsed=collapsed,
        n_params=count_params(model),
    )


# ─────────────────────────────────────────────
# Ablation configurations
# ─────────────────────────────────────────────

ABLATIONS = [
    # (display_name, use_cnn, use_transformer, use_gcn, use_dropout)
    ("Full Model (baseline)",         True,  True,  True,  True),
    ("No CNN",                        False, True,  True,  True),
    ("No Transformer",                True,  False, True,  True),
    ("No GCN",                        True,  True,  False, True),
    ("No Dropout",                    True,  True,  True,  False),
    ("No CNN + No Transformer",       False, False, True,  True),
]

MODEL_WEIGHTS = "earthquake_gnn.pth"
HIDDEN_DIM    = 64


def run_ablations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load dataset once
    dataset = STEADGraphDataset(
        metadata_path="metadata_clean.csv",
        hdf5_path="mock_waveforms.hdf5",
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

    results: List[AblationResult] = []

    for name, use_cnn, use_tx, use_gcn, use_drop in ABLATIONS:
        print(f"⏳  Running: {name} ...")
        model = AblatedGNN(
            hidden_dim=HIDDEN_DIM,
            use_cnn=use_cnn,
            use_transformer=use_tx,
            use_gcn=use_gcn,
            use_dropout=use_drop,
        ).to(device)

        # ── Load pretrained weights for matching keys only ──────────────
        # Keys that don't exist in this variant are simply skipped so we
        # measure the effect of architectural removal, not random init.
        state_dict = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"    ↳ Skipped (not in variant): {missing}")

        result = evaluate(model, loader, device)
        result.name = name
        results.append(result)
        print(f"    ↳ Done. Acc={result.accuracy:.4f}")

    # ─────────────────────────────────────────────
    # Print final table
    # ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  EPGNN ABLATION RESULTS")
    print("═" * 60)

    baseline = results[0]
    for r in results:
        delta = r.accuracy - baseline.accuracy
        sign  = "+" if delta >= 0 else ""
        tag   = "  [baseline]" if r.name == baseline.name else f"  [Δ acc {sign}{delta:.4f}]"
        print(f"\n── {r.name}{tag}")
        print(r)

    print("\n" + "═" * 60)
    print("  RANKING BY DETECTION ACCURACY")
    print("═" * 60)
    for i, r in enumerate(sorted(results, key=lambda x: x.accuracy, reverse=True), 1):
        bar_len = int(r.accuracy * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {i}. [{bar}] {r.accuracy:.4f}  {r.name}")

    print()


if __name__ == "__main__":
    run_ablations()
