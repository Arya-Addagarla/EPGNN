import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class WaveformCNN(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return x

class MultimodalGNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.feature_extractor = WaveformCNN(out_channels=hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.mag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch, pos=None):
        node_features = self.feature_extractor(x)
        
        h = self.conv1(node_features, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        graph_embed = global_mean_pool(h, batch)
        
        logits = self.clf_head(graph_embed)
        mag_pred = self.mag_head(graph_embed)
        
        return logits, mag_pred
