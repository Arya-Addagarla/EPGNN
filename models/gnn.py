import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from modules.waveform_cnn import WaveformCNN

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
