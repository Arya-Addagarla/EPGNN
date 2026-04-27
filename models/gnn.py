import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from modules.waveform_cnn import WaveformCNN
from modules.transformer import TemporalTransformer

class MultimodalGNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Short-term pattern extractor
        self.cnn_extractor = WaveformCNN(out_channels=hidden_dim)
        
        # Long-term trend/precursor detector
        self.temporal_transformer = TemporalTransformer(input_dim=hidden_dim)
        
        # Spatial relationship detector
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2) # Detection
        )
        
        self.precursor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Probability of event in next hour
        )
        
        self.mag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Magnitude estimation
        )

    def forward(self, x, edge_index, batch, pos=None):
        # 1. Extract features from waveforms
        cnn_features = self.cnn_extractor(x)
        
        # 2. Reshape for Transformer (treating batch as sequence for this demo)
        # In a real run, we would stack historical windows here
        seq_features = cnn_features.unsqueeze(1) 
        trend_features = self.temporal_transformer(seq_features)
        
        # 3. Process spatial graph relationships
        h = self.conv1(trend_features, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        graph_embed = global_mean_pool(h, batch)
        
        logits = self.clf_head(graph_embed)
        mag_pred = self.mag_head(graph_embed)
        precursor_prob = torch.sigmoid(self.precursor_head(graph_embed))
        
        return logits, mag_pred, precursor_prob
