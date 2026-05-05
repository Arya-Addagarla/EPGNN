import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from modules.waveform_cnn import WaveformCNN
from modules.transformer import TemporalTransformer

class MultimodalGNN(nn.Module):
    def __init__(self, hidden_dim=64, use_cnn=True, use_transformer=True, use_gcn=True, use_dropout=True):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_transformer = use_transformer
        self.use_gcn = use_gcn
        self.use_dropout = use_dropout
        
        # Short-term pattern extractor
        if self.use_cnn:
            self.cnn_extractor = WaveformCNN(out_channels=hidden_dim)
        else:
            self.raw_proj = nn.LazyLinear(hidden_dim)
            
        # Long-term trend/precursor detector
        if self.use_transformer:
            self.temporal_transformer = TemporalTransformer(input_dim=hidden_dim)
            
        # Spatial relationship detector
        if self.use_gcn:
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            
        dropout_p = 0.3 if self.use_dropout else 0.0
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
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
        if self.use_cnn:
            h = self.cnn_extractor(x)
        else:
            h = self.raw_proj(x)
            
        # 2. Reshape for Transformer (treating batch as sequence for this demo)
        # In a real run, we would stack historical windows here
        if self.use_transformer:
            seq_features = h.unsqueeze(1) 
            h = self.temporal_transformer(seq_features)
            
        # 3. Process spatial graph relationships
        if self.use_gcn:
            h = self.conv1(h, edge_index)
            h = F.relu(h)
            h = self.conv2(h, edge_index)
            h = F.relu(h)
            
        graph_embed = global_mean_pool(h, batch)
        
        logits = self.clf_head(graph_embed)
        mag_pred = self.mag_head(graph_embed)
        precursor_prob = torch.sigmoid(self.precursor_head(graph_embed))
        
        return logits, mag_pred, precursor_prob
