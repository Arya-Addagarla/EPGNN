import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=64, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        # Positional encoding to give the model a sense of "time"
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, input_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: [batch, sequence_length, features]
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        # Return the summary of the whole sequence (global trend)
        return torch.mean(x, dim=1)
