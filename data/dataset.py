import os
import torch
import pandas as pd
import numpy as np
import h5py
from torch_geometric.data import Dataset, Data

class STEADGraphDataset(Dataset):
    def __init__(self, metadata_path, hdf5_path, root='.', transform=None, pre_transform=None):
        self.metadata_path = metadata_path
        self.hdf5_path = hdf5_path
        
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
        else:
            raise FileNotFoundError(f"Metadata file {metadata_path} not found.")
            
        super().__init__(root, transform, pre_transform)
        
        self.edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1]
        ], dtype=torch.long)

    def len(self):
        return len(self.metadata)

    def get(self, idx):
        row = self.metadata.iloc[idx]
        trace_name = row['trace_name']
        label = row['label']
        
        with h5py.File(self.hdf5_path, 'r') as f:
            if f"earthquake/{trace_name}" in f:
                waveform = f[f"earthquake/{trace_name}"][:]
            elif f"non_earthquake/{trace_name}" in f:
                waveform = f[f"non_earthquake/{trace_name}"][:]
            else:
                waveform = f[trace_name][:]
                
        x = torch.tensor(waveform.T, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        mag = torch.tensor([row['source_magnitude'] if not pd.isna(row['source_magnitude']) else 0.0], dtype=torch.float)
        pos = torch.tensor([[row['receiver_latitude'], row['receiver_longitude']]], dtype=torch.float)
        precursor = torch.tensor([row['precursor']], dtype=torch.float)
        
        data = Data(x=x, edge_index=self.edge_index, y=y, pos=pos, mag=mag, precursor=precursor)
        
        return data

if __name__ == '__main__':
    ds = STEADGraphDataset(metadata_path='metadata_clean.csv', hdf5_path='mock_waveforms.hdf5')
