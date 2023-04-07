'''
Author: Qi7
Date: 2023-04-07 10:41:35
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-07 11:12:58
Description: dataloader definition
'''
from torch.utils.data import Dataset
import torch

class waveformDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        features = self.X[idx]
        target = self.y[idx]
        return features, target