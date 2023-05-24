'''
Author: Qi7
Date: 2023-04-07 10:41:35
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-23 17:36:53
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
    

class SiameseDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __getitem__(self, index):
        sample1, target1 = self.data[index], self.targets[index]
        positive_indices = torch.where(self.targets == target1)[0]
        negative_indices = torch.where(self.targets != target1)[0]
        sample2 = self.data[positive_indices[torch.randint(len(positive_indices), (1,))]][0]
        sample3 = self.data[negative_indices[torch.randint(len(negative_indices), (1,))]][0]
        return sample1, sample2, sample3, torch.Tensor([1]), torch.Tensor([0])
    
    def __len__(self):
        return len(self.data)