'''
Author: Qi7
Date: 2023-04-06 21:32:59
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-06 22:05:19
Description: deep learning models definition
'''
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, seq_num, num_class):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size, 
            hidden_size = 256,
            num_layers = 1,
            batch_first = True
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, num_class)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x