'''
Author: Qi7
Date: 2023-04-06 21:32:59
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-08 13:10:48
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
        self.num_class = num_class
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, self.num_class)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        outputs = self.linear(self.dropout(x))
        # multiclass classification
        if self.num_class > 1:
            outputs = self.sigmoid(outputs)
        return outputs