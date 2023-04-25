'''
Author: Qi7
Date: 2023-04-06 21:32:59
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-25 17:08:11
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
        self.linear = nn.Linear(512, self.num_class)
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

class ResBlock(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, 
                 kernel_size, stride, n_pad, dropout=0.2):
        super(ResBlock, self).__init__()
        self.batchnorm1 = nn.BatchNorm1d(n_input_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(n_input_channels, n_output_channels, kernel_size,
                               stride = stride, padding = n_pad)
        self.batchnorm2 = nn.BatchNorm1d(n_output_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_output_channels, n_output_channels, kernel_size,
                               stride = 1, padding = 'same')
        self.max_pooling = nn.MaxPool1d(stride)

    def forward(self, x):
        out = self.batchnorm1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv1(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.conv2(out)
        if out.shape != x.shape:
            # if x.shape[-1] != out.shape[-1]:
            #     x = F.pad(x, pad = (1,1))
            x = self.max_pooling(x)
        out = out + x
        return out

class QNN(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, 
                 kernel_size, stride, n_classes, dropout=0.5):
        super(QNN, self).__init__()
        self.conv1 = nn.Conv1d(n_input_channels, n_output_channels, kernel_size = 7,
                               stride = stride, padding = 'same')
        self.batchnorm1 = nn.BatchNorm1d(n_output_channels)
        self.relu1 = nn.ReLU()
        self.max_pooling = nn.MaxPool1d(2)
        self.resblock1 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 1, 
                                 dropout=dropout)
        self.resblock2 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 0, 
                                 dropout=dropout)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(384*2),n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.max_pooling(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        
    def forward(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        query_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeld support data.
        """
        z_support = self.backbone.forward(support_data)
        z_query = self.bakbone.forward(query_data)
        
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all support features vector with label i
        