#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 2 19:44:13 2023

@author: Yingjian

Description: Yingjian's code for referenceing the Resnet CNN.
"""

import torch.nn as nn
from torch.nn.utils import weight_norm
import torch
import torch.nn.functional as F
import numpy as np

def cal_n_pad(in_size, out_size, kernel_size, stride):
    n_pad = np.ceil(((out_size - 1) * stride + kernel_size - in_size)/2)
    return int(n_pad)

def Receptive_feild(kernel_size, stride, d = [1]):
    n_layers = len(stride)
    s_n = 0
    s_i_mul = 1
    if len(d) == 1:
        d = d * n_layers
    for i in range(n_layers):
        s_i_mul = s_i_mul * stride[0] * d[i]
        s_n += s_i_mul
    RF = kernel_size * s_n + 1
    return RF

def padding_cal(x_length, kernel_size, stride, dilation, target_len):
    #initialize data
    x = np.ones(x_length)
    #initialize kernel
    conv_filter = np.ones(kernel_size)
    #perfrom convolution
    i = 0
    res = []
    current_id = i
    while((i +  (kernel_size - 1) *  dilation )< x_length):
        new_x = np.copy(conv_filter)
        for j in range(kernel_size):
            if j == 0:
                current_id = i
            else:
                current_id += dilation
            new_x[j] = x[current_id] *  conv_filter[j]
        res.append(new_x)
        i += stride
    res_len = len(res)
    if not res_len:
        current_id = i +  (kernel_size - 1) *  dilation 
        res_len = 1
    needed_n = target_len - res_len
    n_padding = stride * needed_n + current_id - (x_length - 1)
    return n_padding

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
        self.fc = nn.Linear(12,n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.max_pooling(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.fc(x)
        return x
    
class AF_Net(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, 
                 kernel_size, stride, n_classes, dropout=0.2):
        super(AF_Net, self).__init__()
        self.conv1 = nn.Conv1d(n_input_channels, n_output_channels, kernel_size,
                               stride = stride, padding = 4)
        self.batchnorm1 = nn.BatchNorm1d(n_output_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_output_channels, n_output_channels, kernel_size,
                               stride = stride, padding = 4)
        self.batchnorm2 = nn.BatchNorm1d(n_output_channels)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(n_output_channels, n_output_channels, kernel_size,
                               stride = stride, padding = 4)

        self.resblock1 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 2, n_pad = 4, dropout=dropout)
        self.resblock2 = ResBlock(n_output_channels, n_output_channels, 
                                 kernel_size, stride = 1, n_pad = 4, dropout=dropout)
        self.resblock3 = ResBlock(n_output_channels, n_output_channels, 
                                  kernel_size, stride = 2, n_pad = 4, dropout=dropout)
        self.resblock4 = ResBlock(n_output_channels, n_output_channels, 
                                  kernel_size, stride = 1, n_pad = 4, dropout=dropout)
        self.resblock5 = ResBlock(n_output_channels, n_output_channels, 
                                  kernel_size, stride = 2, n_pad = 3, dropout=dropout)
        self.resblock6 = ResBlock(n_output_channels, n_output_channels, 
                                  kernel_size, stride = 1, n_pad = 4, dropout=dropout)
        self.resblock7 = ResBlock(n_output_channels, n_output_channels, 
                                  kernel_size, stride = 2, n_pad = 4, dropout=dropout)
        self.resblock8 = ResBlock(n_output_channels, n_output_channels, 
                                  kernel_size, stride = 1, n_pad = 4, dropout=dropout)
        self.resblock9 = ResBlock(n_output_channels, n_output_channels, 
                                  kernel_size, stride = 2, n_pad = 4, dropout=dropout)
        self.resblock10 = ResBlock(n_output_channels, n_output_channels, 
                                  kernel_size, stride = 1, n_pad = 4, dropout=dropout)
        self.resblock11 = ResBlock(n_output_channels, n_output_channels, 
                                  kernel_size, stride = 2, n_pad = 4, dropout=dropout)
        self.resblock12 = ResBlock(n_output_channels, n_output_channels, 
                                  kernel_size, stride = 1, n_pad = 4, dropout=dropout)
        
        self.batchnorm3 = nn.BatchNorm1d(n_output_channels)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, n_classes)
    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out1 = self.relu1(out)
        out = self.conv2(out1)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = out1 + out
        out = self.resblock1(out)
        # out = self.resblock2(out)
        # out = self.resblock3(out)
        # out = self.resblock4(out)
        # out = self.resblock5(out)
        # out = self.resblock6(out)
        # out = self.resblock7(out)
        # out = self.resblock8(out)
        # out = self.resblock9(out)
        # out = self.resblock10(out)
        # out = self.resblock11(out)
        out = self.batchnorm3(out)
        out = self.relu3(out)
        out_mean = out.mean(axis = -1).unsqueeze(1)
        out_max = out.max(axis = -1).values.unsqueeze(1)
        out = torch.cat([out_mean, out_max], 1)
        out = self.flatten(out)
        out = self.linear(out)
        return out