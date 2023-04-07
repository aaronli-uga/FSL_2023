'''
Author: Qi7
Date: 2023-04-06 21:23:30
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-07 11:46:40
Description: main function for doing our task
'''
#%%

import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, train_test_split

from loader import waveformDataset
from model import LSTM

X = np.load('dataset/w100_detection_data_norm.npy')
y = np.load('dataset/w100_detection_label.npy')

# Standard Normalization ((X-mean) / std)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         X[i,j,:] = (X[i,j,:] - X[i,j,:].mean()) / X[i,j,:].std()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=7)
trainset = waveformDataset(X_train, y_train)
testset = waveformDataset(X_test, )

# Hyper parameters
batch_size = 256
learning_rate = 0.001
num_epochs = 100

trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
testloader = DataLoader(testset, shuffle=False, batch_size=testset.__len__()) # get all the samples at once
model = LSTM()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# %%
