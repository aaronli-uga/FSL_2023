'''
Author: Qi7
Date: 2023-04-08 11:54:41
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-24 16:34:58
Description: main function for doing the multiclass classification. Both jinan and jiabao's data acheive good results using CNN (with resblocks)
'''
#%%

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, train_test_split

from loader import waveformDataset
from model import LSTM, QNN
from training import model_train_multiclass

save_model_path = "saved_models/snn/"
X = np.load('dataset/8cases/X_norm.npy')
y = np.load('dataset/8cases/y.npy')
# X = X[np.where((y == 0) | (y == 8) | (y == 7))[0]]
# y = y[np.where((y == 0) | (y == 8) | (y == 7))[0]]

# X = X[np.where((y == 1) | (y == 2) | (y == 3) | (y == 4) | (y == 5))[0]]
# y = y[np.where((y == 1) | (y == 2) | (y == 3) | (y == 4) | (y == 5))[0]]


# Standard Normalization ((X-mean) / std)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         X[i,j,:] = (X[i,j,:] - X[i,j,:].mean()) / X[i,j,:].std()

# X = np.transpose(X, (0,2,1)) # transpose to match the lstm standard
# y = np.expand_dims(y, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=True, random_state=7)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, train_size=0.75, test_size=0.25, shuffle=True, random_state=27)

# save test dataset for ultimate testing.
# np.save("X_test_new.npy", X_test)
# np.save("y_test_new.npy", y_test)

trainset = waveformDataset(X_train, y_train)
validset = waveformDataset(X_cv, y_cv)

# Hyper parameters
batch_size = 256
learning_rate = 0.001
num_epochs = 200
history = dict(test_loss=[], test_acc=[], test_f1=[], test_f1_all=[])

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
testloader = DataLoader(validset, shuffle=True, batch_size=256) # get all the samples at once
# model = LSTM(input_size=6, seq_num=100, num_class=9)
model = QNN(n_input_channels=6,
            n_output_channels=64,
            kernel_size=3,
            stride=1,
            n_classes=8)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
# model training
model_train_multiclass(
    model=model, 
    train_loader=trainloader, 
    val_loader=testloader,
    num_epochs=num_epochs,
    optimizer=optimizer,
    device=device,
    history=history
)

torch.save(model.state_dict(), save_model_path + f"test_8cases_multiclass_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_best_model.pth")
np.save(save_model_path + f"test_8cases_multiclass_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_history.npy", history)