'''
Author: Qi7
Date: 2023-04-06 21:23:30
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-17 22:07:07
Description: binary classification using jiabao's pv data.
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
from model import LSTM
from training import model_train

save_model_path = "saved_models/"
X = np.load('dataset/9cases_w100_jiabao/w100_detection_data_norm.npy')
y = np.load('dataset/9cases_w100_jiabao/w100_detection_label.npy')

# Standard Normalization ((X-mean) / std)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         X[i,j,:] = (X[i,j,:] - X[i,j,:].mean()) / X[i,j,:].std()


X = np.transpose(X, (0,2,1)) # transpose to match the lstm standard
y = np.expand_dims(y, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=7)
trainset = waveformDataset(X_train, y_train)
testset = waveformDataset(X_test, y_test)

# Hyper parameters
batch_size = 256
learning_rate = 0.001
num_epochs = 500
history = dict(train_loss=[], test_loss=[], train_acc=[], test_acc=[], train_f1=[], test_f1=[])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
testloader = DataLoader(testset, shuffle=False, batch_size=1024) # get all the samples at once
model = LSTM(input_size=6, seq_num=100, num_class=1)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
# model training
model_train(
    model=model, 
    train_loader=trainloader, 
    val_loader=testloader,
    num_epochs=num_epochs,
    optimizer=optimizer,
    device=device,
    history=history
)

torch.save(model.state_dict(), save_model_path + f"best_model.pth")
np.save(save_model_path + f"epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_history.npy", history)
# %%
