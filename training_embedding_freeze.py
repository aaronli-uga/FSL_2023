'''
Author: Qi7
Date: 2023-06-02 16:01:42
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-04 17:00:08
Description: 
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn as nn
import copy
import tqdm, time

from loader import siameseDataset
from model import DistanceNet, QNN
from training import fit
from losses import ContrastiveLoss

save_model_path = "saved_models/2d_snn/"
X = np.load('dataset/8cases_jinan/new_training_set/X_train.npy')
y = np.load('dataset/8cases_jinan/new_training_set/y_train.npy')

trained_embedding = "saved_models/new_without_snn/multiclass_epochs50_lr_0.001_bs_256_best_model.pth"

X_train, X_cv, y_train, y_cv = train_test_split(X, y, train_size=0.75, test_size=0.25, shuffle=True, random_state=27)
trainset = siameseDataset(X_train, y_train)
validset = siameseDataset(X_cv, y_cv)
batch_size = 128
train_data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(validset, batch_size=batch_size, shuffle=False)

embedding_net = QNN(n_input_channels=6,
                    n_output_channels=64,
                    kernel_size=3,
                    stride=1,
                    n_classes=8
                )
embedding_net.load_state_dict(torch.load(trained_embedding))
embedding_net.fc1 = nn.Flatten()

model = DistanceNet(embedding_net=embedding_net)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
margin = 1.
lr = 1e-3
n_epochs = 300
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = ContrastiveLoss(margin=margin)
history = dict(train_loss=[], val_loss=[])
model.to(device)

fit(train_loader=train_data_loader,
    val_loader=val_data_loader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    n_epochs=n_epochs,
    device=device,
    history=history,
    save_model_path=save_model_path,
    margin=margin
)

# torch.save(model.state_dict(), save_model_path + f"margin_{margin}_epoch_{n_epochs}_contrastive_model.pth")

np.save(save_model_path + f"margin_{margin}_epoch_{n_epochs}_contrastive_history.npy", history)

