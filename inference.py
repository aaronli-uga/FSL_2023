'''
Author: Qi7
Date: 2023-05-18 12:59:58
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-18 19:50:02
Description: 
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, train_test_split

from loader import waveformDataset
from model import LSTM, QNN
from training import model_train_multiclass, evaluate

X = np.load('dataset/8cases/X_test.npy')
y = np.load('dataset/8cases/y_test.npy')
saved_model = "saved_models/8cases_multiclass_epochs200_lr_0.001_bs_256_best_model.pth"
history = dict(test_loss=[], test_acc=[], test_f1=[], test_f1_all=[])

testset = waveformDataset(X, y)
testloader = DataLoader(testset, shuffle=True, batch_size=256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = QNN(n_input_channels=6,
            n_output_channels=64,
            kernel_size=3,
            stride=1,
            n_classes=8)
model.load_state_dict(torch.load(saved_model, map_location=device))
model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()

evaluate(
    dataloader=testloader,
    model=model,
    loss_fn=loss_fn,
    device=device,
    history=history
)
    
print("test accuracy:", history['test_acc'])
print("test f1:", history['test_f1'])
print("test f1 all:", history['test_f1_all'])
print("test loss:", history['test_loss'])