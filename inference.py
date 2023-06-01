'''
Author: Qi7
Date: 2023-05-18 12:59:58
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-01 15:58:00
Description: doing the inference on test data using a trained model. History information such as accuracy and f1 score will be printed.
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from loader import waveformDataset
from model import QNN
from training import evaluate

X = np.load('dataset/8cases_jinan/new_test_set/X_test.npy')
y = np.load('dataset/8cases_jinan/new_test_set/y_test.npy')

saved_model = "saved_models/new_without_snn/multiclass_epochs50_lr_0.001_bs_256_best_model.pth"
history = dict(test_loss=[], test_acc=[], test_f1=[], test_f1_all=[])

testset = waveformDataset(X, y)
testloader = DataLoader(testset, shuffle=True, batch_size=256)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = QNN(n_input_channels=6,
            n_output_channels=64,
            kernel_size=3,
            stride=1,
            n_classes=8)
model.load_state_dict(torch.load(saved_model, map_location=device))
# model.load_state_dict(torch.load(saved_model))
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