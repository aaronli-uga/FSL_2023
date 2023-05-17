'''
Author: Qi7
Date: 2023-05-16 23:28:18
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-17 00:15:06
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
from training import model_train_multiclass

X = np.load('dataset/w100_diagnosis_data_norm.npy')
y = np.load('dataset/w100_diagnosis_label.npy')
X_1 = X[np.where(y == 1)[0]]
X_2 = X[np.where(y == 2)[0]]
X_3 = X[np.where(y == 3)[0]]
X_4 = X[np.where(y == 4)[0]]
X_5 = X[np.where(y == 5)[0]]
X_6 = X[np.where(y == 6)[0]]

y_1 = y[np.where(y == 1)[0]]
y_2 = y[np.where(y == 2)[0]]
y_3 = y[np.where(y == 3)[0]]
y_4 = y[np.where(y == 4)[0]]
y_5 = y[np.where(y == 5)[0]]
y_6 = y[np.where(y == 6)[0]]

# randomly select the support set.
num_shots = 5
num_query = 1
support_1 = X_1[np.random.randint(0, X_1.shape[0]+1, num_shots)]
support_2 = X_2[np.random.randint(0, X_2.shape[0]+1, num_shots)]
support_3 = X_3[np.random.randint(0, X_3.shape[0]+1, num_shots)]
support_4 = X_4[np.random.randint(0, X_4.shape[0]+1, num_shots)]
support_5 = X_5[np.random.randint(0, X_5.shape[0]+1, num_shots)]
support_6 = X_6[np.random.randint(0, X_6.shape[0]+1, num_shots)]

query_1 = X_1[np.random.randint(0, X_1.shape[0]+1, num_query)]
query_2 = X_2[np.random.randint(0, X_2.shape[0]+1, num_query)]
query_3 = X_3[np.random.randint(0, X_3.shape[0]+1, num_query)]
query_4 = X_4[np.random.randint(0, X_4.shape[0]+1, num_query)]
query_5 = X_5[np.random.randint(0, X_5.shape[0]+1, num_query)]
query_6 = X_6[np.random.randint(0, X_6.shape[0]+1, num_query)]

trained_model = "saved_models/three_multiclass_best_model.pth"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = QNN(n_input_channels=6,
            n_output_channels=64,
            kernel_size=3,
            stride=1,
            n_classes=9)
model.load_state_dict(torch.load(trained_model))
model.fc = torch.nn.Flatten() # get the embedding
model.to(device)

with torch.no_grad():
    support_1 = torch.from_numpy(support_1)
    support_1.to(device, dtype=torch.float)
    pred = model(support_1)
    support_1_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_2 = torch.from_numpy(support_2)
    support_2.to(device, dtype=torch.float)
    pred = model(support_2)
    support_2_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_3 = torch.from_numpy(support_3)
    support_3.to(device, dtype=torch.float)
    pred = model(support_3)
    support_3_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_4 = torch.from_numpy(support_4)
    support_4.to(device, dtype=torch.float)
    pred = model(support_4)
    support_4_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_5 = torch.from_numpy(support_5)
    support_5.to(device, dtype=torch.float)
    pred = model(support_5)
    support_5_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_6 = torch.from_numpy(support_6)
    support_6.to(device, dtype=torch.float)
    pred = model(support_6)
    support_6_embedding = pred.cpu().numpy().mean(axis=0)
    
    query_1 = torch.from_numpy(query_1)
    query_1.to(device, dtype=torch.float)
    pred = model(query_1)
    query_1_embedding = pred.cpu().numpy().mean(axis=0)
    
    query_2 = torch.from_numpy(query_2)
    query_2.to(device, dtype=torch.float)
    pred = model(query_2)
    query_2_embedding = pred.cpu().numpy().mean(axis=0)
    
    query_3 = torch.from_numpy(query_3)
    query_3.to(device, dtype=torch.float)
    pred = model(query_3)
    query_3_embedding = pred.cpu().numpy().mean(axis=0)
    
    query_4 = torch.from_numpy(query_4)
    query_4.to(device, dtype=torch.float)
    pred = model(query_4)
    query_4_embedding = pred.cpu().numpy().mean(axis=0)
    
    query_5 = torch.from_numpy(query_5)
    query_5.to(device, dtype=torch.float)
    pred = model(query_5)
    query_5_embedding = pred.cpu().numpy().mean(axis=0)
    
    query_6 = torch.from_numpy(query_6)
    query_6.to(device, dtype=torch.float)
    pred = model(query_6)
    query_6_embedding = pred.cpu().numpy().mean(axis=0)
    

np.linalg.norm(query_1_embedding - support_1_embedding)
