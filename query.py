'''
Author: Qi7
Date: 2023-05-16 23:28:18
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-01 23:03:08
Description: query on model not trained by snn. Simply remove the last layer of the model to get the feature embedding.
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

from loader import waveformDataset
from model import LSTM, QNN
from training import model_train_multiclass

X = np.load('dataset/8cases_jinan/new_query_set/X_norm.npy')
y = np.load('dataset/8cases_jinan/new_query_set/y.npy') # label from 8 to 14

X_8 = X[np.where(y == 8)[0]]
X_9 = X[np.where(y == 9)[0]]
X_10 = X[np.where(y == 10)[0]]
X_11 = X[np.where(y == 11)[0]]
X_12 = X[np.where(y == 12)[0]]
X_13 = X[np.where(y == 13)[0]]

y_8 = y[np.where(y == 8)[0]]
y_9 = y[np.where(y == 9)[0]]
y_10 = y[np.where(y == 10)[0]]
y_11 = y[np.where(y == 11)[0]]
y_12 = y[np.where(y == 12)[0]]
y_13 = y[np.where(y == 13)[0]]

# randomly select the support set.
num_shots = 10
num_query = 100
support_8 = X_8[np.random.randint(0, X_8.shape[0], num_shots)]
support_9 = X_9[np.random.randint(0, X_9.shape[0], num_shots)]
support_10 = X_10[np.random.randint(0, X_10.shape[0], num_shots)]
support_11 = X_11[np.random.randint(0, X_11.shape[0], num_shots)]
support_12 = X_12[np.random.randint(0, X_12.shape[0], num_shots)]
support_13 = X_13[np.random.randint(0, X_13.shape[0], num_shots)]

query_8 = X_8[np.random.randint(0, X_8.shape[0], num_query)]
query_9 = X_9[np.random.randint(0, X_9.shape[0], num_query)]
query_10 = X_10[np.random.randint(0, X_10.shape[0], num_query)]
query_11 = X_11[np.random.randint(0, X_11.shape[0], num_query)]
query_12 = X_12[np.random.randint(0, X_12.shape[0], num_query)]
query_13 = X_13[np.random.randint(0, X_13.shape[0], num_query)]

# trained_model = "saved_models/five_multiclass_best_model.pth"
# trained_model = "saved_models/8cases_multiclass_epochs200_lr_0.001_bs_256_best_model.pth"
trained_model = "saved_models/new_without_snn/multiclass_epochs50_lr_0.001_bs_256_best_model.pth"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = QNN(n_input_channels=6,
            n_output_channels=64,
            kernel_size=3,
            stride=1,
            n_classes=8)
model.load_state_dict(torch.load(trained_model))
model.fc1 = torch.nn.Flatten() # get the embedding
model.to(device)

with torch.no_grad():
    support_8 = torch.from_numpy(support_8)
    support_8 = support_8.to(device, dtype=torch.float)
    pred = model(support_8)
    support_8_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_9 = torch.from_numpy(support_9)
    support_9 = support_9.to(device, dtype=torch.float)
    pred = model(support_9)
    support_9_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_10 = torch.from_numpy(support_10)
    support_10 = support_10.to(device, dtype=torch.float)
    pred = model(support_10)
    support_10_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_11 = torch.from_numpy(support_11)
    support_11 = support_11.to(device, dtype=torch.float)
    pred = model(support_11)
    support_11_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_12 = torch.from_numpy(support_12)
    support_12 = support_12.to(device, dtype=torch.float)
    pred = model(support_12)
    support_12_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_13 = torch.from_numpy(support_13)
    support_13 = support_13.to(device, dtype=torch.float)
    pred = model(support_13)
    support_13_embedding = pred.cpu().numpy().mean(axis=0)
    
    query_8 = torch.from_numpy(query_8)
    query_8 = query_8.to(device, dtype=torch.float)
    pred = model(query_8)
    query_8_embedding = pred.cpu().numpy()
    
    query_9 = torch.from_numpy(query_9)
    query_9 = query_9.to(device, dtype=torch.float)
    pred = model(query_9)
    query_9_embedding = pred.cpu().numpy()
    
    query_10 = torch.from_numpy(query_10)
    query_10 = query_10.to(device, dtype=torch.float)
    pred = model(query_10)
    query_10_embedding = pred.cpu().numpy()
    
    query_11 = torch.from_numpy(query_11)
    query_11 = query_11.to(device, dtype=torch.float)
    pred = model(query_11)
    query_11_embedding = pred.cpu().numpy()
    
    query_12 = torch.from_numpy(query_12)
    query_12 = query_12.to(device, dtype=torch.float)
    pred = model(query_12)
    query_12_embedding = pred.cpu().numpy()
    
    query_13 = torch.from_numpy(query_13)
    query_13 = query_13.to(device, dtype=torch.float)
    pred = model(query_13)
    query_13_embedding = pred.cpu().numpy()

#%%
# accuracy calculation
cnt = 0
for i in range(num_query):
    dis_query_8 = [
        np.linalg.norm(query_8_embedding[i] - support_8_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_9_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_10_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_11_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_12_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_13_embedding), 
    ]
    if dis_query_8.index(min(dis_query_8)) + 8 == 8:
        cnt += 1
print("query 8 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_9 = [
        np.linalg.norm(query_9_embedding[i] - support_8_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_9_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_10_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_11_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_12_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_13_embedding), 
    ]
    if dis_query_9.index(min(dis_query_9)) + 8 == 9:
        cnt += 1
print("query 9 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_10 = [
        np.linalg.norm(query_10_embedding[i] - support_8_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_9_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_10_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_11_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_12_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_13_embedding),
    ]
    if dis_query_10.index(min(dis_query_10)) + 8 == 10:
        cnt += 1
print("query 10 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_11 = [
        np.linalg.norm(query_11_embedding[i] - support_8_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_9_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_10_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_11_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_12_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_13_embedding), 
    ]
    if dis_query_11.index(min(dis_query_11)) + 8 == 11:
        cnt += 1
print("query 11 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_12 = [
        np.linalg.norm(query_12_embedding[i] - support_8_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_9_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_10_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_11_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_12_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_13_embedding), 
    ]
    if dis_query_12.index(min(dis_query_12)) + 8 == 12:
        cnt += 1
print("query 12 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_13 = [
        np.linalg.norm(query_13_embedding[i] - support_8_embedding), 
        np.linalg.norm(query_13_embedding[i] - support_9_embedding), 
        np.linalg.norm(query_13_embedding[i] - support_10_embedding), 
        np.linalg.norm(query_13_embedding[i] - support_11_embedding), 
        np.linalg.norm(query_13_embedding[i] - support_12_embedding), 
        np.linalg.norm(query_13_embedding[i] - support_13_embedding), 
    ]
    if dis_query_13.index(min(dis_query_13)) + 8 == 13:
        cnt += 1
print("query 13 accuracy:", cnt / num_query)


#%%
# This code block is just for one query and get a instant result!!!
# calculate the Eclidean DistanceX_1_embedding = np.expand_dims(X_1_embedding, axis=0)
# print("distance for query_8:")
dis_query_8 = [
        np.linalg.norm(query_8_embedding - support_8_embedding), 
        np.linalg.norm(query_8_embedding - support_9_embedding), 
        np.linalg.norm(query_8_embedding - support_10_embedding), 
        np.linalg.norm(query_8_embedding - support_11_embedding), 
        np.linalg.norm(query_8_embedding - support_12_embedding), 
        np.linalg.norm(query_8_embedding - support_13_embedding), 
        np.linalg.norm(query_8_embedding - support_14_embedding),
    ]
print("query 8, attack number:", dis_query_8.index(min(dis_query_8)) + 8)

# print("distance for query_1:")
dis_query_9 = [
        np.linalg.norm(query_9_embedding - support_8_embedding), 
        np.linalg.norm(query_9_embedding - support_9_embedding), 
        np.linalg.norm(query_9_embedding - support_10_embedding), 
        np.linalg.norm(query_9_embedding - support_11_embedding), 
        np.linalg.norm(query_9_embedding - support_12_embedding), 
        np.linalg.norm(query_9_embedding - support_13_embedding), 
        np.linalg.norm(query_9_embedding - support_14_embedding),
    ]
print("query 9, attack number:", dis_query_9.index(min(dis_query_9)) + 8)

# print("distance for query_2:")
dis_query_10 = [
        np.linalg.norm(query_10_embedding - support_8_embedding), 
        np.linalg.norm(query_10_embedding - support_9_embedding), 
        np.linalg.norm(query_10_embedding - support_10_embedding), 
        np.linalg.norm(query_10_embedding - support_11_embedding), 
        np.linalg.norm(query_10_embedding - support_12_embedding), 
        np.linalg.norm(query_10_embedding - support_13_embedding), 
        np.linalg.norm(query_10_embedding - support_14_embedding),
    ]
print("query 10, attack number:", dis_query_10.index(min(dis_query_10)) + 8)

# print("distance for query_3:")
dis_query_11 = [
        np.linalg.norm(query_11_embedding - support_8_embedding), 
        np.linalg.norm(query_11_embedding - support_9_embedding), 
        np.linalg.norm(query_11_embedding - support_10_embedding), 
        np.linalg.norm(query_11_embedding - support_11_embedding), 
        np.linalg.norm(query_11_embedding - support_12_embedding), 
        np.linalg.norm(query_11_embedding - support_13_embedding), 
        np.linalg.norm(query_11_embedding - support_14_embedding),
    ]
print("query 11, attack number:", dis_query_11.index(min(dis_query_11)) + 8)

# print("distance for query_4:")
dis_query_12 = [
        np.linalg.norm(query_12_embedding - support_8_embedding), 
        np.linalg.norm(query_12_embedding - support_9_embedding), 
        np.linalg.norm(query_12_embedding - support_10_embedding), 
        np.linalg.norm(query_12_embedding - support_11_embedding), 
        np.linalg.norm(query_12_embedding - support_12_embedding), 
        np.linalg.norm(query_12_embedding - support_13_embedding), 
        np.linalg.norm(query_12_embedding - support_14_embedding),
    ]
print("query 12, attack number:", dis_query_12.index(min(dis_query_12)) + 8)

# print("attack number:", dis_query_4.index(min(dis_query_4)))

# print("distance for query_5:")
dis_query_13 = [
        np.linalg.norm(query_13_embedding - support_8_embedding), 
        np.linalg.norm(query_13_embedding - support_9_embedding), 
        np.linalg.norm(query_13_embedding - support_10_embedding), 
        np.linalg.norm(query_13_embedding - support_11_embedding), 
        np.linalg.norm(query_13_embedding - support_12_embedding), 
        np.linalg.norm(query_13_embedding - support_13_embedding), 
        np.linalg.norm(query_13_embedding - support_14_embedding),
    ]
print("query 13, attack number:", dis_query_13.index(min(dis_query_13)) + 8)
# print("attack number:", dis_query_5.index(min(dis_query_5)))

# print("distance for query_6:")
dis_query_14 = [
        np.linalg.norm(query_14_embedding - support_8_embedding), 
        np.linalg.norm(query_14_embedding - support_9_embedding), 
        np.linalg.norm(query_14_embedding - support_10_embedding), 
        np.linalg.norm(query_14_embedding - support_11_embedding), 
        np.linalg.norm(query_14_embedding - support_12_embedding), 
        np.linalg.norm(query_14_embedding - support_13_embedding), 
        np.linalg.norm(query_14_embedding - support_14_embedding),
    ]
print("query 14, attack number:", dis_query_14.index(min(dis_query_14)) + 8)