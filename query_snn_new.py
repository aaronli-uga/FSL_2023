'''
Author: Qi7
Date: 2023-05-16 23:28:18
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-04 21:18:14
Description: Test the accuracy from the query set. Number of shots and number of query is provided.
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
from model import SiameseNet, DistanceNet, QNN
from training import model_train_multiclass

X = np.load('dataset/8cases_jinan/new_query_set/X_norm.npy')
y = np.load('dataset/8cases_jinan/new_query_set/y.npy') # label from 8 to 14

X_test = np.load('dataset/8cases_jinan/new_test_set/X_test.npy')
y_test = np.load('dataset/8cases_jinan/new_test_set/y_test.npy')

X_0 = X_test[np.where(y_test == 0)[0]]
X_1 = X_test[np.where(y_test == 1)[0]]
X_2 = X_test[np.where(y_test == 2)[0]]
X_3 = X_test[np.where(y_test == 3)[0]]
X_4 = X_test[np.where(y_test == 4)[0]]
X_5 = X_test[np.where(y_test == 5)[0]]
X_6 = X_test[np.where(y_test == 6)[0]]
X_7 = X_test[np.where(y_test == 7)[0]]


X_8 = X[np.where(y == 8)[0]]
X_9 = X[np.where(y == 9)[0]]
X_10 = X[np.where(y == 10)[0]]
X_11 = X[np.where(y == 11)[0]]
X_12 = X[np.where(y == 12)[0]]

y_8 = y[np.where(y == 8)[0]]
y_9 = y[np.where(y == 9)[0]]
y_10 = y[np.where(y == 10)[0]]
y_11 = y[np.where(y == 11)[0]]
y_12 = y[np.where(y == 12)[0]]

# randomly select the support set.
num_shots = 30
num_query = 500


support_0 = X_0[np.random.randint(0, X_0.shape[0], num_shots)]
support_1 = X_1[np.random.randint(0, X_1.shape[0], num_shots)]
support_2 = X_2[np.random.randint(0, X_2.shape[0], num_shots)]
support_3 = X_3[np.random.randint(0, X_3.shape[0], num_shots)]
support_4 = X_4[np.random.randint(0, X_4.shape[0], num_shots)]
support_5 = X_5[np.random.randint(0, X_5.shape[0], num_shots)]
support_6 = X_6[np.random.randint(0, X_6.shape[0], num_shots)]
support_7 = X_7[np.random.randint(0, X_7.shape[0], num_shots)]

support_8 = X_8[np.random.randint(0, X_8.shape[0], num_shots)]
support_9 = X_9[np.random.randint(0, X_9.shape[0], num_shots)]
support_10 = X_10[np.random.randint(0, X_10.shape[0], num_shots)]
support_11 = X_11[np.random.randint(0, X_11.shape[0], num_shots)]
support_12 = X_12[np.random.randint(0, X_12.shape[0], num_shots)]

query_8 = X_8[np.random.randint(0, X_8.shape[0], num_query)]
query_9 = X_9[np.random.randint(0, X_9.shape[0], num_query)]
query_10 = X_10[np.random.randint(0, X_10.shape[0], num_query)]
query_11 = X_11[np.random.randint(0, X_11.shape[0], num_query)]
query_12 = X_12[np.random.randint(0, X_12.shape[0], num_query)]


trained_model = "saved_models/2d_snn/margin_1.0_epoch_300_contrastive_model.pth"

embedding_net = QNN(n_input_channels=6,
                    n_output_channels=64,
                    kernel_size=3,
                    stride=1,
                    n_classes=8
                )
embedding_net.fc1 = torch.nn.Flatten()

model = DistanceNet(embedding_net=embedding_net)
model.load_state_dict(torch.load(trained_model))
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    
    support_0 = torch.from_numpy(support_0)
    support_0 = support_0.to(device, dtype=torch.float)
    pred = model.get_embedding(support_0)
    support_0_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_1 = torch.from_numpy(support_1)
    support_1 = support_1.to(device, dtype=torch.float)
    pred = model.get_embedding(support_1)
    support_1_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_2 = torch.from_numpy(support_2)
    support_2 = support_2.to(device, dtype=torch.float)
    pred = model.get_embedding(support_2)
    support_2_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_3 = torch.from_numpy(support_3)
    support_3 = support_3.to(device, dtype=torch.float)
    pred = model.get_embedding(support_3)
    support_3_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_4 = torch.from_numpy(support_4)
    support_4 = support_4.to(device, dtype=torch.float)
    pred = model.get_embedding(support_4)
    support_4_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_5 = torch.from_numpy(support_5)
    support_5 = support_5.to(device, dtype=torch.float)
    pred = model.get_embedding(support_5)
    support_5_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_6 = torch.from_numpy(support_6)
    support_6 = support_6.to(device, dtype=torch.float)
    pred = model.get_embedding(support_6)
    support_6_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_7 = torch.from_numpy(support_7)
    support_7 = support_7.to(device, dtype=torch.float)
    pred = model.get_embedding(support_7)
    support_7_embedding = pred.cpu().numpy().mean(axis=0)
    
    
    support_8 = torch.from_numpy(support_8)
    support_8 = support_8.to(device, dtype=torch.float)
    pred = model.get_embedding(support_8)
    support_8_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_9 = torch.from_numpy(support_9)
    support_9 = support_9.to(device, dtype=torch.float)

    pred = model.get_embedding(support_9)
    support_9_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_10 = torch.from_numpy(support_10)
    support_10 = support_10.to(device, dtype=torch.float)
    pred = model.get_embedding(support_10)
    support_10_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_11 = torch.from_numpy(support_11)
    support_11 = support_11.to(device, dtype=torch.float)
    pred = model.get_embedding(support_11)
    support_11_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_12 = torch.from_numpy(support_12)
    support_12 = support_12.to(device, dtype=torch.float)
    pred = model.get_embedding(support_12)
    support_12_embedding = pred.cpu().numpy().mean(axis=0)
    
    
    query_8 = torch.from_numpy(query_8)
    query_8 = query_8.to(device, dtype=torch.float)
    pred = model.get_embedding(query_8)
    query_8_embedding = pred.cpu().numpy()
    
    query_9 = torch.from_numpy(query_9)
    query_9 = query_9.to(device, dtype=torch.float)
    pred = model.get_embedding(query_9)
    query_9_embedding = pred.cpu().numpy()
    
    query_10 = torch.from_numpy(query_10)
    query_10 = query_10.to(device, dtype=torch.float)
    pred = model.get_embedding(query_10)
    query_10_embedding = pred.cpu().numpy()
    
    query_11 = torch.from_numpy(query_11)
    query_11 = query_11.to(device, dtype=torch.float)
    pred = model.get_embedding(query_11)
    query_11_embedding = pred.cpu().numpy()
    
    query_12 = torch.from_numpy(query_12)
    query_12 = query_12.to(device, dtype=torch.float)
    pred = model.get_embedding(query_12)
    query_12_embedding = pred.cpu().numpy()


# accuracy calculation
cnt = 0
for i in range(num_query):
    dis_query_8 = [
        np.linalg.norm(query_8_embedding[i] - support_0_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_1_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_2_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_3_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_4_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_5_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_6_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_7_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_8_embedding), 
    ]
    if dis_query_8.index(min(dis_query_8)) == 8:
        cnt += 1
print("query 8 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_9 = [
        np.linalg.norm(query_9_embedding[i] - support_0_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_1_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_2_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_3_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_4_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_5_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_6_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_7_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_9_embedding), 
    ]
    if dis_query_9.index(min(dis_query_9)) == 8:
        cnt += 1
print("query 9 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_10 = [
        np.linalg.norm(query_10_embedding[i] - support_0_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_1_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_2_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_3_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_4_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_5_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_6_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_7_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_10_embedding), 
    ]
    if dis_query_10.index(min(dis_query_10)) == 8:
        cnt += 1
print("query 10 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_11 = [
        np.linalg.norm(query_11_embedding[i] - support_0_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_1_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_2_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_3_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_4_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_5_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_6_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_7_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_11_embedding), 
    ]
    if dis_query_11.index(min(dis_query_11)) == 8:
        cnt += 1
print("query 11 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_12 = [
        np.linalg.norm(query_12_embedding[i] - support_0_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_1_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_2_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_3_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_4_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_5_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_6_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_7_embedding), 
        np.linalg.norm(query_12_embedding[i] - support_12_embedding), 
    ]
    if dis_query_12.index(min(dis_query_12)) == 8:
        cnt += 1
print("query 12 accuracy:", cnt / num_query)
# %%
