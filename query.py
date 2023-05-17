'''
Author: Qi7
Date: 2023-05-16 23:28:18
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-17 15:06:05
Description: 
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

X = np.load('dataset/w100_diagnosis_data_norm.npy')
y = np.load('dataset/w100_diagnosis_label.npy')
X_0 = X[np.where(y == 0)[0]]
X_1 = X[np.where(y == 1)[0]]
X_2 = X[np.where(y == 2)[0]]
X_3 = X[np.where(y == 3)[0]]
X_4 = X[np.where(y == 4)[0]]
X_5 = X[np.where(y == 5)[0]]
X_6 = X[np.where(y == 6)[0]]
X_7 = X[np.where(y == 7)[0]]
X_8 = X[np.where(y == 8)[0]]

y_0 = y[np.where(y == 0)[0]]
y_1 = y[np.where(y == 1)[0]]
y_2 = y[np.where(y == 2)[0]]
y_3 = y[np.where(y == 3)[0]]
y_4 = y[np.where(y == 4)[0]]
y_5 = y[np.where(y == 5)[0]]
y_6 = y[np.where(y == 6)[0]]
y_7 = y[np.where(y == 7)[0]]
y_8 = y[np.where(y == 8)[0]]

# randomly select the support set.
num_shots = 70
num_query = 100
support_0 = X_0[np.random.randint(0, X_0.shape[0], num_shots)]
support_1 = X_1[np.random.randint(0, X_1.shape[0], num_shots)]
support_2 = X_2[np.random.randint(0, X_2.shape[0], num_shots)]
support_3 = X_3[np.random.randint(0, X_3.shape[0], num_shots)]
support_4 = X_4[np.random.randint(0, X_4.shape[0], num_shots)]
support_5 = X_5[np.random.randint(0, X_5.shape[0], num_shots)]
support_6 = X_6[np.random.randint(0, X_6.shape[0], num_shots)]
support_7 = X_7[np.random.randint(0, X_7.shape[0], num_shots)]
support_8 = X_8[np.random.randint(0, X_8.shape[0], num_shots)]

query_0 = X_0[np.random.randint(0, X_0.shape[0], num_query)]
query_1 = X_1[np.random.randint(0, X_1.shape[0], num_query)]
query_2 = X_2[np.random.randint(0, X_2.shape[0], num_query)]
query_3 = X_3[np.random.randint(0, X_3.shape[0], num_query)]
query_4 = X_4[np.random.randint(0, X_4.shape[0], num_query)]
query_5 = X_5[np.random.randint(0, X_5.shape[0], num_query)]
query_6 = X_6[np.random.randint(0, X_6.shape[0], num_query)]
query_7 = X_7[np.random.randint(0, X_7.shape[0], num_query)]
query_8 = X_8[np.random.randint(0, X_8.shape[0], num_query)]

trained_model = "saved_models/five_multiclass_best_model.pth"
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
    support_0 = torch.from_numpy(support_0)
    support_0.to(device, dtype=torch.float)
    pred = model(support_0)
    support_0_embedding = pred.cpu().numpy().mean(axis=0)
    
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
    
    support_7 = torch.from_numpy(support_7)
    support_7.to(device, dtype=torch.float)
    pred = model(support_7)
    support_7_embedding = pred.cpu().numpy().mean(axis=0)
    
    support_8 = torch.from_numpy(support_8)
    support_8.to(device, dtype=torch.float)
    pred = model(support_8)
    support_8_embedding = pred.cpu().numpy().mean(axis=0)
    
    query_0 = torch.from_numpy(query_0)
    query_0.to(device, dtype=torch.float)
    pred = model(query_0)
    query_0_embedding = pred.cpu().numpy()
    
    query_1 = torch.from_numpy(query_1)
    query_1.to(device, dtype=torch.float)
    pred = model(query_1)
    query_1_embedding = pred.cpu().numpy()
    
    query_2 = torch.from_numpy(query_2)
    query_2.to(device, dtype=torch.float)
    pred = model(query_2)
    query_2_embedding = pred.cpu().numpy()
    
    query_3 = torch.from_numpy(query_3)
    query_3.to(device, dtype=torch.float)
    pred = model(query_3)
    query_3_embedding = pred.cpu().numpy()
    
    query_4 = torch.from_numpy(query_4)
    query_4.to(device, dtype=torch.float)
    pred = model(query_4)
    query_4_embedding = pred.cpu().numpy()
    
    query_5 = torch.from_numpy(query_5)
    query_5.to(device, dtype=torch.float)
    pred = model(query_5)
    query_5_embedding = pred.cpu().numpy()
    
    query_6 = torch.from_numpy(query_6)
    query_6.to(device, dtype=torch.float)
    pred = model(query_6)
    query_6_embedding = pred.cpu().numpy()
    
    query_7 = torch.from_numpy(query_7)
    query_7.to(device, dtype=torch.float)
    pred = model(query_7)
    query_7_embedding = pred.cpu().numpy()
    
    query_8 = torch.from_numpy(query_8)
    query_8.to(device, dtype=torch.float)
    pred = model(query_8)
    query_8_embedding = pred.cpu().numpy()
    
# accuracy calculation
cnt = 0
for i in range(num_query):
    dis_query_0 = [
        np.linalg.norm(query_0_embedding[i] - support_0_embedding), 
        np.linalg.norm(query_0_embedding[i] - support_6_embedding), 
        np.linalg.norm(query_0_embedding[i] - support_7_embedding), 
        np.linalg.norm(query_0_embedding[i] - support_8_embedding)]
    if dis_query_0.index(min(dis_query_0)) == 0:
        cnt += 1
print("query 0 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_6 = [
        np.linalg.norm(query_6_embedding[i] - support_0_embedding), 
        np.linalg.norm(query_6_embedding[i] - support_6_embedding), 
        np.linalg.norm(query_6_embedding[i] - support_7_embedding), 
        np.linalg.norm(query_6_embedding[i] - support_8_embedding)]
    if dis_query_6.index(min(dis_query_6)) == 1:
        cnt += 1
print("query 6 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_7 = [
        np.linalg.norm(query_7_embedding[i] - support_0_embedding), 
        np.linalg.norm(query_7_embedding[i] - support_6_embedding), 
        np.linalg.norm(query_7_embedding[i] - support_7_embedding), 
        np.linalg.norm(query_7_embedding[i] - support_8_embedding)]
    if dis_query_7.index(min(dis_query_7)) == 2:
        cnt += 1
print("query 7 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_8 = [
        np.linalg.norm(query_8_embedding[i] - support_0_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_6_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_7_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_8_embedding)]
    if dis_query_8.index(min(dis_query_8)) == 0:
        cnt += 1
print("query 0 accuracy:", cnt / num_query)


#%%

# calculate the Eclidean DistanceX_1_embedding = np.expand_dims(X_1_embedding, axis=0)
print("distance for query_0:")
dis_query_0 = [np.linalg.norm(query_0_embedding - support_0_embedding), 
        np.linalg.norm(query_0_embedding - support_1_embedding), 
        np.linalg.norm(query_0_embedding - support_2_embedding), 
        np.linalg.norm(query_0_embedding - support_3_embedding), 
        np.linalg.norm(query_0_embedding - support_4_embedding), 
        np.linalg.norm(query_0_embedding - support_5_embedding), 
        np.linalg.norm(query_0_embedding - support_6_embedding),
        np.linalg.norm(query_0_embedding - support_7_embedding),
        np.linalg.norm(query_0_embedding - support_8_embedding)]
print("attack number:", dis_query_0.index(min(dis_query_0)))

# print("distance for query_1:")
dis_query_1 = [np.linalg.norm(query_1_embedding - support_0_embedding),
        np.linalg.norm(query_1_embedding - support_1_embedding), 
        np.linalg.norm(query_1_embedding - support_2_embedding), 
        np.linalg.norm(query_1_embedding - support_3_embedding), 
        np.linalg.norm(query_1_embedding - support_4_embedding), 
        np.linalg.norm(query_1_embedding - support_5_embedding), 
        np.linalg.norm(query_1_embedding - support_6_embedding),
        np.linalg.norm(query_1_embedding - support_7_embedding),
        np.linalg.norm(query_1_embedding - support_8_embedding)]
# print("attack number:", dis_query_1.index(min(dis_query_1)))

# print("distance for query_2:")
dis_query_2 = [np.linalg.norm(query_2_embedding - support_0_embedding),
        np.linalg.norm(query_2_embedding - support_1_embedding), 
        np.linalg.norm(query_2_embedding - support_2_embedding), 
        np.linalg.norm(query_2_embedding - support_3_embedding), 
        np.linalg.norm(query_2_embedding - support_4_embedding), 
        np.linalg.norm(query_2_embedding - support_5_embedding), 
        np.linalg.norm(query_2_embedding - support_6_embedding),
        np.linalg.norm(query_2_embedding - support_7_embedding), 
        np.linalg.norm(query_2_embedding - support_8_embedding)]
# print("attack number:", dis_query_2.index(min(dis_query_2)))

# print("distance for query_3:")
dis_query_3 = [np.linalg.norm(query_3_embedding - support_0_embedding),
        np.linalg.norm(query_3_embedding - support_1_embedding), 
        np.linalg.norm(query_3_embedding - support_2_embedding), 
        np.linalg.norm(query_3_embedding - support_3_embedding), 
        np.linalg.norm(query_3_embedding - support_4_embedding), 
        np.linalg.norm(query_3_embedding - support_5_embedding), 
        np.linalg.norm(query_3_embedding - support_6_embedding),
        np.linalg.norm(query_3_embedding - support_7_embedding),
        np.linalg.norm(query_3_embedding - support_8_embedding)]
# print("attack number:", dis_query_3.index(min(dis_query_3)))

# print("distance for query_4:")
dis_query_4 = [np.linalg.norm(query_4_embedding - support_0_embedding),
        np.linalg.norm(query_4_embedding - support_1_embedding), 
        np.linalg.norm(query_4_embedding - support_2_embedding), 
        np.linalg.norm(query_4_embedding - support_3_embedding), 
        np.linalg.norm(query_4_embedding - support_4_embedding), 
        np.linalg.norm(query_4_embedding - support_5_embedding), 
        np.linalg.norm(query_4_embedding - support_6_embedding),
        np.linalg.norm(query_4_embedding - support_7_embedding),
        np.linalg.norm(query_4_embedding - support_8_embedding)]

# print("attack number:", dis_query_4.index(min(dis_query_4)))

# print("distance for query_5:")
dis_query_5 = [np.linalg.norm(query_5_embedding - support_0_embedding),
        np.linalg.norm(query_5_embedding - support_1_embedding), 
        np.linalg.norm(query_5_embedding - support_2_embedding), 
        np.linalg.norm(query_5_embedding - support_3_embedding), 
        np.linalg.norm(query_5_embedding - support_4_embedding), 
        np.linalg.norm(query_5_embedding - support_5_embedding), 
        np.linalg.norm(query_5_embedding - support_6_embedding),
        np.linalg.norm(query_5_embedding - support_7_embedding),
        np.linalg.norm(query_5_embedding - support_8_embedding)]
# print("attack number:", dis_query_5.index(min(dis_query_5)))

print("distance for query_6:")
dis_query_6 = [np.linalg.norm(query_6_embedding - support_0_embedding),
        np.linalg.norm(query_6_embedding - support_1_embedding), 
        np.linalg.norm(query_6_embedding - support_2_embedding), 
        np.linalg.norm(query_6_embedding - support_3_embedding), 
        np.linalg.norm(query_6_embedding - support_4_embedding), 
        np.linalg.norm(query_6_embedding - support_5_embedding), 
        np.linalg.norm(query_6_embedding - support_6_embedding),
        np.linalg.norm(query_6_embedding - support_7_embedding),
        np.linalg.norm(query_6_embedding - support_8_embedding)]
print("attack number:", dis_query_6.index(min(dis_query_6)))

print("distance for query_7:")
dis_query_7 = [np.linalg.norm(query_7_embedding - support_0_embedding),
        np.linalg.norm(query_7_embedding - support_1_embedding), 
        np.linalg.norm(query_7_embedding - support_2_embedding), 
        np.linalg.norm(query_7_embedding - support_3_embedding), 
        np.linalg.norm(query_7_embedding - support_4_embedding), 
        np.linalg.norm(query_7_embedding - support_5_embedding), 
        np.linalg.norm(query_7_embedding - support_6_embedding),
        np.linalg.norm(query_7_embedding - support_7_embedding),
        np.linalg.norm(query_7_embedding - support_8_embedding)]
print("attack number:", dis_query_7.index(min(dis_query_7)))

print("distance for query_8:")
dis_query_8 = [np.linalg.norm(query_7_embedding - support_0_embedding),
        np.linalg.norm(query_8_embedding - support_1_embedding), 
        np.linalg.norm(query_8_embedding - support_2_embedding), 
        np.linalg.norm(query_8_embedding - support_3_embedding), 
        np.linalg.norm(query_8_embedding - support_4_embedding), 
        np.linalg.norm(query_8_embedding - support_5_embedding), 
        np.linalg.norm(query_8_embedding - support_6_embedding),
        np.linalg.norm(query_8_embedding - support_7_embedding),
        np.linalg.norm(query_8_embedding - support_8_embedding)]
print("attack number:", dis_query_8.index(min(dis_query_8)))

#%% Tsne results

with torch.no_grad():
    X_1 = torch.from_numpy(X_1)
    X_1.to(device, dtype=torch.float)
    pred = model(X_1)
    X_1_embedding = pred.cpu().numpy().mean(axis=0)
    X_1_embedding = np.expand_dims(X_1_embedding, axis=0)
    
    X_2 = torch.from_numpy(X_2)
    X_2.to(device, dtype=torch.float)
    pred = model(X_2)
    X_2_embedding = pred.cpu().numpy().mean(axis=0)
    X_2_embedding = np.expand_dims(X_2_embedding, axis=0)
    
    X_3 = torch.from_numpy(X_3)
    X_3.to(device, dtype=torch.float)
    pred = model(X_3)
    X_3_embedding = pred.cpu().numpy().mean(axis=0)
    X_3_embedding = np.expand_dims(X_3_embedding, axis=0)
    
    X_4 = torch.from_numpy(X_4)
    X_4.to(device, dtype=torch.float)
    pred = model(X_4)
    X_4_embedding = pred.cpu().numpy().mean(axis=0)
    X_4_embedding = np.expand_dims(X_4_embedding, axis=0)
    
    X_5 = torch.from_numpy(X_5)
    X_5.to(device, dtype=torch.float)
    pred = model(X_5)
    X_5_embedding = pred.cpu().numpy().mean(axis=0)
    X_5_embedding = np.expand_dims(X_5_embedding, axis=0)
    
    X_6 = torch.from_numpy(X_6)
    X_6.to(device, dtype=torch.float)
    pred = model(X_6)
    X_6_embedding = pred.cpu().numpy().mean(axis=0)
    X_6_embedding = np.expand_dims(X_6_embedding, axis=0)


X_embedding = np.concatenate((X_1_embedding, X_2_embedding, X_3_embedding, X_4_embedding, X_5_embedding, X_6_embedding), axis=0)

# X_embedding = np.array([X_1_embedding, X_2_embedding, X_3_embedding, X_4_embedding, X_5_embedding, X_6_embedding])

y_embedding = np.concatenate((y_1, y_2, y_3, y_4, y_5, y_6), axis=0)

# y_embedding = np.array([y_1.mean(), y_2.mean(), y_3.mean(), y_4.mean(), y_5.mean(), y_6.mean()])
n_components = 2
tsne = TSNE(n_components)

tsne_result = tsne.fit_transform(X_embedding)

tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y_embedding})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
# %%
with torch.no_grad():
    X_0 = torch.from_numpy(X_0)
    X_0.to(device, dtype=torch.float)
    pred = model(X_0)
    X_0_embedding = pred.cpu().numpy()
    
    X_7 = torch.from_numpy(X_7)
    X_7.to(device, dtype=torch.float)
    pred = model(X_7)
    X_7_embedding = pred.cpu().numpy()
    


X_embedding = np.concatenate((X_0_embedding, X_7_embedding), axis=0)
y_embedding = np.concatenate((y_0, y_7), axis=0)
n_components = 2
tsne = TSNE(n_components)

tsne_result = tsne.fit_transform(X_embedding)

tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y_embedding})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
# %%
