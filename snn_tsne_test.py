'''
Author: Qi7
Date: 2023-05-21 22:20:10
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-24 16:51:03
Description: SNN model. test the tsne result to see if the features embedding could be distinguished with each other
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
import pandas as pd
import seaborn as sns

from loader import waveformDataset
from model import SiameseNet

X = np.load('dataset/8cases/X_test.npy')
y = np.load('dataset/8cases/y_test.npy')

X_0 = X[np.where(y == 0)[0]]
X_1 = X[np.where(y == 1)[0]]
X_2 = X[np.where(y == 2)[0]]
X_3 = X[np.where(y == 3)[0]]
X_4 = X[np.where(y == 4)[0]]
X_5 = X[np.where(y == 5)[0]]
X_6 = X[np.where(y == 6)[0]]
X_7 = X[np.where(y == 7)[0]]

y_0 = y[np.where(y == 0)[0]]
y_1 = y[np.where(y == 1)[0]]
y_2 = y[np.where(y == 2)[0]]
y_3 = y[np.where(y == 3)[0]]
y_4 = y[np.where(y == 4)[0]]
y_5 = y[np.where(y == 5)[0]]
y_6 = y[np.where(y == 6)[0]]
y_7 = y[np.where(y == 7)[0]]

num_query = 500
r = np.random.randint(0, X_0.shape[0], num_query)
query_0, query_0_y = X_0[r], y_0[r]

r = np.random.randint(0, X_1.shape[0], num_query)
query_1, query_1_y = X_1[r], y_1[r]

r = np.random.randint(0, X_2.shape[0], num_query)
query_2, query_2_y = X_2[r], y_2[r]

r = np.random.randint(0, X_3.shape[0], num_query)
query_3, query_3_y = X_3[r], y_3[r]

r = np.random.randint(0, X_4.shape[0], num_query)
query_4, query_4_y = X_4[r], y_4[r]

r = np.random.randint(0, X_5.shape[0], num_query)
query_5, query_5_y = X_5[r], y_5[r]

r = np.random.randint(0, X_6.shape[0], num_query)
query_6, query_6_y = X_6[r], y_6[r]

r = np.random.randint(0, X_7.shape[0], num_query)
query_7, query_7_y = X_7[r], y_7[r]

trained_model = "saved_models/snn/snn_margin0.5_8cases_epochs25_lr_0.001_bs_128_best_model.pth"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = SiameseNet(n_input_channels=6,
            n_output_channels=64,
            kernel_size=3,
            stride=1,
            n_classes=8)
model.load_state_dict(torch.load(trained_model))
# model.fc = torch.nn.Flatten() # get the embedding
model.to(device)

with torch.no_grad():

    query_0 = torch.from_numpy(query_0)
    query_0 = query_0.to(device, dtype=torch.float)
    pred, _ = model(query_0, query_0)
    query_0_embedding = pred.cpu().numpy()
    
    query_1 = torch.from_numpy(query_1)
    query_1 = query_1.to(device, dtype=torch.float)
    pred, _ = model(query_1, query_1)
    query_1_embedding = pred.cpu().numpy()
    # query_1_embedding = np.expand_dims(query_1_embedding, axis=0)

    query_2 = torch.from_numpy(query_2)
    query_2 = query_2.to(device, dtype=torch.float)
    pred, _ = model(query_2, query_2)
    query_2_embedding = pred.cpu().numpy()
    # query_2_embedding = np.expand_dims(query_2_embedding, axis=0)

    query_3 = torch.from_numpy(query_3)
    query_3 = query_3.to(device, dtype=torch.float)
    pred, _ = model(query_3, query_3)
    query_3_embedding = pred.cpu().numpy()
    # query_3_embedding = np.expand_dims(query_3_embedding, axis=0)

    query_4 = torch.from_numpy(query_4)
    query_4 = query_4.to(device, dtype=torch.float)
    pred, _ = model(query_4, query_4)
    query_4_embedding = pred.cpu().numpy()
    # query_4_embedding = np.expand_dims(query_4_embedding, axis=0)

    query_5 = torch.from_numpy(query_5)
    query_5 = query_5.to(device, dtype=torch.float)
    pred, _ = model(query_5, query_5)
    query_5_embedding = pred.cpu().numpy()
    # query_5_embedding = np.expand_dims(query_5_embedding, axis=0)

    query_6 = torch.from_numpy(query_6)
    query_6 = query_6.to(device, dtype=torch.float)
    pred, _ = model(query_6, query_6)
    query_6_embedding = pred.cpu().numpy()
    # query_6_embedding = np.expand_dims(query_6_embedding, axis=0)

    query_7 = torch.from_numpy(query_7)
    query_7 = query_7.to(device, dtype=torch.float)
    pred, _ = model(query_7, query_7)
    query_7_embedding = pred.cpu().numpy()
    # query_7_embedding = np.expand_dims(query_7_embedding, axis=0)
    
X_embedding = np.concatenate((query_1_embedding, query_2_embedding, query_3_embedding, query_4_embedding, query_5_embedding, query_6_embedding, query_7_embedding,), axis=0)

y_embedding = np.concatenate((query_1_y, query_2_y, query_3_y, query_4_y, query_5_y, query_6_y, query_7_y), axis=0)

n_components = 2
tsne = TSNE(n_components)

tsne_result = tsne.fit_transform(X_embedding)

tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y_embedding})
fig, ax = plt.subplots(1, figsize=(15,10))
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
# %%
