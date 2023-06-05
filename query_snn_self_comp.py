'''
Author: Qi7
Date: 2023-05-16 23:28:18
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-05 11:38:06
Description: Test the accuracy from the query set. Number of shots and number of query is provided.
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from loader import waveformDataset
from model import DistanceNet, QNN
from util import extract_embeddings, plot_2d_embeddings


# load dataset
X = np.load('dataset/8cases_jinan/new_query_set/X_embedding_plot.npy')
y = np.load('dataset/8cases_jinan/new_query_set/y_embedding_plot.npy') # label from 8 to 14

X_test = np.load('dataset/8cases_jinan/new_test_set/X_test.npy') # label from 0 to 7
y_test = np.load('dataset/8cases_jinan/new_test_set/y_test.npy')

X_0 = X_test[np.where(y_test == 0)[0]]
y_0 = y_test[np.where(y_test == 0)[0]]

X_1 = X_test[np.where(y_test == 1)[0]]
y_1 = y_test[np.where(y_test == 1)[0]]

X_2 = X_test[np.where(y_test == 2)[0]]
y_2 = y_test[np.where(y_test == 2)[0]]

X_3 = X_test[np.where(y_test == 3)[0]]
y_3 = y_test[np.where(y_test == 3)[0]]

X_4 = X_test[np.where(y_test == 4)[0]]
y_4 = y_test[np.where(y_test == 4)[0]]

X_5 = X_test[np.where(y_test == 5)[0]]
y_5 = y_test[np.where(y_test == 5)[0]]

X_6 = X_test[np.where(y_test == 6)[0]]
y_6 = y_test[np.where(y_test == 6)[0]]

X_7 = X_test[np.where(y_test == 7)[0]]
y_7 = y_test[np.where(y_test == 7)[0]]

X_8 = X[np.where(y == 8)[0]]
y_8 = y[np.where(y == 8)[0]]

X_9 = X[np.where(y == 9)[0]]
y_9 = y[np.where(y == 9)[0]]

X_10 = X[np.where(y == 10)[0]]
y_10 = y[np.where(y == 10)[0]]

X_11 = X[np.where(y == 11)[0]]
y_11 = y[np.where(y == 11)[0]]

X_12 = X[np.where(y == 12)[0]]
y_12 = y[np.where(y == 12)[0]]


# randomly select the support set.
num_shots = 30
num_query = 500

# Prepare the support set
rand_index = np.random.randint(0, X_0.shape[0], num_shots)
support_0 = X_0[rand_index]
support_0_label = y_0[rand_index]
testset = waveformDataset(support_0, support_0_label)
support_0_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_1.shape[0], num_shots)
support_1 = X_1[rand_index]
support_1_label = y_1[rand_index]
testset = waveformDataset(support_1, support_1_label)
support_1_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_2.shape[0], num_shots)
support_2 = X_2[rand_index]
support_2_label = y_2[rand_index]
testset = waveformDataset(support_2, support_2_label)
support_2_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_3.shape[0], num_shots)
support_3 = X_3[rand_index]
support_3_label = y_3[rand_index]
testset = waveformDataset(support_3, support_3_label)
support_3_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_4.shape[0], num_shots)
support_4 = X_4[rand_index]
support_4_label = y_4[rand_index]
testset = waveformDataset(support_4, support_4_label)
support_4_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_5.shape[0], num_shots)
support_5 = X_5[rand_index]
support_5_label = y_5[rand_index]
testset = waveformDataset(support_5, support_5_label)
support_5_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_6.shape[0], num_shots)
support_6 = X_6[rand_index]
support_6_label = y_6[rand_index]
testset = waveformDataset(support_6, support_6_label)
support_6_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_7.shape[0], num_shots)
support_7 = X_7[rand_index]
support_7_label = y_7[rand_index]
testset = waveformDataset(support_7, support_7_label)
support_7_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_8.shape[0], num_shots)
support_8 = X_8[rand_index]
support_8_label = y_8[rand_index]
testset = waveformDataset(support_8, support_8_label)
support_8_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_9.shape[0], num_shots)
support_9 = X_9[rand_index]
support_9_label = y_9[rand_index]
testset = waveformDataset(support_9, support_9_label)
support_9_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_10.shape[0], num_shots)
support_10 = X_10[rand_index]
support_10_label = y_10[rand_index]
testset = waveformDataset(support_10, support_10_label)
support_10_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_11.shape[0], num_shots)
support_11 = X_11[rand_index]
support_11_label = y_11[rand_index]
testset = waveformDataset(support_11, support_11_label)
support_11_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_12.shape[0], num_shots)
support_12 = X_12[rand_index]
support_12_label = y_12[rand_index]
testset = waveformDataset(support_12, support_12_label)
support_12_loader = DataLoader(testset, batch_size=128, shuffle=False)

# prepare the query set
rand_index = np.random.randint(0, X_8.shape[0], num_query)
query_8 = X_8[rand_index]
query_8_label = y_8[rand_index]
testset = waveformDataset(query_8, query_8_label)
query_8_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_9.shape[0], num_query)
query_9 = X_9[rand_index]
query_9_label = y_9[rand_index]
testset = waveformDataset(query_9, query_9_label)
query_9_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_10.shape[0], num_query)
query_10 = X_10[rand_index]
query_10_label = y_10[rand_index]
testset = waveformDataset(query_10, query_10_label)
query_10_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_11.shape[0], num_query)
query_11 = X_11[rand_index]
query_11_label = y_11[rand_index]
testset = waveformDataset(query_11, query_11_label)
query_11_loader = DataLoader(testset, batch_size=128, shuffle=False)

rand_index = np.random.randint(0, X_12.shape[0], num_query)
query_12 = X_12[rand_index]
query_12_label = y_12[rand_index]
testset = waveformDataset(query_12, query_12_label)
query_12_loader = DataLoader(testset, batch_size=128, shuffle=False)



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




support_0_embedding, support_0_label_embedding = extract_embeddings(support_0_loader, model, device)
support_1_embedding, support_1_label_embedding = extract_embeddings(support_1_loader, model, device)
support_2_embedding, support_2_label_embedding = extract_embeddings(support_2_loader, model, device)
support_3_embedding, support_3_label_embedding = extract_embeddings(support_3_loader, model, device)
support_4_embedding, support_4_label_embedding = extract_embeddings(support_4_loader, model, device)
support_5_embedding, support_5_label_embedding = extract_embeddings(support_5_loader, model, device)
support_6_embedding, support_6_label_embedding = extract_embeddings(support_6_loader, model, device)
support_7_embedding, support_7_label_embedding = extract_embeddings(support_7_loader, model, device)
support_8_embedding, support_8_label_embedding = extract_embeddings(support_8_loader, model, device)
support_9_embedding, support_9_label_embedding = extract_embeddings(support_9_loader, model, device)
support_10_embedding, support_10_label_embedding = extract_embeddings(support_10_loader, model, device)
support_11_embedding, support_11_label_embedding = extract_embeddings(support_11_loader, model, device)
support_12_embedding, support_12_label_embedding = extract_embeddings(support_12_loader, model, device)

support_0_embedding = support_0_embedding.mean(axis=0)
support_1_embedding = support_1_embedding.mean(axis=0)
support_2_embedding = support_2_embedding.mean(axis=0)
support_3_embedding = support_3_embedding.mean(axis=0)
support_4_embedding = support_4_embedding.mean(axis=0)
support_5_embedding = support_5_embedding.mean(axis=0)
support_6_embedding = support_6_embedding.mean(axis=0)
support_7_embedding = support_7_embedding.mean(axis=0)
support_8_embedding = support_8_embedding.mean(axis=0)
support_9_embedding = support_9_embedding.mean(axis=0)
support_10_embedding = support_10_embedding.mean(axis=0)
support_11_embedding = support_11_embedding.mean(axis=0)
support_12_embedding = support_12_embedding.mean(axis=0)

query_8_embedding, query_8_label_embedding = extract_embeddings(query_8_loader, model, device)
query_9_embedding, query_9_label_embedding = extract_embeddings(query_9_loader, model, device)
query_10_embedding, query_10_label_embedding = extract_embeddings(query_10_loader, model, device)
query_11_embedding, query_11_label_embedding = extract_embeddings(query_11_loader, model, device)
query_12_embedding, query_12_label_embedding = extract_embeddings(query_12_loader, model, device)



# accuracy calculation
cnt = 0
for i in range(num_query):
    dis_query_8 = [
        np.linalg.norm(query_8_embedding[i] - support_9_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_10_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_11_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_12_embedding), 
        np.linalg.norm(query_8_embedding[i] - support_8_embedding), 
    ]
    if dis_query_8.index(min(dis_query_8)) == 4:
        cnt += 1
print("query 8 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_9 = [
        np.linalg.norm(query_9_embedding[i] - support_8_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_10_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_11_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_12_embedding), 
        np.linalg.norm(query_9_embedding[i] - support_9_embedding), 
    ]
    if dis_query_9.index(min(dis_query_9)) == 4:
        cnt += 1
print("query 9 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_10 = [
        np.linalg.norm(query_10_embedding[i] - support_8_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_9_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_11_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_12_embedding), 
        np.linalg.norm(query_10_embedding[i] - support_10_embedding),
    ]
    if dis_query_10.index(min(dis_query_10)) == 4:
        cnt += 1
print("query 10 accuracy:", cnt / num_query)

cnt = 0
for i in range(num_query):
    dis_query_11 = [
        np.linalg.norm(query_11_embedding[i] - support_8_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_9_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_10_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_12_embedding), 
        np.linalg.norm(query_11_embedding[i] - support_11_embedding), 
    ]
    if dis_query_11.index(min(dis_query_11)) == 4:
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
    ]
    if dis_query_12.index(min(dis_query_12)) == 4:
        cnt += 1
print("query 12 accuracy:", cnt / num_query)
# %%
