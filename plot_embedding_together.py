#%%
import numpy as np
import torch
from torch.utils.data import DataLoader
from loader import waveformDataset
from model import DistanceNet, QNN
from util import extract_embeddings, plot_embeddings, plot_2d_embeddings

X = np.load('dataset/8cases_jinan/new_test_set/X_test.npy')
y = np.load('dataset/8cases_jinan/new_test_set/y_test.npy')

# X = X[np.where(y == 6)[0]]
# y = y[np.where(y == 6)[0]]


X_query = np.load('dataset/8cases_jinan/new_query_set/X_embedding_plot.npy')
y_query = np.load('dataset/8cases_jinan/new_query_set/y_embedding_plot.npy')
X_8 = X_query[np.where(y_query == 8)[0]]
X_9 = X_query[np.where(y_query == 9)[0]]
X_10 = X_query[np.where(y_query == 10)[0]]
X_11 = X_query[np.where(y_query == 11)[0]]
X_12 = X_query[np.where(y_query == 12)[0]]

y_8 = y_query[np.where(y_query == 8)[0]]
y_9 = y_query[np.where(y_query == 9)[0]]
y_10 = y_query[np.where(y_query == 10)[0]]
y_11 = y_query[np.where(y_query == 11)[0]]
y_12 = y_query[np.where(y_query == 12)[0]]


# X = np.concatenate((X, X_12), axis=0)
# y = np.concatenate((y, y_12))


trained_model = "saved_models/2d_snn/margin_1.0_epoch_300_contrastive_model.pth"

testset = waveformDataset(X, y)
batch_size = 128
test_data_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

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


test_embeddings, test_labels = extract_embeddings(test_data_loader, model, device)
plot_2d_embeddings(test_embeddings, test_labels)
# %%
