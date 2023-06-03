#%%
import numpy as np
import torch
from torch.utils.data import DataLoader
from loader import waveformDataset
from model import DistanceNet, QNN
from util import extract_embeddings, plot_embeddings

X = np.load('dataset/8cases_jinan/new_test_set/X_test.npy')
y = np.load('dataset/8cases_jinan/new_test_set/y_test.npy')

trained_model = "saved_models/new_snn/contrastive_loss_model.pth"

testset = waveformDataset(X, y)
batch_size = 128
test_data_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

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
plot_embeddings(test_embeddings, test_labels)
# %%
