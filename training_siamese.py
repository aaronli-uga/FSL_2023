'''
Author: Qi7
Date: 2023-05-23 17:06:32
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-24 16:49:22
Description: training the SNN model
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn as nn
import copy

from loader import SiameseDataset
from model import SiameseNet
from training import model_train_multiclass

save_model_path = "saved_models/snn/"
X = np.load('dataset/8cases_jinan/training_set/X_norm.npy')
y = np.load('dataset/8cases_jinan/training_set/y.npy')

X_train, X_cv, y_train, y_cv = train_test_split(X, y, train_size=0.75, test_size=0.25, shuffle=True, random_state=27)
trainset = SiameseDataset(X_train, y_train)
validset = SiameseDataset(X_cv, y_cv)

# Hyper parameters
batch_size = 128
learning_rate = 0.001
num_epochs = 50
history = dict(test_loss=[], train_loss=[], test_acc=[], test_f1=[], test_f1_all=[])

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

model = SiameseNet(
                n_input_channels=6,
                n_output_channels=64,
                kernel_size=3,
                stride=1,
                n_classes=8
            )

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
margin = 10.0 # parameters for penalizing
triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
best_loss = np.inf
best_weights = None

for epoch in range(num_epochs):
    # training
    model.train()
    total_loss = 0.0
    for batch_idx, (sample1, sample2, sample3, label_pos, label_neg) in enumerate(data_loader):
        sample1, sample2, sample3 = sample1.to(device), sample2.to(device), sample3.to(device)
        label_pos, label_neg = label_pos.to(device), label_neg.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output1, output2 = model(sample1, sample2)
        _, output3 = model(sample1, sample3)
        
        # Compute the contrastive loss
        # euclidean_distance_p = nn.functional.pairwise_distance(output1, output2)
        # euclidean_distance_n = nn.functional.pairwise_distance(output1, output3)
        
        # triple loss
        loss_contrastive = triplet_loss(output1, output2, output3)
        
        # loss_contrastive = torch.mean(label_pos * torch.pow(euclidean_distance, 2) +
                                    #   label_neg * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
        
        # Backward pass and optimization
        loss_contrastive.backward()
        optimizer.step()
        
        total_loss += loss_contrastive.item()
        
        if batch_idx % 20 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, batch_idx+1, len(data_loader), loss_contrastive.item()))
            
    history['train_loss'].append(total_loss / len(data_loader))

    print('Epoch [{}/{}], Average Train Loss: {:.4f}'
          .format(epoch+1, num_epochs, total_loss / len(data_loader)))

    
    # validation
    model.eval()
    total_loss = 0.0
    for batch_idx, (sample1, sample2, sample3, label_pos, label_neg) in enumerate(test_data_loader):
        sample1, sample2, sample3 = sample1.to(device), sample2.to(device), sample3.to(device)
        label_pos, label_neg = label_pos.to(device), label_neg.to(device)
        
        output1, output2 = model(sample1, sample2)
        _, output3 = model(sample1, sample3)
        loss_contrastive = triplet_loss(output1, output2, output3)
        total_loss += loss_contrastive.item()
    
    test_loss = total_loss / len(test_data_loader)
    if best_loss > test_loss:
        best_loss = test_loss
        best_weights = copy.deepcopy(model.state_dict())
    history['test_loss'].append(test_loss)

    print('Epoch [{}/{}], Average Test Loss: {:.4f}'
          .format(epoch+1, num_epochs, test_loss))

model.load_state_dict(best_weights)
torch.save(model.state_dict(), save_model_path + f"snn_margin{margin}_8cases_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_best_model.pth")
np.save(save_model_path + f"snn_margin{margin}_8cases_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_history.npy", history)




# # Set up the Siamese network, optimizer, and loss function
# siamese_net = SiameseNet()
# optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)
# margin = 1.0

# # Set device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# siamese_net.to(device)

# # Prepare your data and targets
# train_data = ...
# train_targets = ...

# # Create the Siamese dataset and data loader
# siamese_dataset = SiameseDataset(train_data, train_targets)
# data_loader = DataLoader(siamese_dataset, batch_size=64, shuffle=True)

# # Training loop
