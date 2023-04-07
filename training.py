'''
Author: Qi7
Date: 2023-04-07 11:13:41
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-07 12:21:15
Description: helper function for training the model
'''
import numpy as np
import copy
import torch
import torch.nn as nn 
import tqdm

def model_train(model, train_loader, val_loader, learning_rate, num_epochs, optimizer):
    # collect statistics
    train_loss = []
    train_acc = []
    val_acc = []
    
    loss_fn = nn.CrossEntropyLoss()
    
    best_acc = -np.inf
    best_weights = None
    
    for epoch in range(num_epochs):
        model.train()
        with tqdm.tqdm(unit="batch", total=len(train_loader), disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for X_batch, y_batch in train_loader:
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                acc = (y_pred.round() == y_batch).float().mean()
                
                # store metrics
                train_loss.append(float(loss))
                train_acc.append(float(acc))
                
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # update weights
                optimizer.step()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        
        # evaluate at end of each epoch
        model.eval()
        for X_val, y_val in val_loader:
            y_pred = model(X_val)
            acc = (y_pred.round() == y_val).float().mean()
                