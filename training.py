'''
Author: Qi7
Date: 2023-04-07 11:13:41
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-07 17:39:54
Description: helper function for training the model
'''
import numpy as np
import copy
import torch
import torch.nn as nn 
import tqdm

def model_train(model, train_loader, val_loader, num_epochs, optimizer, device, history):
    """_summary_

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_
        val_loader (_type_): _description_
        num_epochs (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_
        history (_type_): the statistic information such as accuracy, f1-score, loss

    Returns:
        _type_: statistical information. history
    """
    # collect statistics
    train_loss = []
    train_acc = []
    val_acc = []
    
    loss_fn = nn.BCELoss()
    
    best_acc = -np.inf
    best_weights = None
    
    for epoch in range(num_epochs):
        train_loss = []
        train_acc = []
        model.train()
        # with tqdm.tqdm(unit="batch", total=len(train_loader), disable=True) as bar:
        #     bar.set_description(f"Epoch {epoch}")
        for X_batch, y_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"):
            # forward pass
            X_batch = X_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.float32)
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
        
        # evaluate at end of each epoch
        model.eval()
        for X_val, y_val in val_loader:
            X_val = X_val.to(device, dtype=torch.float32)
            y_val = y_val.to(device, dtype=torch.float32)
            y_pred = model(X_val)
            acc = (y_pred.round() == y_val).float().mean()
            
            history['train_loss'].append(sum(train_loss) / len(train_loss))
            history['train_acc'].append(sum(train_acc) / len(train_acc))
            history['test_acc'].append(float(acc))
            print(f"train acc: {history['train_acc'][-1]}. val acc: {history['test_acc'][-1]}")
            print(f"train loss: {history['train_loss'][-1]}")
            
            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
                
            # only validate one batch
            break
    
    print(f"Training Done. best acc: {best_acc}")
    model.load_state_dict(best_weights)
    
    return train_loss, train_acc, val_acc
                