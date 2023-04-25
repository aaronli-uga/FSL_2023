'''
Author: Qi7
Date: 2023-04-07 23:27:47
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-25 15:47:06
Description: 
'''
#%%
import numpy as np
from matplotlib import pyplot as plt

history = np.load("saved_models/epochs500_lr_0.001_bs_256_history.npy", allow_pickle=True).item()
# %%
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(history["test_f1"])
ax.set(title='Test F1', xlabel='Epochs', ylabel='F1')

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(history["test_acc"])
ax.set(title='Test accuracy', xlabel='Epochs', ylabel='accuracy')


fig, ax = plt.subplots(figsize=(12,6))
ax.plot(history["test_loss"], label="test loss")
ax.plot(history["train_loss"], label="train loss")
ax.legend()
ax.set(title='loss', xlabel='Epochs', ylabel='accuracy')

# %%

# find the index of max f1 score
np.array(history['test_f1']).argmax()
# %%
