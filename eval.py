'''
Author: Qi7
Date: 2023-04-07 23:27:47
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-10 00:28:21
Description: 
'''
#%%
import numpy as np
from matplotlib import pyplot as plt

random_history = np.load("saved_models/multiclass_epochs100_lr_0.001_bs_256_history.npy", allow_pickle=True).item()
# %%
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(random_history["test_f1"])
ax.set(title='Test Accuracy', xlabel='Epochs', 
       ylabel='Accuracy');
# %%
