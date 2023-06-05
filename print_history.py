'''
Author: Qi7
Date: 2023-04-07 23:27:47
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-04 18:30:38
Description: print out the history information of trained model, such as test loss and test f1 score.
'''
#%%
import numpy as np
from matplotlib import pyplot as plt

# trained_model = "saved_models/snn/2_loss_last_round_2d_snn_margin2_8cases_epochs30_lr_0.001_bs_128_best_model.pth"
# trained_model = "saved_models/snn/new_2_loss_2d_snn_margin2_8cases_epochs30_lr_0.001_bs_128_history.npy"
trained_model = "saved_models/2d_snn/margin_1.0_epoch_300_contrastive_history.npy"

history = np.load(trained_model, allow_pickle=True).item()


# %%
# fig, ax = plt.subplots(figsize=(12,6))
# ax.plot(history["test_f1"])
# ax.set(title='Test F1', xlabel='Epochs', ylabel='F1')

# fig, ax = plt.subplots(figsize=(12,6))
# ax.plot(history["test_acc"])
# ax.set(title='Test accuracy', xlabel='Epochs', ylabel='accuracy')


fig, ax = plt.subplots(figsize=(12,6))
ax.plot(history["val_loss"], label="test loss")
ax.plot(history["train_loss"], label="train loss")
ax.legend()
ax.set(title='loss', xlabel='Epochs', ylabel='loss')

# %%

# find the index of max f1 score
np.array(history['val_loss']).argmin()
# %%
np.array(history['test_f1_all'][42])
# %%
