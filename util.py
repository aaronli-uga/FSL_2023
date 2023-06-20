'''
Author: Qi7
Date: 2023-05-17 16:24:45
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-12 01:13:08
Description: utils function for sliding window
'''
# %matplotlib inline
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def sliding_windows(array, sub_window_size, step_size, start_index=0):
    """return the sliding window sized matrix. (preprocessing)
    input: array is a list with dimension of m x n. m is the timestamp and n is the features number.
    output: 1. the window data 2. window+1 data represent the predict target
    """
    array = np.array(array)
    start = start_index
    num_windows = len(array) - start - 1
    sub_windows = (
        start +
        np.expand_dims(np.arange(sub_window_size), axis=0) +
        np.expand_dims(np.arange(num_windows - sub_window_size + 1), 0).T
    )
    target_index = list(range(start + sub_window_size, len(array), step_size))
    
    return array[sub_windows[::step_size]], array[target_index]


# testing
# x = [[1,11,111,1111],[5,6,7,8],[9,10,11,12],[2,3,4,5],[5,4,3,2],[2,3,4,1],[4,5,3,2],[2,3,1,4],[2,3,4,1],[2,3,4,1]]
# o1, o2 = sliding_windows(x, sub_window_size=3, step_size=1)

# # plot embedding for test set, 8 classes in total
# mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# # plot embedding for query set, 6 classes in total
# mnist_classes = ['8', '9', '10', '11', '12']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#               '#9467bd']

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '12']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#27408B']


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    tsne = TSNE(n_components = 2, random_state=27)
    embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10,10))
    for i in range(8):
    # for i in range(8, 14): # for query embedding plot
        inds = np.where(targets==i)[0]
        # plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i - 8])
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # plt.legend(mnist_classes, fontsize = 23)
    
def plot_2d_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(8):
    # for i in range(8, 13): # for query embedding plot
        if i == 8:
            inds = np.where(targets==12)[0]
        else:
            inds = np.where(targets==i)[0]
        # plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i - 8])
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.8, color=colors[i], s=100)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(mnist_classes, fontsize = 25)


def extract_embeddings(dataloader, model, device):
    with torch.no_grad():
        model.to(device)
        model.eval()
        # embeddings = np.zeros((len(dataloader.dataset), 32))
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for samples, target in dataloader:
            samples = samples.to(device, dtype=torch.float32)
            embeddings[k:k+len(samples)] = model.get_embedding(samples).data.cpu().numpy()
            labels[k:k+len(samples)] = target.numpy()
            k += len(samples)
            # break
    return embeddings, labels