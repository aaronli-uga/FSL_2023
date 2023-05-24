'''
Author: Qi7
Date: 2023-05-17 16:29:18
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-22 13:47:42
Description: using sliding window algorithm to convert the csv to the trainable data and do the normalization.
'''
import pandas as pd
import numpy as np
from util import sliding_windows

# files = ["attack11", "attack16", "attack18", "attack27", "attack50", "attack58", "attack78"]
# for file in files:
#     data = f"dataset/8cases/query_set/{file}.csv"
#     df = pd.read_csv(data)
#     data_npy = df.to_numpy()
#     o1, _ = sliding_windows(data_npy, sub_window_size=2000, step_size=20)
#     np.save(f"dataset/8cases/query_set/X_{file}.npy", o1)

# X_8 = np.load("dataset/8cases/query_set/X_attack11.npy")
# X_9 = np.load("dataset/8cases/query_set/X_attack16.npy")
# X_10 = np.load("dataset/8cases/query_set/X_attack18.npy")
# X_11 = np.load("dataset/8cases/query_set/X_attack27.npy")
# X_12 = np.load("dataset/8cases/query_set/X_attack50.npy")
# X_13 = np.load("dataset/8cases/query_set/X_attack58.npy")
# X_14 = np.load("dataset/8cases/query_set/X_attack78.npy")

# y_8 = np.ones(X_8.shape[0]) * 8
# y_9 = np.ones(X_9.shape[0]) * 9
# y_10 = np.ones(X_10.shape[0]) * 10
# y_11 = np.ones(X_11.shape[0]) * 11
# y_12 = np.ones(X_12.shape[0]) * 12
# y_13 = np.ones(X_13.shape[0]) * 13
# y_14 = np.ones(X_14.shape[0]) * 14

# X = np.concatenate((X_8, X_9, X_10, X_11, X_12, X_13, X_14), axis=0)

X = np.load("dataset/8cases/X.npy")
X = np.transpose(X, (0, 2, 1))
# np.save(f"dataset/8cases/query_set/X.npy", X)

# y = np.concatenate((y_8, y_9, y_10, y_11, y_12, y_13, y_14))
# Standard Normalization ((X-mean) / std)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i,j,:] = (X[i,j,:] - X[i,j,:].mean()) / X[i,j,:].std()


np.save(f"dataset/8cases/X_norm.npy", X)
# np.save(f"dataset/8cases/query_set/y.npy", y)