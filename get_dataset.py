'''
Author: Qi7
Date: 2023-05-17 16:29:18
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-17 17:17:17
Description: 
'''
import pandas as pd
import numpy as np
from util import sliding_windows

# files = ["normal", "attack10", "attack15", "attack17", "attack26", "attack49", "attack57", "attack77"]
# for file in files:
#     data = f"dataset/8cases/{file}.csv"
#     df = pd.read_csv(data)
#     data_npy = df.to_numpy()
#     o1, _ = sliding_windows(data_npy, sub_window_size=2000, step_size=20)
#     np.save(f"dataset/8cases/X_{file}.npy", o1)

# X_0 = np.load("dataset/8cases/X_normal.npy")
# X_1 = np.load("dataset/8cases/X_attack10.npy")
# X_2 = np.load("dataset/8cases/X_attack15.npy")
# X_3 = np.load("dataset/8cases/X_attack17.npy")
# X_4 = np.load("dataset/8cases/X_attack26.npy")
# X_5 = np.load("dataset/8cases/X_attack49.npy")
# X_6 = np.load("dataset/8cases/X_attack57.npy")
# X_7 = np.load("dataset/8cases/X_attack77.npy")
# X = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7), axis=0)
X = np.load("dataset/8cases/X.npy")
X = np.transpose(X, (0, 2, 1))

# Standard Normalization ((X-mean) / std)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i,j,:] = (X[i,j,:] - X[i,j,:].mean()) / X[i,j,:].std()


np.save(f"dataset/8cases/X_norm.npy", X)