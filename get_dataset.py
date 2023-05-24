'''
Author: Qi7
Date: 2023-05-17 16:29:18
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-24 16:20:20
Description: using sliding window algorithm to convert the csv to the trainable data and do the normalization.
'''
import pandas as pd
import numpy as np
from util import sliding_windows

files = ["normal", "attack10", "attack15", "attack17", "attack26", "attack49", "attack57", "attack77"]
for file in files:
    data = f"dataset/8cases_jinan/raw_dataset_training/{file}.csv"
    df = pd.read_csv(data)
    data_npy = df.to_numpy()
    o1, _ = sliding_windows(data_npy, sub_window_size=2000, step_size=20)
    np.save(f"dataset/8cases_jinan/training_set/X_{file}.npy", o1)

X_0 = np.load("dataset/8cases_jinan/training_set/X_normal.npy")
X_1 = np.load("dataset/8cases_jinan/training_set/X_attack10.npy")
X_2 = np.load("dataset/8cases_jinan/training_set/X_attack15.npy")
X_3 = np.load("dataset/8cases_jinan/training_set/X_attack17.npy")
X_4 = np.load("dataset/8cases_jinan/training_set/X_attack26.npy")
X_5 = np.load("dataset/8cases_jinan/training_set/X_attack49.npy")
X_6 = np.load("dataset/8cases_jinan/training_set/X_attack57.npy")
X_7 = np.load("dataset/8cases_jinan/training_set/X_attack77.npy")

y_0 = np.ones(X_0.shape[0]) * 0
y_1 = np.ones(X_1.shape[0]) * 1
y_2 = np.ones(X_2.shape[0]) * 2
y_3 = np.ones(X_3.shape[0]) * 3
y_4 = np.ones(X_4.shape[0]) * 4
y_5 = np.ones(X_5.shape[0]) * 5
y_6 = np.ones(X_6.shape[0]) * 6
y_7 = np.ones(X_7.shape[0]) * 7

X = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7), axis=0)

# X = np.load("dataset/8cases_jinan/training_set/X.npy")
X = np.transpose(X, (0, 2, 1))
np.save(f"dataset/8cases_jinan/training_set/X.npy", X)

y = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7))
# Standard Normalization ((X-mean) / std)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i,j,:] = (X[i,j,:] - X[i,j,:].mean()) / X[i,j,:].std()


np.save(f"dataset/8cases_jinan/training_set/X_norm.npy", X)
np.save(f"dataset/8cases_jinan/training_set/y.npy", y)