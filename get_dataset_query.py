'''
Author: Qi7
Date: 2023-06-01 17:15:30
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-04 15:13:52
Description: 
'''
import pandas as pd
import numpy as np
from util import sliding_windows
from sklearn.model_selection import train_test_split

# files = ["attack27", "attack39", "attack50", "attack51", "attack55", "attack66"]

# for file in files:
#     data = f"dataset/8cases_jinan/new_query_set/raw_data/{file}.csv"
#     df = pd.read_csv(data)
#     data_npy = df.to_numpy()
#     o1, _ = sliding_windows(data_npy, sub_window_size=2000, step_size=20)
#     np.save(f"dataset/8cases_jinan/new_query_set/X_{file}.npy", o1)

X_8 = np.load("dataset/8cases_jinan/new_query_set/X_attack27.npy")
X_9 = np.load("dataset/8cases_jinan/new_query_set/X_attack39.npy")
X_10 = np.load("dataset/8cases_jinan/new_query_set/X_attack50.npy")
X_11 = np.load("dataset/8cases_jinan/new_query_set/X_attack51.npy")
X_12 = np.load("dataset/8cases_jinan/new_query_set/X_attack66.npy")

y_8 = np.ones(X_8.shape[0]) * 8
y_9 = np.ones(X_9.shape[0]) * 9
y_10 = np.ones(X_10.shape[0]) * 10
y_11 = np.ones(X_11.shape[0]) * 11
y_12 = np.ones(X_12.shape[0]) * 12

X = np.concatenate((X_8, X_9, X_10, X_11, X_12), axis=0)

# X = np.load("dataset/8cases_jinan/training_set/X.npy")
X = np.transpose(X, (0, 2, 1))
np.save(f"dataset/8cases_jinan/new_query_set/X.npy", X)

y = np.concatenate((y_8, y_9, y_10, y_11, y_12))
# Standard Normalization ((X-mean) / std)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i,j,:] = (X[i,j,:] - X[i,j,:].mean()) / X[i,j,:].std()


np.save(f"dataset/8cases_jinan/new_query_set/X_norm.npy", X)
np.save(f"dataset/8cases_jinan/new_query_set/y.npy", y)

#%% save the subset for embedding plot.
X = np.load('dataset/8cases_jinan/new_query_set/X_norm.npy')
y = np.load('dataset/8cases_jinan/new_query_set/y.npy')

X_train, X_cv, y_train, y_cv = train_test_split(X, y, train_size=0.85, test_size=0.15, shuffle=True, random_state=27)
np.save(f"dataset/8cases_jinan/new_query_set/X_embedding_plot.npy", X_cv)
np.save(f"dataset/8cases_jinan/new_query_set/y_embedding_plot.npy", y_cv)
# %%
