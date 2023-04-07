'''
Author: Qi7
Date: 2023-04-06 21:23:30
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-06 23:23:24
Description: 
'''
#%%
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, train_test_split

X = np.load('dataset/w100_detection_data.npy')
y = np.load('dataset/w100_detection_label.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=7)



# %%
