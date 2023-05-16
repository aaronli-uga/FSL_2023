'''
Author: Qi7
Date: 2023-05-16 16:42:34
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-16 16:56:35
Description: 
'''
import numpy as np
from collections import Counter

labels = np.load('dataset/w100_diagnosis_label.npy')

counter = Counter(labels)

# index of the label equals to '8'
labels[np.where(labels == 8)[0]]