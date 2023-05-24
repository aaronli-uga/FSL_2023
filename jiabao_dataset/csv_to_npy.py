'''
Author: Qi7
Date: 2022-12-13 22:48:05
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-24 11:20:32
Description: This python script convert all the window csv data to one npy file for training purpose. (Jiabao dataset)
'''
import os
import numpy as np


detect_path = 'original_dataset/w100_final_dataset/fault_diagnosis/deep_learning/'
diagnosis_path = 'original_dataset/w100_final_dataset/fault_diagnosis/deep_learning/'

# normal_path = detect_path + 'Normal/'
# abnormal_path = detect_path + 'Abnormal/'

# normal_files = [f for f in os.listdir(normal_path) if f.endswith('.csv')]
# abnormal_files = [f for f in os.listdir(abnormal_path) if f.endswith('.csv')]

# np_normal = []
# np_abnormal = []

# for f in normal_files:
#     temp_file = np.loadtxt(normal_path + f, delimiter=',', dtype=np.float32)
#     np_normal.append(temp_file)

# for f in abnormal_files:
#     temp_file = np.loadtxt(abnormal_path + f, delimiter=',', dtype=np.float32)
#     np_abnormal.append(temp_file)
    
# # x: input data, y: target
# y_data = np.concatenate([np.zeros(len(np_normal),dtype=np.long), \
#                                 np.zeros(len(np_abnormal),dtype=np.long)+1])
# x_data = np.concatenate([np_normal, np_abnormal])
# x_data = x_data.reshape(x_data.shape[0], 6, -1)
# length = x_data.shape[0]
# # print(length)
# # x_data = torch.from_numpy(self.x_data)
# # y_data = torch.from_numpy(self.y_data)
# with open('w100_detection_data.npy', 'wb') as f:
#     np.save(f, x_data)
    
# with open('w100_detection_label.npy', 'wb') as f:
#     np.save(f, y_data)
    

############################################### diagnosis
normal_path = diagnosis_path + 'Normal/'
fault_1_path = diagnosis_path + 'Fault_1/'
fault_2_path = diagnosis_path + 'Fault_2/'
fault_3_path = diagnosis_path + 'Fault_3/'
fault_4_path = diagnosis_path + 'Fault_4/'
fault_5_path = diagnosis_path + 'Fault_5/'
fault_6_path = diagnosis_path + 'Fault_6/'
fault_7_path = diagnosis_path + 'Fault_7/'
fault_8_path = diagnosis_path + 'Fault_8/'


normal_files = [f for f in os.listdir(normal_path) if f.endswith('.csv')]
fault_1_files = [f for f in os.listdir(fault_1_path) if f.endswith('.csv')]
fault_2_files = [f for f in os.listdir(fault_2_path) if f.endswith('.csv')]
fault_3_files = [f for f in os.listdir(fault_3_path) if f.endswith('.csv')]
fault_4_files = [f for f in os.listdir(fault_4_path) if f.endswith('.csv')]
fault_5_files = [f for f in os.listdir(fault_5_path) if f.endswith('.csv')]
fault_6_files = [f for f in os.listdir(fault_6_path) if f.endswith('.csv')]
fault_7_files = [f for f in os.listdir(fault_7_path) if f.endswith('.csv')]
fault_8_files = [f for f in os.listdir(fault_8_path) if f.endswith('.csv')]

np_normal = []
np_fault_1 = []
np_fault_2 = []
np_fault_3 = []
np_fault_4 = []
np_fault_5 = []
np_fault_6 = []
np_fault_7 = []
np_fault_8 = []


for f in normal_files:
    temp_file = np.loadtxt(normal_path + f, delimiter=',', dtype=np.float32)
    np_normal.append(temp_file)

for f in fault_1_files:
    temp_file = np.loadtxt(fault_1_path + f, delimiter=',', dtype=np.float32)
    np_fault_1.append(temp_file)
    
for f in fault_2_files:
    temp_file = np.loadtxt(fault_2_path + f, delimiter=',', dtype=np.float32)
    np_fault_2.append(temp_file)
    
for f in fault_3_files:
    temp_file = np.loadtxt(fault_3_path + f, delimiter=',', dtype=np.float32)
    np_fault_3.append(temp_file)
    
for f in fault_4_files:
    temp_file = np.loadtxt(fault_4_path + f, delimiter=',', dtype=np.float32)
    np_fault_4.append(temp_file)
    
for f in fault_5_files:
    temp_file = np.loadtxt(fault_5_path + f, delimiter=',', dtype=np.float32)
    np_fault_5.append(temp_file)
    
for f in fault_6_files:
    temp_file = np.loadtxt(fault_6_path + f, delimiter=',', dtype=np.float32)
    np_fault_6.append(temp_file)
    
for f in fault_7_files:
    temp_file = np.loadtxt(fault_7_path + f, delimiter=',', dtype=np.float32)
    np_fault_7.append(temp_file)
    
for f in fault_8_files:
    temp_file = np.loadtxt(fault_8_path + f, delimiter=',', dtype=np.float32)
    np_fault_8.append(temp_file)
    
# x: input data, y: target
y_data = np.concatenate([np.zeros(len(np_normal),dtype=np.long), \
                                      np.zeros(len(np_fault_1),dtype=np.long)+1, \
                                      np.zeros(len(np_fault_2),dtype=np.long)+2, \
                                      np.zeros(len(np_fault_3),dtype=np.long)+3, \
                                      np.zeros(len(np_fault_4),dtype=np.long)+4, \
                                      np.zeros(len(np_fault_5),dtype=np.long)+5, \
                                      np.zeros(len(np_fault_6),dtype=np.long)+6, \
                                      np.zeros(len(np_fault_7),dtype=np.long)+7, \
                                      np.zeros(len(np_fault_8),dtype=np.long)+8, ])
x_data = np.concatenate([np_normal, np_fault_1, np_fault_2, np_fault_3, np_fault_4, 
                                      np_fault_5, np_fault_6, np_fault_7, np_fault_8])
x_data = x_data.reshape(x_data.shape[0], 6, -1)

with open('w100_diagnosis_data.npy', 'wb') as f:
    np.save(f, x_data)
    
with open('w100_diagnosis_label.npy', 'wb') as f:
    np.save(f, y_data)