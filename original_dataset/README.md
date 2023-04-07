<!--
 * @Author: Qi7
 * @Date: 2023-04-06 15:00:15
 * @LastEditors: aaronli-uga ql61608@uga.edu
 * @LastEditTime: 2023-04-06 15:10:07
 * @Description: 
-->
**window_extraction.m** will parse the orginal dataset with the parameters (window size) into 2 folders.

- fault_detection (2 classes: normal and fault)
- fault_diagnosis (9 classes: normal and 8 faults)

Each folder will contain two subfolder, **feature** (for machine learning algorithms) and **deep_learning** (for training the deep learning models)