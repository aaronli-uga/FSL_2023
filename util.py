'''
Author: Qi7
Date: 2023-05-17 16:24:45
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-05-24 15:37:36
Description: utils function for sliding window
'''
import numpy as np

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