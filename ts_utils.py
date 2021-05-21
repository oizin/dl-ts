import numpy as np
import pandas as pd

def create_sliding_window(data, sequence_length, stride=1, output_size=1):
    X_list, y_list = [], []
    for i in range(len(data)):
        if(i + sequence_length + output_size) < len(data):
            end = i+sequence_length
            X_list.append(data.iloc[i:end:stride, :].values)
            if output_size > 1:
                y_list.append(data.iloc[end:(end+output_size), -1].values)
            else:
                y_list.append(data.iloc[end, -1])
    return np.array(X_list), np.array(y_list)