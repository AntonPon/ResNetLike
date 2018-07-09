import numpy as np
import pandas as pd


def data_prep(path_csv):
    df = pd.read_csv(path_csv, header=None)
    y_real = np.zeros((2, df.shape[0]))
    for i in range(df.shape[0]):
        idx = 0
        if df[1].values[i] == 'M':
            idx = 1
        y_real[idx, i] = 1
    return (y_real, df[np.arange(2, df.shape[-1], 1)].values)