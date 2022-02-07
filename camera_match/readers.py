import numpy as np

def read_csv(path):
    return np.fromstring(path, sep=" ")
