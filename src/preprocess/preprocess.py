import numpy as np


def center(data: np.ndarray, axis: int = 0) -> np.ndarray:
    '''
    centers data by subtracting the mean
    '''
    means = np.mean(data, axis=axis)
    return data - means


def minmax_scale(data: np.ndarray, axis: int = 0) -> np.ndarray:
    '''
    Scales data by dividing by the range
    '''
    naive_rngs = np.max(data, axis=axis) - np.min(data, axis=axis) # may include 0
    rngs = np.where(naive_rngs == 0, 1, naive_rngs)
    return data/rngs
