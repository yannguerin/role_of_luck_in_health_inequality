import numpy as np

def min_max(data: np.ndarray):
    return (data - data.min()) / (data.max() - data.min())

def linear_scale(data: np.ndarray, high: float, low: float):
    return (((data - data.min()) / (data.max() - data.min())) * (high - low)) + low