import numpy as np


def get_ecg_features(data):
    data = data[0]
    data = data[:5000]
    return data.reshape(-1, 1)