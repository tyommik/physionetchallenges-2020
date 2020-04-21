import numpy as np


def get_ecg_features(data):
    lenght = 5000
    waveform = data[0]
    if len(waveform) < lenght:
        remainder = lenght - len(waveform)
        waveform = np.pad(waveform, (0, remainder))
    else:
        waveform = waveform[:5000]
    return waveform.reshape(-1, 1)