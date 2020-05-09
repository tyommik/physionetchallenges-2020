import numpy as np

from ecg import ECG

LENGTH = 5000
LEADS_NUM = 12


def _set_duration(waveform):
    """Set duration of ecg waveform."""
    if len(waveform) > LENGTH:
        return waveform[0:LENGTH]
    else:
        return waveform

def _zero_pad(waveform, align):
    """Zero pad waveform (align: left, center, right)."""
    # Get remainder
    remainder = LENGTH - len(waveform)
    if align == 'left':
        for lead in range(LEADS_NUM):
            np.pad(waveform[:, lead], (0, remainder), 'constant', constant_values=0)
        return waveform
    elif align == 'center':
        for lead in range(LEADS_NUM):
            np.pad(waveform[:, lead], (int(remainder / 2), remainder - int(remainder / 2)), 'constant', constant_values=0)
        return waveform
    elif align == 'right':
        for lead in range(LEADS_NUM):
            return np.pad(waveform[:, lead], (remainder, 0), 'constant', constant_values=0)
        return waveform


def get_ecg_features(data):
    ecg = ECG(file_name=None, label=None,
              waveform=data.transpose(), filter_bands=[3, 45], fs=500)

    # Set waveform duration
    waveform = _set_duration(waveform=ecg.filtered)
    waveform_pad = _zero_pad(waveform=waveform, align='center')

    return waveform_pad