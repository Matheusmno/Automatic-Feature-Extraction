import numpy as np
from scipy.signal import butter, filtfilt


def apply_butterworth(signal:np.ndarray, fs:float, f_type:str="high", cutoff:int=15, order:int=3) -> np.ndarray:
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype=f_type, analog=False)
    
    return np.array(filtfilt(b, a, signal))

def apply_notch_filter(signal:np.ndarray, notch_freq: float, fs: float) -> np.ndarray:
    nyquist = 0.5 * fs
    notch_normalized = notch_freq / nyquist
    b, a = butter(2, [notch_normalized - 0.02, notch_normalized + 0.02], btype='bandstop', analog=False)
    
    return np.array(filtfilt(b, a, signal))