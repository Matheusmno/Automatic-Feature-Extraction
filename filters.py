import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def apply_butterworth(signal:np.ndarray, fs:float, f_type:str="high", cutoff:int=15, order:int=3) -> np.ndarray:
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype=f_type, analog=False)
    
    return np.array(filtfilt(b, a, signal))

def apply_notch(signal: np.ndarray, fs: float, notch_freq: float=50.0, Q: int=15) -> np.ndarray:
    # Create/view notch filter
    b_notch, a_notch = iirnotch(notch_freq, Q, fs)

    # Apply notch filter to signal
    notched_signal = filtfilt(b_notch, a_notch, signal)
    
    return np.array(notched_signal)