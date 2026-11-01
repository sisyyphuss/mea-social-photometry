"""
Preprocessing utilities for dual-channel fiber photometry.
Author: Saad Yusuf
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import linregress


def butter_lowpass_filter(signal, cutoff=3.0, fs=30.0, order=2):
    """Apply a zero-phase low-pass Butterworth filter to a time-series signal."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def motion_correct_and_zscore(f_signal, f_reference):
    """Fit 410 nm reference to 470 nm activity via linear regression and compute dF/F and Z-score."""
    slope, intercept, _, _, _ = linregress(f_reference, f_signal)
    f_fitted = slope * f_reference + intercept
    dff = (f_signal - f_fitted) / f_fitted * 100.0
    std_val = np.std(dff)
    z_score = (dff - np.mean(dff)) / (std_val if std_val > 1e-9 else 1.0)
    return dff, z_score, f_fitted
