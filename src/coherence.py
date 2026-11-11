"""
Inter-hemispheric spectral coherence analysis between Left and Right MeA.
"""

import numpy as np
from scipy.signal import coherence


def compute_spectral_coherence_bouts(time, z_left, z_right, onsets, win_len=5.0, fs=30.0):
    """Compute magnitude-squared coherence between Left and Right MeA during baseline vs interaction."""
    n_win = int(round(win_len * fs))
    nperseg = int(round(fs * 2.5))  # 75 points for exact 0.4 Hz frequency resolution
    noverlap = int(np.ceil(nperseg / 2))  # 38 points -> 3 symmetric Welch segments across 150 samples

    pre_cohs, post_cohs = [], []
    valid_bouts = []

    for idx, o in enumerate(onsets):
        t_pre_start = o - win_len
        t_post_end = o + win_len

        if t_pre_start < time[0] or t_post_end > time[-1]:
            continue

        # Resample onto a uniform fs grid so FFT bins match exactly across mice
        t_pre = np.linspace(t_pre_start, o, n_win, endpoint=False)
        t_post = np.linspace(o, t_post_end, n_win, endpoint=False)

        sig_l_pre = np.interp(t_pre, time, z_left)
        sig_r_pre = np.interp(t_pre, time, z_right)
        sig_l_post = np.interp(t_post, time, z_left)
        sig_r_post = np.interp(t_post, time, z_right)

        freqs, c_pre = coherence(sig_l_pre, sig_r_pre, fs=fs, nperseg=nperseg, noverlap=noverlap)
        _, c_post = coherence(sig_l_post, sig_r_post, fs=fs, nperseg=nperseg, noverlap=noverlap)

        pre_cohs.append(c_pre)
        post_cohs.append(c_post)
        valid_bouts.append(idx)

    if len(pre_cohs) == 0:
        return np.array([]), np.array([]), np.array([]), valid_bouts

    return np.array(freqs), np.array(pre_cohs), np.array(post_cohs), valid_bouts


def band_average_coherence(freqs, coh_list, band=(0.1, 3.0)):
    """Compute mean coherence within a specified frequency band."""
    if len(coh_list) == 0 or len(freqs) == 0:
        return []
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return [np.mean(c[mask]) for c in coh_list]
