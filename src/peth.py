"""
Peri-event time histogram (PETH) extraction, baseline correction, and bout metrics.
Author: Saad Yusuf
"""

import numpy as np
from scipy.integrate import simpson


def extract_bouts_from_events(df, time):
    """Extract onset timestamps and durations for social and non-social interaction bouts."""
    events_str = df["Events"].fillna("").astype(str)
    soc_start = time[events_str.str.contains("Social_Start", case=False, na=False)]
    soc_end = time[events_str.str.contains("Social_End", case=False, na=False)]
    nonsoc_start = time[events_str.str.contains("Non-Social_Start", case=False, na=False)]
    nonsoc_end = time[events_str.str.contains("Non-Social_End", case=False, na=False)]

    def match_durations(onsets, offsets):
        valid_o, valid_d = [], []
        for o in onsets:
            later = offsets[offsets > o]
            if len(later) > 0:
                dur = later[0] - o
                if 0.2 <= dur <= 60.0:
                    valid_o.append(o)
                    valid_d.append(dur)
            else:
                valid_o.append(o)
                valid_d.append(3.0)
        return np.array(valid_o), np.array(valid_d)

    soc_on, soc_dur = match_durations(soc_start, soc_end)
    obj_on, obj_dur = match_durations(nonsoc_start, nonsoc_end)
    return soc_on, soc_dur, obj_on, obj_dur


def slice_peth(time, signal, onsets, durs, pre_s=3.0, post_s=6.0, fs=30.0):
    """Slice event-aligned traces with pre-stimulus baseline subtraction and whole-bout mean Z."""
    dt = 1.0 / fs
    rel_time = np.arange(-pre_s, post_s + dt / 2, dt)
    if len(onsets) == 0:
        return rel_time, np.empty((0, len(rel_time))), np.array([])

    traces = []
    whole_bout_means = []

    for o, d in zip(onsets, durs):
        t_start, t_end = o - pre_s, o + post_s
        if t_start < time[0] or t_end > time[-1]:
            continue
        sample_t = o + rel_time
        interp = np.interp(sample_t, time, signal)
        base_val = np.mean(interp[rel_time < 0])
        traces.append(interp - base_val)

        # Baseline-corrected mean Z across the exact bout duration [0, d]
        b_mask = (time >= o) & (time <= o + d)
        base_mask = (time >= o - pre_s) & (time < o)
        if np.sum(b_mask) > 0 and np.sum(base_mask) > 0:
            bout_z = np.mean(signal[b_mask]) - np.mean(signal[base_mask])
            whole_bout_means.append(bout_z)
        else:
            whole_bout_means.append(0.0)

    if len(traces) == 0:
        return rel_time, np.empty((0, len(rel_time))), np.array([])
    return rel_time, np.array(traces), np.array(whole_bout_means)


def compute_trial_metrics(rel_time, peth_mat, win_start=0.0, win_end=4.0):
    """Compute trial Peak Z (0-4 s), AUC (0-4 s), and fixed 0-2 s Mean Z window."""
    if len(peth_mat) == 0:
        return np.array([]), np.array([]), np.array([])

    mask_4s = (rel_time >= win_start) & (rel_time <= win_end)
    t_win = rel_time[mask_4s]
    peaks = [np.max(tr[mask_4s]) for tr in peth_mat]
    aucs = [simpson(tr[mask_4s], x=t_win) for tr in peth_mat]

    mask_2s = (rel_time >= 0.0) & (rel_time <= 2.0)
    means_0_2s = [np.mean(tr[mask_2s]) for tr in peth_mat]

    return np.array(peaks), np.array(aucs), np.array(means_0_2s)


def calc_lateralization_index(l_val, r_val):
    """Compute bilateral lateralization index: (Left - Right) / (|Left| + |Right|)."""
    denom = np.abs(l_val) + np.abs(r_val)
    if denom == 0 or np.isnan(denom):
        return 0.0
    return (l_val - r_val) / denom
