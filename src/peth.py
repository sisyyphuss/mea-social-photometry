"""
Peri-event time histogram (PETH) extraction, baseline correction, and bout metrics.
"""

import numpy as np
from scipy.integrate import simpson

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


def extract_bouts_from_events(df, time):
    """Extract onset timestamps and durations for social and non-social interaction bouts."""
    events_str = df["Events"].fillna("").astype(str)
    is_nonsoc_s = events_str.str.contains("Non-Social_Start", case=False, na=False)
    is_nonsoc_e = events_str.str.contains("Non-Social_End", case=False, na=False)
    is_soc_s = events_str.str.contains("Social_Start", case=False, na=False) & (~is_nonsoc_s)
    is_soc_e = events_str.str.contains("Social_End", case=False, na=False) & (~is_nonsoc_e)

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

    soc_on, soc_dur = match_durations(time[is_soc_s], time[is_soc_e])
    obj_on, obj_dur = match_durations(time[is_nonsoc_s], time[is_nonsoc_e])
    return soc_on, soc_dur, obj_on, obj_dur


def slice_peth(time, signal, onsets, durs, pre_s=3.0, post_s=6.0, fs=30.0):
    """Slice event-aligned traces with pre-stimulus baseline subtraction and whole-bout mean Z."""
    n_grid = int(round((pre_s + post_s) * fs)) + 1
    rel_time = np.linspace(-pre_s, post_s, n_grid)
    if len(onsets) == 0:
        return rel_time, np.empty((0, n_grid)), np.array([]), []

    traces = []
    whole_bout_means = []
    valid_indices = []

    for orig_idx, (o, d) in enumerate(zip(onsets, durs)):
        t_start, t_end = o - pre_s, o + post_s
        if t_start < time[0] or t_end > time[-1]:
            continue
        interp = np.interp(o + rel_time, time, signal)
        base_val = np.mean(interp[rel_time < 0])
        traces.append(interp - base_val)

        # Duration-normalized mean Z across exact bout span [0, d] on 10 ms grid
        n_bout_pts = max(11, int(round(d * 100.0)) + 1)
        w_grid = np.linspace(o, o + d, n_bout_pts)
        bout_z = _trapezoid(np.interp(w_grid, time, signal) - base_val, w_grid) / d
        whole_bout_means.append(bout_z)
        valid_indices.append(orig_idx)

    if len(traces) == 0:
        return rel_time, np.empty((0, n_grid)), np.array([]), []
    return rel_time, np.array(traces), np.array(whole_bout_means), valid_indices


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
