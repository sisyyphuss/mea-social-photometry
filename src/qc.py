"""
Optical quality control, isosbestic control PETH, and velocity-artifact checks.
"""

import numpy as np
from scipy.stats import linregress, pearsonr
from .preprocessing import butter_lowpass_filter


def compute_optical_qc(
    time,
    raw_470_l,
    raw_410_l,
    raw_470_r,
    raw_410_r,
    dff_l,
    dff_r,
    onsets,
    fs=30.0,
    pre_s=3.0,
    post_s=6.0,
):
    """Compute photobleaching, SNR, OLS 410/470 R^2, and 410 nm control PETH."""
    m_first = time < 60.0
    m_last = time > (time[-1] - 60.0)
    b1 = (np.mean(raw_410_l[m_last]) - np.mean(raw_410_l[m_first])) / (np.mean(raw_410_l[m_first]) + 1e-9) * 100.0
    b2 = (np.mean(raw_410_r[m_last]) - np.mean(raw_410_r[m_first])) / (np.mean(raw_410_r[m_first]) + 1e-9) * 100.0
    bleach_pct = float((b1 + b2) / 2.0)

    noise_l = np.median(np.abs(np.diff(dff_l))) * 1.4826 / np.sqrt(2.0) + 1e-9
    noise_r = np.median(np.abs(np.diff(dff_r))) * 1.4826 / np.sqrt(2.0) + 1e-9
    snr = float((np.percentile(dff_l, 95) / noise_l + np.percentile(dff_r, 95) / noise_r) / 2.0)

    f470_l = butter_lowpass_filter(raw_470_l, 3.0, fs)
    f410_l = butter_lowpass_filter(raw_410_l, 3.0, fs)
    f470_r = butter_lowpass_filter(raw_470_r, 3.0, fs)
    f410_r = butter_lowpass_filter(raw_410_r, 3.0, fs)

    r1 = linregress(f410_l, f470_l).rvalue
    r2 = linregress(f410_r, f470_r).rvalue
    fit_r2 = float((r1 ** 2 + r2 ** 2) / 2.0)

    z410_l = (f410_l - np.mean(f410_l)) / (np.std(f410_l) + 1e-9)
    z410_r = (f410_r - np.mean(f410_r)) / (np.std(f410_r) + 1e-9)

    dt = 1.0 / 30.0
    rel_time = np.arange(-pre_s, post_s + dt / 2.0, dt)
    iso_traces = []
    for o in onsets:
        if o - pre_s < time[0] or o + post_s > time[-1]:
            continue
        i1 = np.interp(o + rel_time, time, z410_l)
        i2 = np.interp(o + rel_time, time, z410_r)
        i1 -= np.mean(i1[rel_time < 0])
        i2 -= np.mean(i2[rel_time < 0])
        iso_traces.append((i1 + i2) / 2.0)

    iso_mat = np.array(iso_traces) if len(iso_traces) > 0 else np.empty((0, len(rel_time)))
    m02 = (rel_time >= 0.0) & (rel_time <= 2.0)
    iso_0_2s = float(np.mean(iso_mat[:, m02])) if len(iso_mat) > 0 else np.nan
    iso_mean_trace = np.mean(iso_mat, axis=0) if len(iso_mat) > 0 else np.zeros(len(rel_time))

    return {
        "bleach_410_pct": round(bleach_pct, 3),
        "snr": round(snr, 3),
        "fit_r2": round(fit_r2, 4),
        "iso_0_2s_mean_z": round(iso_0_2s, 4),
        "rel_time": rel_time,
        "iso_mean_trace": iso_mean_trace,
        "f470_l": f470_l,
        "f470_r": f470_r,
    }


def compute_speed_coupling(df_tr, time_phot, f470_l, f470_r):
    """Compare speed correlation before (raw 470 nm) vs after (410 nm corrected Z) regression."""
    t_tr = df_tr["video_time_sec"].values
    spd = df_tr["speed_px_sec"].values
    z1_tr = df_tr["CH1_z"].values
    z2_tr = df_tr["CH2_z"].values

    raw1_interp = np.interp(t_tr, time_phot, f470_l)
    raw2_interp = np.interp(t_tr, time_phot, f470_r)

    valid = ~(np.isnan(spd) | np.isnan(z1_tr) | np.isnan(z2_tr))
    r_raw_l, _ = pearsonr(spd[valid], raw1_interp[valid])
    r_raw_r, _ = pearsonr(spd[valid], raw2_interp[valid])
    r_cor_l, _ = pearsonr(spd[valid], z1_tr[valid])
    r_cor_r, _ = pearsonr(spd[valid], z2_tr[valid])

    return {
        "speed_r_raw470": round(float((r_raw_l + r_raw_r) / 2.0), 4),
        "speed_r_corrected": round(float((r_cor_l + r_cor_r) / 2.0), 4),
    }
