"""
Photometry, Tracking, and Spectral Coherence Analysis for Mouse 311 (Young)

Usage:
    python analyze_Mouse_311.py
"""

import os
import sys
import io
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, coherence
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter
from scipy.integrate import simpson

MOUSE_ID = "311"
AGE_GROUP = "Young"
COLOR_LEFT = "#1b9e77"
COLOR_RIGHT = "#7570b3"

def butter_lowpass_filter(signal, cutoff=3.0, fs=30.0, order=2):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='low', analog=False)
    return filtfilt(b, a, signal)

def motion_correct_and_zscore(f_signal, f_reference):
    slope, intercept, _, _, _ = linregress(f_reference, f_signal)
    f_fitted = slope * f_reference + intercept
    dff = (f_signal - f_fitted) / f_fitted * 100.0
    std_val = np.std(dff)
    z_score = (dff - np.mean(dff)) / (std_val if std_val > 1e-9 else 1.0)
    return dff, z_score, f_fitted

def extract_bouts_from_events(df, time):
    events_str = df['Events'].fillna('').astype(str)
    soc_start = time[events_str.str.contains('Social_Start', case=False, na=False)]
    soc_end   = time[events_str.str.contains('Social_End', case=False, na=False)]
    nonsoc_start = time[events_str.str.contains('Non-Social_Start', case=False, na=False)]
    nonsoc_end   = time[events_str.str.contains('Non-Social_End', case=False, na=False)]

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
    dt = 1.0 / fs
    rel_time = np.arange(-pre_s, post_s + dt/2, dt)
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
    denom = np.abs(l_val) + np.abs(r_val)
    if denom == 0 or np.isnan(denom):
        return 0.0
    return (l_val - r_val) / denom

def compute_spectral_coherence_bouts(time, z_left, z_right, onsets, win_len=5.0, fs=30.0):
    nperseg = int(fs * 2.5)
    noverlap = nperseg // 2
    pre_cohs, post_cohs = [], []
    valid_bouts = []
    for idx, o in enumerate(onsets):
        t_pre_start = o - win_len
        t_post_end   = o + win_len
        if t_pre_start < time[0] or t_post_end > time[-1]:
            continue
        mask_pre  = (time >= t_pre_start) & (time < o)
        mask_post = (time >= o) & (time < t_post_end)
        sig_l_pre, sig_r_pre = z_left[mask_pre], z_right[mask_pre]
        sig_l_post, sig_r_post = z_left[mask_post], z_right[mask_post]
        if len(sig_l_pre) < nperseg or len(sig_l_post) < nperseg:
            continue
        freqs, c_pre = coherence(sig_l_pre, sig_r_pre, fs=fs, nperseg=nperseg, noverlap=noverlap)
        _, c_post    = coherence(sig_l_post, sig_r_post, fs=fs, nperseg=nperseg, noverlap=noverlap)
        pre_cohs.append(c_pre)
        post_cohs.append(c_post)
        valid_bouts.append(idx)
    if len(pre_cohs) == 0:
        return np.array([]), np.array([]), np.array([]), valid_bouts
    return np.array(freqs), np.array(pre_cohs), np.array(post_cohs), valid_bouts

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.expanduser("~/Downloads/Data_2.zip")
    tracking_csv = os.path.expanduser(f"~/Downloads/results_social/results_social/individual/Social/Young/311/synchronized_tracking_photometry.csv")
    
    print(f"Executing standalone analysis for Mouse 311 (Young)...")
    
    # 1. Load Photometry Raw CSV
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_in_zip = f"Data_2/Social/young/Fluorescence_311.csv"
        with z.open(csv_in_zip) as f:
            df_raw = pd.read_csv(io.StringIO(f.read().decode('utf-8', errors='ignore')), skiprows=1)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    
    raw_t = pd.to_numeric(df_raw['TimeStamp'], errors='coerce').values
    time = (raw_t - raw_t[0]) / 1000.0 if np.nanmedian(np.diff(raw_t)) > 2.0 else raw_t - raw_t[0]
    actual_fs = 1.0 / np.median(np.diff(time))
    total_dur_min = (time[-1] - time[0]) / 60.0
    
    # 2. Filter & Motion Correct
    ch1_470 = butter_lowpass_filter(pd.to_numeric(df_raw['CH1-470'], errors='coerce').interpolate().values, 3.0, actual_fs)
    ch1_410 = butter_lowpass_filter(pd.to_numeric(df_raw['CH1-410'], errors='coerce').interpolate().values, 3.0, actual_fs)
    ch2_470 = butter_lowpass_filter(pd.to_numeric(df_raw['CH2-470'], errors='coerce').interpolate().values, 3.0, actual_fs)
    ch2_410 = butter_lowpass_filter(pd.to_numeric(df_raw['CH2-410'], errors='coerce').interpolate().values, 3.0, actual_fs)
    
    ch1_dff, ch1_z, ch1_fit = motion_correct_and_zscore(ch1_470, ch1_410)
    ch2_dff, ch2_z, ch2_fit = motion_correct_and_zscore(ch2_470, ch2_410)
    
    # 3. Events & Metrics
    soc_on, soc_dur, obj_on, obj_dur = extract_bouts_from_events(df_raw, time)
    fs_nominal = 30.0
    rel_time, peth_soc_l, wb_soc_l = slice_peth(time, ch1_z, soc_on, soc_dur, 3.0, 6.0, fs_nominal)
    _,        peth_soc_r, wb_soc_r = slice_peth(time, ch2_z, soc_on, soc_dur, 3.0, 6.0, fs_nominal)
    _,        peth_obj_l, wb_obj_l = slice_peth(time, ch1_z, obj_on, obj_dur, 3.0, 6.0, fs_nominal)
    _,        peth_obj_r, wb_obj_r = slice_peth(time, ch2_z, obj_on, obj_dur, 3.0, 6.0, fs_nominal)
    
    soc_pks_l, soc_aucs_l, soc_m02_l = compute_trial_metrics(rel_time, peth_soc_l)
    soc_pks_r, soc_aucs_r, soc_m02_r = compute_trial_metrics(rel_time, peth_soc_r)
    obj_pks_l, obj_aucs_l, obj_m02_l = compute_trial_metrics(rel_time, peth_obj_l)
    obj_pks_r, obj_aucs_r, obj_m02_r = compute_trial_metrics(rel_time, peth_obj_r)
    
    # 4. Multi-Band Coherence (PMC10592781)
    freqs_soc, coh_soc_pre, coh_soc_post, v_soc_idx = compute_spectral_coherence_bouts(time, ch1_z, ch2_z, soc_on, 5.0, actual_fs)
    freqs_obj, coh_obj_pre, coh_obj_post, v_obj_idx = compute_spectral_coherence_bouts(time, ch1_z, ch2_z, obj_on, 5.0, actual_fs)
    
    m_full = (freqs_soc >= 0.1) & (freqs_soc <= 3.0) if len(freqs_soc) > 0 else []
    m_b1   = (freqs_soc >= 0.1) & (freqs_soc <= 0.5) if len(freqs_soc) > 0 else []
    m_b2   = (freqs_soc >= 0.5) & (freqs_soc <= 1.5) if len(freqs_soc) > 0 else []
    m_b3   = (freqs_soc >= 1.5) & (freqs_soc <= 3.0) if len(freqs_soc) > 0 else []
    m_std  = (freqs_soc >= 0.8) & (freqs_soc <= 1.6) if len(freqs_soc) > 0 else []
    
    soc_c_pre_full = [np.mean(c[m_full]) for c in coh_soc_pre] if len(coh_soc_pre) > 0 else []
    soc_c_post_full = [np.mean(c[m_full]) for c in coh_soc_post] if len(coh_soc_post) > 0 else []
    soc_c_b1_del   = [np.mean(post[m_b1]) - np.mean(pre[m_b1]) for pre, post in zip(coh_soc_pre, coh_soc_post)] if len(coh_soc_pre) > 0 else []
    soc_c_b2_del   = [np.mean(post[m_b2]) - np.mean(pre[m_b2]) for pre, post in zip(coh_soc_pre, coh_soc_post)] if len(coh_soc_pre) > 0 else []
    soc_c_b3_del   = [np.mean(post[m_b3]) - np.mean(pre[m_b3]) for pre, post in zip(coh_soc_pre, coh_soc_post)] if len(coh_soc_pre) > 0 else []
    soc_c_std_pre  = [np.mean(c[m_std]) for c in coh_soc_pre] if len(coh_soc_pre) > 0 else []
    soc_c_std_post = [np.mean(c[m_std]) for c in coh_soc_post] if len(coh_soc_post) > 0 else []
    
    # 5. Build Trial Metrics Table
    mouse_trials = []
    for k in range(len(soc_aucs_l)):
        li = calc_lateralization_index(soc_aucs_l[k], soc_aucs_r[k])
        c_full_pre  = round(float(soc_c_pre_full[k]), 3) if k < len(soc_c_pre_full) else np.nan
        c_full_post = round(float(soc_c_post_full[k]), 3) if k < len(soc_c_post_full) else np.nan
        c_full_del  = round(c_full_post - c_full_pre, 3) if (not np.isnan(c_full_pre) and not np.isnan(c_full_post)) else np.nan
        c_b1_del    = round(float(soc_c_b1_del[k]), 3) if k < len(soc_c_b1_del) else np.nan
        c_b2_del    = round(float(soc_c_b2_del[k]), 3) if k < len(soc_c_b2_del) else np.nan
        c_b3_del    = round(float(soc_c_b3_del[k]), 3) if k < len(soc_c_b3_del) else np.nan
        c_pre  = round(float(soc_c_std_pre[k]), 3) if k < len(soc_c_std_pre) else np.nan
        c_post = round(float(soc_c_std_post[k]), 3) if k < len(soc_c_std_post) else np.nan
        c_del  = round(c_post - c_pre, 3) if (not np.isnan(c_pre) and not np.isnan(c_post)) else np.nan
        
        m02_bilat = round(float((soc_m02_l[k] + soc_m02_r[k]) / 2.0), 3)
        wb_bilat  = round(float((wb_soc_l[k] + wb_soc_r[k]) / 2.0), 3)
        
        rec = {
            'Mouse_ID': MOUSE_ID, 'Age_Group': AGE_GROUP, 'Stimulus': 'Social',
            'Bout': k+1, 'Duration_s': round(soc_dur[k], 2) if k < len(soc_dur) else np.nan,
            'Left_MeA_AUC': round(soc_aucs_l[k], 3), 'Right_MeA_AUC': round(soc_aucs_r[k], 3),
            'Left_MeA_Peak_Z': round(soc_pks_l[k], 3), 'Right_MeA_Peak_Z': round(soc_pks_r[k], 3),
            'Left_MeA_Whole_Bout_Mean_Z': round(wb_soc_l[k], 3), 'Right_MeA_Whole_Bout_Mean_Z': round(wb_soc_r[k], 3),
            'Whole_Bout_Mean_Z_Bilateral': wb_bilat,
            'Left_MeA_Mean_Z_0_2s': round(soc_m02_l[k], 3), 'Right_MeA_Mean_Z_0_2s': round(soc_m02_r[k], 3),
            'Mean_Z_0_2s_Bilateral': m02_bilat,
            'Lateralization_Index': round(li, 3),
            'Coherence_Full_0_1_3_0Hz_Base': c_full_pre, 'Coherence_Full_0_1_3_0Hz_Social': c_full_post, 'Coherence_Full_0_1_3_0Hz_Delta': c_full_del,
            'Coherence_0_1_0_5Hz_Delta': c_b1_del, 'Coherence_0_5_1_5Hz_Delta': c_b2_del, 'Coherence_1_5_3_0Hz_Delta': c_b3_del,
            'Baseline_Coherence': c_pre, 'Interaction_Coherence': c_post, 'Coherence_Delta': c_del
        }
        mouse_trials.append(rec)
        
    for k in range(len(obj_aucs_l)):
        li = calc_lateralization_index(obj_aucs_l[k], obj_aucs_r[k])
        m02_bilat = round(float((obj_m02_l[k] + obj_m02_r[k]) / 2.0), 3)
        wb_bilat  = round(float((wb_obj_l[k] + wb_obj_r[k]) / 2.0), 3)
        rec = {
            'Mouse_ID': MOUSE_ID, 'Age_Group': AGE_GROUP, 'Stimulus': 'NonSocial',
            'Bout': k+1, 'Duration_s': round(obj_dur[k], 2) if k < len(obj_dur) else np.nan,
            'Left_MeA_AUC': round(obj_aucs_l[k], 3), 'Right_MeA_AUC': round(obj_aucs_r[k], 3),
            'Left_MeA_Peak_Z': round(obj_pks_l[k], 3), 'Right_MeA_Peak_Z': round(obj_pks_r[k], 3),
            'Left_MeA_Whole_Bout_Mean_Z': round(wb_obj_l[k], 3), 'Right_MeA_Whole_Bout_Mean_Z': round(wb_obj_r[k], 3),
            'Whole_Bout_Mean_Z_Bilateral': wb_bilat,
            'Left_MeA_Mean_Z_0_2s': round(obj_m02_l[k], 3), 'Right_MeA_Mean_Z_0_2s': round(obj_m02_r[k], 3),
            'Mean_Z_0_2s_Bilateral': m02_bilat,
            'Lateralization_Index': round(li, 3),
            'Coherence_Full_0_1_3_0Hz_Base': np.nan, 'Coherence_Full_0_1_3_0Hz_Social': np.nan, 'Coherence_Full_0_1_3_0Hz_Delta': np.nan,
            'Coherence_0_1_0_5Hz_Delta': np.nan, 'Coherence_0_5_1_5Hz_Delta': np.nan, 'Coherence_1_5_3_0Hz_Delta': np.nan,
            'Baseline_Coherence': np.nan, 'Interaction_Coherence': np.nan, 'Coherence_Delta': np.nan
        }
        mouse_trials.append(rec)
        
    pd.DataFrame(mouse_trials).to_csv(os.path.join(out_dir, "trial_metrics.csv"), index=False)
    
    # 6. Read Tracking & Save Summary JSON
    df_tr = pd.read_csv(tracking_csv)
    x_pts, y_pts = df_tr['x_px'].dropna().values, df_tr['y_px'].dropna().values
    speed = df_tr['speed_px_sec'].dropna().values
    dist_px = float(np.sum(np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2)))
    stim = df_tr['compartment_stimulus'].values
    fps_tr = 60.0
    t_soc = float(np.sum(stim == 'Novel Mouse') / fps_tr)
    t_obj = float(np.sum(stim == 'Object') / fps_tr)
    spi_beh = (t_soc - t_obj) / (t_soc + t_obj + 1e-9)
    
    m_summary = {
        'Mouse_ID': str(MOUSE_ID),
        'Age_Group': AGE_GROUP,
        'Recording_Duration_Min': round(total_dur_min, 1),
        'Social_Bout_Count': len(soc_on),
        'Analyzed_Social_Bouts_in_PETH': len(soc_aucs_l),
        'Social_Total_Time_s': round(float(np.sum(soc_dur)), 1),
        'NonSocial_Bout_Count': len(obj_on),
        'Analyzed_NonSocial_Bouts_in_PETH': len(obj_aucs_l),
        'NonSocial_Total_Time_s': round(float(np.sum(obj_dur)), 1),
        'Social_AUC_Left': round(float(np.mean(soc_aucs_l)), 3),
        'Social_AUC_Right': round(float(np.mean(soc_aucs_r)), 3),
        'Social_Peak_Left': round(float(np.mean(soc_pks_l)), 3),
        'Social_Peak_Right': round(float(np.mean(soc_pks_r)), 3),
        'Social_Whole_Bout_Mean_Z_Left': round(float(np.mean(wb_soc_l)), 3),
        'Social_Whole_Bout_Mean_Z_Right': round(float(np.mean(wb_soc_r)), 3),
        'Social_Whole_Bout_Mean_Z_Bilateral': round(float((np.mean(wb_soc_l) + np.mean(wb_soc_r)) / 2.0), 3),
        'Social_Mean_Z_0_2s_Left': round(float(np.mean(soc_m02_l)), 3),
        'Social_Mean_Z_0_2s_Right': round(float(np.mean(soc_m02_r)), 3),
        'Social_Mean_Z_0_2s_Bilateral': round(float((np.mean(soc_m02_l) + np.mean(soc_m02_r)) / 2.0), 3),
        'Social_Lateralization_Index': round(calc_lateralization_index(np.mean(soc_aucs_l), np.mean(soc_aucs_r)), 3),
        'NonSocial_AUC_Left': round(float(np.mean(obj_aucs_l)), 3),
        'NonSocial_AUC_Right': round(float(np.mean(obj_aucs_r)), 3),
        'NonSocial_Peak_Left': round(float(np.mean(obj_pks_l)), 3),
        'NonSocial_Peak_Right': round(float(np.mean(obj_pks_r)), 3),
        'NonSocial_Whole_Bout_Mean_Z_Left': round(float(np.mean(wb_obj_l)), 3),
        'NonSocial_Whole_Bout_Mean_Z_Right': round(float(np.mean(wb_obj_r)), 3),
        'NonSocial_Whole_Bout_Mean_Z_Bilateral': round(float((np.mean(wb_obj_l) + np.mean(wb_obj_r)) / 2.0), 3),
        'NonSocial_Mean_Z_0_2s_Left': round(float(np.mean(obj_m02_l)), 3),
        'NonSocial_Mean_Z_0_2s_Right': round(float(np.mean(obj_m02_r)), 3),
        'NonSocial_Mean_Z_0_2s_Bilateral': round(float((np.mean(obj_m02_l) + np.mean(obj_m02_r)) / 2.0), 3),
        'NonSocial_Lateralization_Index': round(calc_lateralization_index(np.mean(obj_aucs_l), np.mean(obj_aucs_r)), 3),
        'Coherence_Full_Spectrum_0_1_3_0Hz_Base': round(float(np.nanmean(soc_c_pre_full)), 3) if len(soc_c_pre_full) > 0 else np.nan,
        'Coherence_Full_Spectrum_0_1_3_0Hz_Social': round(float(np.nanmean(soc_c_post_full)), 3) if len(soc_c_post_full) > 0 else np.nan,
        'Coherence_Full_Spectrum_0_1_3_0Hz_Delta': round(float(np.nanmean(soc_c_post_full) - np.nanmean(soc_c_pre_full)), 3) if len(soc_c_pre_full) > 0 else np.nan,
        'Coherence_Band1_0_1_0_5Hz_Delta': round(float(np.nanmean(soc_c_b1_del)), 3) if len(soc_c_b1_del) > 0 else np.nan,
        'Coherence_Band2_0_5_1_5Hz_Delta': round(float(np.nanmean(soc_c_b2_del)), 3) if len(soc_c_b2_del) > 0 else np.nan,
        'Coherence_Band3_1_5_3_0Hz_Delta': round(float(np.nanmean(soc_c_b3_del)), 3) if len(soc_c_b3_del) > 0 else np.nan,
        'Mean_Baseline_Coherence_0_8_1_6Hz': round(float(np.nanmean(soc_c_std_pre)), 3) if len(soc_c_std_pre) > 0 else np.nan,
        'Mean_Social_Coherence_0_8_1_6Hz': round(float(np.nanmean(soc_c_std_post)), 3) if len(soc_c_std_post) > 0 else np.nan,
        'Mean_Coherence_Delta': round(float(np.nanmean(soc_c_std_post) - np.nanmean(soc_c_std_pre)), 3) if len(soc_c_std_pre) > 0 else np.nan,
        'Locomotor_Distance_px': round(dist_px, 1),
        'Mean_Speed_px_s': round(float(np.mean(speed)), 2),
        'Chamber_Time_Social_s': round(t_soc, 1),
        'Chamber_Time_Object_s': round(t_obj, 1),
        'Behavioral_Social_Preference_Index': round(float(spi_beh), 3),
        'Notes': 'Dual MeA photometry. References: GuPPy (filtering/motion-correction), ezTrack (tracking/SPI), Roland et al. 2024 / PMC10592781 (bilateral spectral coherence).'
    }
    with open(os.path.join(out_dir, "mouse_summary.json"), "w") as jf:
        json.dump(m_summary, jf, indent=2)
        
    print(f"Analysis complete for Mouse 311. Output files written to {out_dir}.")

if __name__ == "__main__":
    main()
