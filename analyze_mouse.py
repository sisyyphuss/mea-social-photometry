"""
Single-Animal Photometry, Tracking, and Coherence Analysis

Usage:
    python analyze_mouse.py --mouse_id 303 --age_group Young
"""

import os
import sys
import io
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import coherence
from scipy.stats import sem

# Add repository root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import butter_lowpass_filter, motion_correct_and_zscore
from src.peth import extract_bouts_from_events, slice_peth, compute_trial_metrics, calc_lateralization_index
from src.coherence import compute_spectral_coherence_bouts
from src.visualization import set_publication_style, COLOR_LEFT, COLOR_RIGHT


def analyze_mouse(mouse_id, age_group, raw_zip=None, tracking_dir=None, output_dir=None):
    set_publication_style()

    if raw_zip is None:
        raw_zip = os.path.expanduser("~/Downloads/Data_2.zip")
    if tracking_dir is None:
        tracking_dir = os.path.expanduser("~/Downloads/results_social/results_social")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "individual", age_group, f"Mouse_{mouse_id}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Analyzing Mouse {mouse_id} ({age_group})...")

    # 1. Load Photometry Raw CSV
    with zipfile.ZipFile(raw_zip, 'r') as z:
        csv_path_in_zip = f"Data_2/Social/{age_group.lower()}/Fluorescence_{mouse_id}.csv"
        with z.open(csv_path_in_zip) as f:
            content = f.read().decode('utf-8', errors='ignore')
            df_raw = pd.read_csv(io.StringIO(content), skiprows=1)
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

    # 4. Spectral Coherence
    freqs_soc, coh_soc_pre, coh_soc_post, v_soc_idx = compute_spectral_coherence_bouts(time, ch1_z, ch2_z, soc_on, 5.0, actual_fs)

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

    # 5. Build Trial Metrics
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

        mouse_trials.append({
            'Mouse_ID': mouse_id, 'Age_Group': age_group, 'Stimulus': 'Social',
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
        })

    for k in range(len(obj_aucs_l)):
        li = calc_lateralization_index(obj_aucs_l[k], obj_aucs_r[k])
        m02_bilat = round(float((obj_m02_l[k] + obj_m02_r[k]) / 2.0), 3)
        wb_bilat  = round(float((wb_obj_l[k] + wb_obj_r[k]) / 2.0), 3)
        mouse_trials.append({
            'Mouse_ID': mouse_id, 'Age_Group': age_group, 'Stimulus': 'NonSocial',
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
        })

    pd.DataFrame(mouse_trials).to_csv(os.path.join(output_dir, "trial_metrics.csv"), index=False)

    # 6. Tracking & Summary
    tracking_csv = os.path.join(tracking_dir, "individual", "Social", age_group, mouse_id, "synchronized_tracking_photometry.csv")
    df_tr = pd.read_csv(tracking_csv)
    x_pts, y_pts = df_tr['x_px'].dropna().values, df_tr['y_px'].dropna().values
    dist_px = float(np.sum(np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2)))
    stim = df_tr['compartment_stimulus'].values
    fps_tr = 60.0
    t_soc = float(np.sum(stim == 'Novel Mouse') / fps_tr)
    t_obj = float(np.sum(stim == 'Object') / fps_tr)
    spi_beh = (t_soc - t_obj) / (t_soc + t_obj + 1e-9)

    m_summary = {
        'Mouse_ID': str(mouse_id),
        'Age_Group': age_group,
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
        'Coherence_0_1_0_5Hz_Delta': round(float(np.nanmean(soc_c_b1_del)), 3) if len(soc_c_b1_del) > 0 else np.nan,
        'Coherence_0_5_1_5Hz_Delta': round(float(np.nanmean(soc_c_b2_del)), 3) if len(soc_c_b2_del) > 0 else np.nan,
        'Coherence_1_5_3_0Hz_Delta': round(float(np.nanmean(soc_c_b3_del)), 3) if len(soc_c_b3_del) > 0 else np.nan,
        'Mean_Baseline_Coherence_0_8_1_6Hz': round(float(np.nanmean(soc_c_std_pre)), 3) if len(soc_c_std_pre) > 0 else np.nan,
        'Mean_Social_Coherence_0_8_1_6Hz': round(float(np.nanmean(soc_c_std_post)), 3) if len(soc_c_std_post) > 0 else np.nan,
        'Mean_Coherence_Delta': round(float(np.nanmean(soc_c_std_post) - np.nanmean(soc_c_std_pre)), 3) if len(soc_c_std_pre) > 0 else np.nan,
        'Locomotor_Distance_px': round(dist_px, 1),
        'Time_in_Social_Chamber_s': round(t_soc, 1),
        'Time_in_Object_Chamber_s': round(t_obj, 1),
        'Social_Preference_Index': round(spi_beh, 3)
    }

    with open(os.path.join(output_dir, "mouse_summary.json"), "w") as jf:
        json.dump(m_summary, jf, indent=2)

    print(f"Analysis complete for Mouse {mouse_id}. Summary written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Single-Animal Photometry & Tracking Analysis")
    parser.add_argument("--mouse_id", type=str, required=True, help="Mouse identifier (e.g., 303)")
    parser.add_argument("--age_group", type=str, required=True, choices=["Young", "Old"], help="Age cohort (Young or Old)")
    parser.add_argument("--raw_zip", type=str, default=None, help="Path to raw photometry Data_2.zip")
    parser.add_argument("--tracking_dir", type=str, default=None, help="Path to tracking results directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    analyze_mouse(args.mouse_id, args.age_group, raw_zip=args.raw_zip, tracking_dir=args.tracking_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
