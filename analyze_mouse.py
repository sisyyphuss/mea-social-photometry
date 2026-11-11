"""
Single-Animal Photometry, Tracking, and Spectral Coherence Analysis

Usage:
    python analyze_mouse.py --mouse_id 303 --age_group Young
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.io_paths import resolve_data_paths, load_photometry_csv, load_tracking_csv
from src.preprocessing import butter_lowpass_filter, motion_correct_and_zscore
from src.peth import extract_bouts_from_events, slice_peth, compute_trial_metrics, calc_lateralization_index
from src.coherence import compute_spectral_coherence_bouts
from src.visualization import (
    set_publication_style,
    compute_normalized_occupancy,
    plot_individual_mouse_figures,
)


def process_single_mouse(mouse_id, age_group, raw_source=None, tracking_dir=None, output_dir=None, generate_plots=True):
    """Run complete photometry, tracking, and coherence analysis for a single mouse."""
    set_publication_style()
    mouse_id = str(mouse_id)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    raw_source, tracking_dir = resolve_data_paths(raw_source, tracking_dir, repo_root=repo_root)

    if output_dir is None:
        output_dir = os.path.join(repo_root, "results", "individual", age_group, f"Mouse_{mouse_id}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load photometry time-series
    df_raw = load_photometry_csv(raw_source, age_group, mouse_id)
    raw_t = pd.to_numeric(df_raw["TimeStamp"], errors="coerce").values
    time = (raw_t - raw_t[0]) / 1000.0 if np.nanmedian(np.diff(raw_t)) > 2.0 else raw_t - raw_t[0]
    actual_fs = 1.0 / np.median(np.diff(time))
    total_dur_min = (time[-1] - time[0]) / 60.0

    # 2. Filter and motion-correct against 410 nm reference
    ch1_470_f = butter_lowpass_filter(pd.to_numeric(df_raw["CH1-470"], errors="coerce").interpolate().values, 3.0, actual_fs)
    ch1_410_f = butter_lowpass_filter(pd.to_numeric(df_raw["CH1-410"], errors="coerce").interpolate().values, 3.0, actual_fs)
    ch2_470_f = butter_lowpass_filter(pd.to_numeric(df_raw["CH2-470"], errors="coerce").interpolate().values, 3.0, actual_fs)
    ch2_410_f = butter_lowpass_filter(pd.to_numeric(df_raw["CH2-410"], errors="coerce").interpolate().values, 3.0, actual_fs)

    _, ch1_z, ch1_fit = motion_correct_and_zscore(ch1_470_f, ch1_410_f)
    _, ch2_z, ch2_fit = motion_correct_and_zscore(ch2_470_f, ch2_410_f)

    # 3. Extract pure Social and Non-Social bouts and PETH metrics
    soc_on, soc_dur, obj_on, obj_dur = extract_bouts_from_events(df_raw, time)
    fs_nominal = 30.0
    rel_time, peth_soc_l, wb_soc_l, v_peth_soc_idx = slice_peth(time, ch1_z, soc_on, soc_dur, 3.0, 6.0, fs_nominal)
    _,        peth_soc_r, wb_soc_r, _              = slice_peth(time, ch2_z, soc_on, soc_dur, 3.0, 6.0, fs_nominal)
    _,        peth_obj_l, wb_obj_l, v_peth_obj_idx = slice_peth(time, ch1_z, obj_on, obj_dur, 3.0, 6.0, fs_nominal)
    _,        peth_obj_r, wb_obj_r, _              = slice_peth(time, ch2_z, obj_on, obj_dur, 3.0, 6.0, fs_nominal)

    soc_pks_l, soc_aucs_l, soc_m02_l = compute_trial_metrics(rel_time, peth_soc_l)
    soc_pks_r, soc_aucs_r, soc_m02_r = compute_trial_metrics(rel_time, peth_soc_r)
    obj_pks_l, obj_aucs_l, obj_m02_l = compute_trial_metrics(rel_time, peth_obj_l)
    obj_pks_r, obj_aucs_r, obj_m02_r = compute_trial_metrics(rel_time, peth_obj_r)

    # 4. Spectral coherence indexed strictly by original bout index
    freqs_soc, coh_soc_pre, coh_soc_post, v_coh_soc_idx = compute_spectral_coherence_bouts(
        time, ch1_z, ch2_z, soc_on, win_len=5.0, fs=actual_fs
    )

    coh_by_bout = {}
    if len(freqs_soc) > 0 and len(coh_soc_pre) > 0:
        m_full = (freqs_soc >= 0.1) & (freqs_soc <= 3.0)
        m_b1 = (freqs_soc >= 0.1) & (freqs_soc <= 0.5)
        m_b2 = (freqs_soc >= 0.5) & (freqs_soc <= 1.5)
        m_b3 = (freqs_soc >= 1.5) & (freqs_soc <= 3.0)
        m_std = (freqs_soc >= 0.8) & (freqs_soc <= 1.6)
        for pos, orig_idx in enumerate(v_coh_soc_idx):
            c_pre = coh_soc_pre[pos]
            c_post = coh_soc_post[pos]
            coh_by_bout[orig_idx] = {
                "full_pre": round(float(np.mean(c_pre[m_full])), 3),
                "full_post": round(float(np.mean(c_post[m_full])), 3),
                "full_del": round(float(np.mean(c_post[m_full]) - np.mean(c_pre[m_full])), 3),
                "b1_del": round(float(np.mean(c_post[m_b1]) - np.mean(c_pre[m_b1])), 3),
                "b2_del": round(float(np.mean(c_post[m_b2]) - np.mean(c_pre[m_b2])), 3),
                "b3_del": round(float(np.mean(c_post[m_b3]) - np.mean(c_pre[m_b3])), 3),
                "std_pre": round(float(np.mean(c_pre[m_std])), 3),
                "std_post": round(float(np.mean(c_post[m_std])), 3),
                "std_del": round(float(np.mean(c_post[m_std]) - np.mean(c_pre[m_std])), 3),
            }

    # 5. Build trial metrics table aligned strictly to original bout IDs
    mouse_trials = []
    soc_lis = []
    for k, orig_idx in enumerate(v_peth_soc_idx):
        li = calc_lateralization_index(soc_aucs_l[k], soc_aucs_r[k])
        soc_lis.append(li)
        c_info = coh_by_bout.get(orig_idx, {})
        mouse_trials.append({
            "Mouse_ID": mouse_id,
            "Age_Group": age_group,
            "Stimulus": "Social",
            "Bout": k + 1,
            "Duration_s": round(float(soc_dur[orig_idx]), 2),
            "Left_MeA_AUC": round(float(soc_aucs_l[k]), 3),
            "Right_MeA_AUC": round(float(soc_aucs_r[k]), 3),
            "Left_MeA_Peak_Z": round(float(soc_pks_l[k]), 3),
            "Right_MeA_Peak_Z": round(float(soc_pks_r[k]), 3),
            "Left_MeA_Whole_Bout_Mean_Z": round(float(wb_soc_l[k]), 3),
            "Right_MeA_Whole_Bout_Mean_Z": round(float(wb_soc_r[k]), 3),
            "Whole_Bout_Mean_Z_Bilateral": round(float((wb_soc_l[k] + wb_soc_r[k]) / 2.0), 3),
            "Left_MeA_Mean_Z_0_2s": round(float(soc_m02_l[k]), 3),
            "Right_MeA_Mean_Z_0_2s": round(float(soc_m02_r[k]), 3),
            "Mean_Z_0_2s_Bilateral": round(float((soc_m02_l[k] + soc_m02_r[k]) / 2.0), 3),
            "Lateralization_Index": round(float(li), 3),
            "Coherence_Full_0_1_3_0Hz_Base": c_info.get("full_pre", np.nan),
            "Coherence_Full_0_1_3_0Hz_Social": c_info.get("full_post", np.nan),
            "Coherence_Full_0_1_3_0Hz_Delta": c_info.get("full_del", np.nan),
            "Coherence_0_1_0_5Hz_Delta": c_info.get("b1_del", np.nan),
            "Coherence_0_5_1_5Hz_Delta": c_info.get("b2_del", np.nan),
            "Coherence_1_5_3_0Hz_Delta": c_info.get("b3_del", np.nan),
            "Baseline_Coherence": c_info.get("std_pre", np.nan),
            "Interaction_Coherence": c_info.get("std_post", np.nan),
            "Coherence_Delta": c_info.get("std_del", np.nan),
        })

    obj_lis = []
    for k, orig_idx in enumerate(v_peth_obj_idx):
        li = calc_lateralization_index(obj_aucs_l[k], obj_aucs_r[k])
        obj_lis.append(li)
        mouse_trials.append({
            "Mouse_ID": mouse_id,
            "Age_Group": age_group,
            "Stimulus": "NonSocial",
            "Bout": k + 1,
            "Duration_s": round(float(obj_dur[orig_idx]), 2),
            "Left_MeA_AUC": round(float(obj_aucs_l[k]), 3),
            "Right_MeA_AUC": round(float(obj_aucs_r[k]), 3),
            "Left_MeA_Peak_Z": round(float(obj_pks_l[k]), 3),
            "Right_MeA_Peak_Z": round(float(obj_pks_r[k]), 3),
            "Left_MeA_Whole_Bout_Mean_Z": round(float(wb_obj_l[k]), 3),
            "Right_MeA_Whole_Bout_Mean_Z": round(float(wb_obj_r[k]), 3),
            "Whole_Bout_Mean_Z_Bilateral": round(float((wb_obj_l[k] + wb_obj_r[k]) / 2.0), 3),
            "Left_MeA_Mean_Z_0_2s": round(float(obj_m02_l[k]), 3),
            "Right_MeA_Mean_Z_0_2s": round(float(obj_m02_r[k]), 3),
            "Mean_Z_0_2s_Bilateral": round(float((obj_m02_l[k] + obj_m02_r[k]) / 2.0), 3),
            "Lateralization_Index": round(float(li), 3),
            "Coherence_Full_0_1_3_0Hz_Base": np.nan,
            "Coherence_Full_0_1_3_0Hz_Social": np.nan,
            "Coherence_Full_0_1_3_0Hz_Delta": np.nan,
            "Coherence_0_1_0_5Hz_Delta": np.nan,
            "Coherence_0_5_1_5Hz_Delta": np.nan,
            "Coherence_1_5_3_0Hz_Delta": np.nan,
            "Baseline_Coherence": np.nan,
            "Interaction_Coherence": np.nan,
            "Coherence_Delta": np.nan,
        })

    pd.DataFrame(mouse_trials).to_csv(os.path.join(output_dir, "trial_metrics.csv"), index=False)

    # 6. Load 60 fps tracking data
    df_tr = load_tracking_csv(tracking_dir, age_group, mouse_id)
    x_pts = df_tr["x_px"].dropna().values
    y_pts = df_tr["y_px"].dropna().values
    speed = df_tr["speed_px_sec"].dropna().values
    dist_px = float(np.sum(np.sqrt(np.diff(x_pts) ** 2 + np.diff(y_pts) ** 2)))

    stim = df_tr["compartment_stimulus"].values
    fps_tr = 60.0
    time_soc_chamber = float(np.sum(stim == "Novel Mouse") / fps_tr)
    time_obj_chamber = float(np.sum(stim == "Object") / fps_tr)
    spi_beh = (time_soc_chamber - time_obj_chamber) / (time_soc_chamber + time_obj_chamber + 1e-9)

    h_norm, h_smooth_counts = compute_normalized_occupancy(x_pts, y_pts)

    # 7. Mouse summary metadata
    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else np.nan

    mean_soc_auc_l, mean_soc_auc_r = safe_mean(soc_aucs_l), safe_mean(soc_aucs_r)
    mean_soc_pk_l, mean_soc_pk_r = safe_mean(soc_pks_l), safe_mean(soc_pks_r)
    mean_soc_wb_l, mean_soc_wb_r = safe_mean(wb_soc_l), safe_mean(wb_soc_r)
    mean_soc_wb_bilat = float((mean_soc_wb_l + mean_soc_wb_r) / 2.0)
    mean_soc_m02_l, mean_soc_m02_r = safe_mean(soc_m02_l), safe_mean(soc_m02_r)
    mean_soc_m02_bilat = float((mean_soc_m02_l + mean_soc_m02_r) / 2.0)
    soc_li = calc_lateralization_index(mean_soc_auc_l, mean_soc_auc_r)

    mean_obj_auc_l, mean_obj_auc_r = safe_mean(obj_aucs_l), safe_mean(obj_aucs_r)
    mean_obj_pk_l, mean_obj_pk_r = safe_mean(obj_pks_l), safe_mean(obj_pks_r)
    mean_obj_wb_l, mean_obj_wb_r = safe_mean(wb_obj_l), safe_mean(wb_obj_r)
    mean_obj_wb_bilat = float((mean_obj_wb_l + mean_obj_wb_r) / 2.0)
    mean_obj_m02_l, mean_obj_m02_r = safe_mean(obj_m02_l), safe_mean(obj_m02_r)
    mean_obj_m02_bilat = float((mean_obj_m02_l + mean_obj_m02_r) / 2.0)
    obj_li = calc_lateralization_index(mean_obj_auc_l, mean_obj_auc_r)

    c_full_pre_vals = [v["full_pre"] for v in coh_by_bout.values()]
    c_full_post_vals = [v["full_post"] for v in coh_by_bout.values()]
    c_b1_vals = [v["b1_del"] for v in coh_by_bout.values()]
    c_b2_vals = [v["b2_del"] for v in coh_by_bout.values()]
    c_b3_vals = [v["b3_del"] for v in coh_by_bout.values()]
    c_std_pre_vals = [v["std_pre"] for v in coh_by_bout.values()]
    c_std_post_vals = [v["std_post"] for v in coh_by_bout.values()]

    m_c_full_pre = float(np.mean(c_full_pre_vals)) if len(c_full_pre_vals) > 0 else np.nan
    m_c_full_post = float(np.mean(c_full_post_vals)) if len(c_full_post_vals) > 0 else np.nan
    m_c_full_del = float(m_c_full_post - m_c_full_pre) if not np.isnan(m_c_full_pre) else np.nan
    m_c_b1_del = float(np.mean(c_b1_vals)) if len(c_b1_vals) > 0 else np.nan
    m_c_b2_del = float(np.mean(c_b2_vals)) if len(c_b2_vals) > 0 else np.nan
    m_c_b3_del = float(np.mean(c_b3_vals)) if len(c_b3_vals) > 0 else np.nan
    m_coh_pre = float(np.mean(c_std_pre_vals)) if len(c_std_pre_vals) > 0 else np.nan
    m_coh_post = float(np.mean(c_std_post_vals)) if len(c_std_post_vals) > 0 else np.nan
    m_coh_del = float(m_coh_post - m_coh_pre) if not np.isnan(m_coh_pre) else np.nan

    m_summary = {
        "Mouse_ID": mouse_id,
        "Age_Group": age_group,
        "Recording_Duration_Min": round(total_dur_min, 1),
        "Social_Bout_Count": len(soc_on),
        "Analyzed_Social_Bouts_in_PETH": len(soc_aucs_l),
        "Social_Total_Time_s": round(float(np.sum(soc_dur)), 1),
        "NonSocial_Bout_Count": len(obj_on),
        "Analyzed_NonSocial_Bouts_in_PETH": len(obj_aucs_l),
        "NonSocial_Total_Time_s": round(float(np.sum(obj_dur)), 1),
        "Social_AUC_Left": round(mean_soc_auc_l, 3),
        "Social_AUC_Right": round(mean_soc_auc_r, 3),
        "Social_Peak_Left": round(mean_soc_pk_l, 3),
        "Social_Peak_Right": round(mean_soc_pk_r, 3),
        "Social_Whole_Bout_Mean_Z_Left": round(mean_soc_wb_l, 3),
        "Social_Whole_Bout_Mean_Z_Right": round(mean_soc_wb_r, 3),
        "Social_Whole_Bout_Mean_Z_Bilateral": round(mean_soc_wb_bilat, 3),
        "Social_Mean_Z_0_2s_Left": round(mean_soc_m02_l, 3),
        "Social_Mean_Z_0_2s_Right": round(mean_soc_m02_r, 3),
        "Social_Mean_Z_0_2s_Bilateral": round(mean_soc_m02_bilat, 3),
        "Social_Lateralization_Index": round(float(soc_li), 3),
        "NonSocial_AUC_Left": round(mean_obj_auc_l, 3),
        "NonSocial_AUC_Right": round(mean_obj_auc_r, 3),
        "NonSocial_Peak_Left": round(mean_obj_pk_l, 3),
        "NonSocial_Peak_Right": round(mean_obj_pk_r, 3),
        "NonSocial_Whole_Bout_Mean_Z_Left": round(mean_obj_wb_l, 3),
        "NonSocial_Whole_Bout_Mean_Z_Right": round(mean_obj_wb_r, 3),
        "NonSocial_Whole_Bout_Mean_Z_Bilateral": round(mean_obj_wb_bilat, 3),
        "NonSocial_Mean_Z_0_2s_Left": round(mean_obj_m02_l, 3),
        "NonSocial_Mean_Z_0_2s_Right": round(mean_obj_m02_r, 3),
        "NonSocial_Mean_Z_0_2s_Bilateral": round(mean_obj_m02_bilat, 3),
        "NonSocial_Lateralization_Index": round(float(obj_li), 3),
        "Coherence_Full_Spectrum_0_1_3_0Hz_Base": round(m_c_full_pre, 3),
        "Coherence_Full_Spectrum_0_1_3_0Hz_Social": round(m_c_full_post, 3),
        "Coherence_Full_Spectrum_0_1_3_0Hz_Delta": round(m_c_full_del, 3),
        "Coherence_Band1_0_1_0_5Hz_Delta": round(m_c_b1_del, 3),
        "Coherence_Band2_0_5_1_5Hz_Delta": round(m_c_b2_del, 3),
        "Coherence_Band3_1_5_3_0Hz_Delta": round(m_c_b3_del, 3),
        "Mean_Baseline_Coherence_0_8_1_6Hz": round(m_coh_pre, 3),
        "Mean_Social_Coherence_0_8_1_6Hz": round(m_coh_post, 3),
        "Mean_Coherence_Delta": round(m_coh_del, 3),
        "Locomotor_Distance_px": round(dist_px, 1),
        "Mean_Speed_px_s": round(float(np.mean(speed)), 2),
        "Chamber_Time_Social_s": round(time_soc_chamber, 1),
        "Chamber_Time_Object_s": round(time_obj_chamber, 1),
        "Behavioral_Social_Preference_Index": round(float(spi_beh), 3),
    }

    with open(os.path.join(output_dir, "mouse_summary.json"), "w", encoding="utf-8") as jf:
        json.dump(m_summary, jf, indent=2)

    if generate_plots:
        c_full_del_list = [v["full_del"] for v in coh_by_bout.values()]
        c_del_bands = [c_b1_vals, c_b2_vals, c_b3_vals, c_full_del_list]
        plot_individual_mouse_figures(
            out_dir=output_dir,
            mouse_id=mouse_id,
            age_group=age_group,
            time=time,
            ch1_470_f=ch1_470_f,
            ch1_fit=ch1_fit,
            ch2_470_f=ch2_470_f,
            ch2_fit=ch2_fit,
            ch1_z=ch1_z,
            ch2_z=ch2_z,
            soc_on=soc_on,
            soc_dur=soc_dur,
            obj_on=obj_on,
            obj_dur=obj_dur,
            rel_time=rel_time,
            peth_soc_l=peth_soc_l,
            peth_soc_r=peth_soc_r,
            peth_obj_l=peth_obj_l,
            peth_obj_r=peth_obj_r,
            mean_soc_wb_bilat=mean_soc_wb_bilat,
            mean_obj_wb_bilat=mean_obj_wb_bilat,
            soc_lis=soc_lis,
            obj_lis=obj_lis,
            x_pts=x_pts,
            y_pts=y_pts,
            dist_px=dist_px,
            spi_beh=spi_beh,
            h_smooth_counts=h_smooth_counts,
            freqs_soc=freqs_soc,
            coh_soc_pre=coh_soc_pre,
            coh_soc_post=coh_soc_post,
            c_del_bands=c_del_bands,
            m_c_full_del=m_c_full_del,
        )

    return {
        "summary": m_summary,
        "trials": mouse_trials,
        "rel_time": rel_time,
        "peth_soc_l": peth_soc_l,
        "peth_soc_r": peth_soc_r,
        "peth_obj_l": peth_obj_l,
        "peth_obj_r": peth_obj_r,
        "h_norm": h_norm,
        "freqs_soc": freqs_soc,
        "coh_soc_pre": coh_soc_pre,
        "coh_soc_post": coh_soc_post,
        "m_c_full_pre": m_c_full_pre,
        "m_c_full_post": m_c_full_post,
        "m_c_full_del": m_c_full_del,
        "m_coh_del": m_coh_del,
    }


def main():
    parser = argparse.ArgumentParser(description="Single-Animal Photometry & Tracking Analysis")
    parser.add_argument("--mouse_id", type=str, required=True, help="Mouse identifier (e.g., 303)")
    parser.add_argument("--age_group", type=str, required=True, choices=["Young", "Old"], help="Age cohort (Young or Old)")
    parser.add_argument("--raw_zip", type=str, default=None, help="Path to Data_2.zip or extracted Data_2 folder")
    parser.add_argument("--tracking_dir", type=str, default=None, help="Path to results_social tracking directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    res = process_single_mouse(
        mouse_id=args.mouse_id,
        age_group=args.age_group,
        raw_source=args.raw_zip,
        tracking_dir=args.tracking_dir,
        output_dir=args.output_dir,
    )
    print(f"Analysis complete for Mouse {args.mouse_id} ({args.age_group}).")
    return res


if __name__ == "__main__":
    main()
