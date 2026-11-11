"""
Extended Analysis: Optical QC, Transient Dynamics, Mixed Modeling, and Stimulus Selectivity

Usage:
    python run_extended_analysis.py [--raw_zip PATH] [--tracking_dir PATH] [--results_dir PATH]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, pearsonr, sem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.io_paths import resolve_data_paths, load_photometry_csv, load_tracking_csv
from src.preprocessing import butter_lowpass_filter, motion_correct_and_zscore
from src.peth import extract_bouts_from_events, slice_peth
from src.visualization import set_publication_style, save_figure, COLOR_YOUNG, COLOR_OLD
from src.qc import compute_optical_qc, compute_speed_coupling
from src.transients import detect_transients
from src.modeling import run_nonparametric_and_lomo, fit_trial_lmm, find_consecutive_peth_windows
from src.behavior_neural import (
    compute_social_object_auroc,
    compute_bout_stratification,
    compute_offset_and_zone_entry,
    compute_lagged_crosscorr,
    compute_brain_behavior_coupling,
)


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def run_extended(raw_zip=None, tracking_dir=None, results_dir=None):
    set_publication_style()
    rng = np.random.default_rng(42)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if results_dir is None:
        results_dir = os.path.join(repo_root, "results")
    raw_source, tracking_dir = resolve_data_paths(raw_zip, tracking_dir, repo_root=repo_root)

    tables_dir = os.path.join(results_dir, "tables")
    figs_dir = os.path.join(results_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    df_sum = pd.read_csv(os.path.join(tables_dir, "master_per_mouse_summary.csv"))
    df_trials = pd.read_csv(os.path.join(tables_dir, "master_all_trials_dataset.csv"))
    df_sum["Mouse_ID"] = df_sum["Mouse_ID"].astype(str)
    df_trials["Mouse_ID"] = df_trials["Mouse_ID"].astype(str)

    young_ids = ["303", "305", "307", "308", "309", "311", "326", "331"]
    old_ids = ["315", "316", "317", "318", "319", "323", "50", "56", "66", "69"]

    extended_rows = []
    peth_470 = {"Young": [], "Old": []}
    peth_410 = {"Young": [], "Old": []}
    offset_waves = {"Young": [], "Old": []}
    xcorr_waves = {"Young": [], "Old": []}
    rel_time_peth = None
    lags_s = None

    print("Running extended optical QC, transient, selectivity, and behavioral analyses...")
    for grp, ids in [("Young", young_ids), ("Old", old_ids)]:
        for mid in ids:
            df_raw = load_photometry_csv(raw_source, grp, mid)
            raw_t = pd.to_numeric(df_raw["TimeStamp"], errors="coerce").values
            time = (raw_t - raw_t[0]) / 1000.0 if np.nanmedian(np.diff(raw_t)) > 2.0 else raw_t - raw_t[0]
            actual_fs = 1.0 / np.median(np.diff(time))

            raw470_l = pd.to_numeric(df_raw["CH1-470"], errors="coerce").interpolate().values
            raw410_l = pd.to_numeric(df_raw["CH1-410"], errors="coerce").interpolate().values
            raw470_r = pd.to_numeric(df_raw["CH2-470"], errors="coerce").interpolate().values
            raw410_r = pd.to_numeric(df_raw["CH2-410"], errors="coerce").interpolate().values

            f470_l = butter_lowpass_filter(raw470_l, 3.0, actual_fs)
            f410_l = butter_lowpass_filter(raw410_l, 3.0, actual_fs)
            f470_r = butter_lowpass_filter(raw470_r, 3.0, actual_fs)
            f410_r = butter_lowpass_filter(raw410_r, 3.0, actual_fs)

            dff_l, z_l, _ = motion_correct_and_zscore(f470_l, f410_l)
            dff_r, z_r, _ = motion_correct_and_zscore(f470_r, f410_r)

            soc_on, soc_dur, _, _ = extract_bouts_from_events(df_raw, time)
            soc_offsets = soc_on + soc_dur

            rt, p_l, _, _ = slice_peth(time, z_l, soc_on, soc_dur, 3.0, 6.0, 30.0)
            _,  p_r, _, _ = slice_peth(time, z_r, soc_on, soc_dur, 3.0, 6.0, 30.0)
            rel_time_peth = rt
            peth_470[grp].append(np.mean((p_l + p_r) / 2.0, axis=0))

            qc_res = compute_optical_qc(
                time, raw470_l, raw410_l, raw470_r, raw410_r, dff_l, dff_r, soc_on, fs=actual_fs
            )
            peth_410[grp].append(qc_res["iso_mean_trace"])

            df_tr = load_tracking_csv(tracking_dir, grp, mid)
            spd_res = compute_speed_coupling(df_tr, time, qc_res["f470_l"], qc_res["f470_r"])
            trans_res = detect_transients(time, z_l, z_r, soc_on, soc_dur, fs=actual_fs)
            off_zone_res = compute_offset_and_zone_entry(time, z_l, z_r, soc_offsets, df_tr)
            xc_res = compute_lagged_crosscorr(df_tr, max_lag_s=2.0, fps=60.0)

            offset_waves[grp].append(off_zone_res["offset_mean_trace"])
            lags_s = xc_res["lags_s"]
            xcorr_waves[grp].append(xc_res["xcorr"])

            extended_rows.append({
                "Mouse_ID": mid,
                "Age_Group": grp,
                "Bleach_410_Pct": qc_res["bleach_410_pct"],
                "SNR": qc_res["snr"],
                "Fit_410_470_R2": qc_res["fit_r2"],
                "Iso_410_0_2s_Mean_Z": qc_res["iso_0_2s_mean_z"],
                "Speed_r_Raw470": spd_res["speed_r_raw470"],
                "Speed_r_CorrectedZ": spd_res["speed_r_corrected"],
                "Transient_Count_Total": trans_res["transient_count_total"],
                "Transient_Rate_Total_per_min": trans_res["transient_rate_total_per_min"],
                "Transient_Rate_Basal_per_min": trans_res["transient_rate_basal_per_min"],
                "Transient_Rate_Social_per_min": trans_res["transient_rate_social_per_min"],
                "Transient_Peak_Amp_Mean": trans_res["transient_peak_amp_mean"],
                "Transient_FWHM_s": trans_res["transient_fwhm_s"],
                "Offset_Disengagement_Delta_Z": off_zone_res["offset_delta"],
                "Zone_Entry_Social_Delta_Z": off_zone_res["zone_entry_soc_delta"],
                "Zone_Entry_Object_Delta_Z": off_zone_res["zone_entry_obj_delta"],
                "CrossCorr_ZeroLag_r": xc_res["zero_lag_r"],
                "CrossCorr_Max_r": xc_res["max_r"],
                "CrossCorr_Peak_Lag_s": xc_res["peak_lag_s"],
            })

    df_ext = pd.DataFrame(extended_rows)

    df_auroc = compute_social_object_auroc(df_trials)
    df_strat = compute_bout_stratification(df_trials)
    df_ext = df_ext.merge(df_auroc.drop(columns=["Age_Group"]), on="Mouse_ID", how="left")
    df_ext = df_ext.merge(df_strat.drop(columns=["Age_Group"]), on="Mouse_ID", how="left")
    df_ext.to_csv(os.path.join(tables_dir, "extended_mouse_metrics.csv"), index=False)

    df_nonparam_lomo, df_lomo_drops = run_nonparametric_and_lomo(df_sum)
    df_nonparam_lomo.to_csv(os.path.join(tables_dir, "nonparametric_and_lomo_summary.csv"), index=False)
    df_lomo_drops.to_csv(os.path.join(tables_dir, "lomo_jackknife_results.csv"), index=False)

    df_lmm, dur_checks = fit_trial_lmm(df_trials)
    df_lmm.to_csv(os.path.join(tables_dir, "lmm_model_summary.csv"), index=False)

    peth_div = find_consecutive_peth_windows(
        rel_time_peth, np.array(peth_470["Young"]), np.array(peth_470["Old"]), min_consec_s=0.20, fs=30.0, n_boot=2000
    )
    df_bb = compute_brain_behavior_coupling(df_sum)
    df_bb.to_csv(os.path.join(tables_dir, "brain_behavior_coupling.csv"), index=False)

    # Figure 7: Optical controls
    fig7, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    y470_m, y470_s = np.mean(peth_470["Young"], axis=0), sem(peth_470["Young"], axis=0)
    o470_m, o470_s = np.mean(peth_470["Old"], axis=0), sem(peth_470["Old"], axis=0)
    all410 = np.vstack([peth_410["Young"], peth_410["Old"]])
    m410, s410 = np.mean(all410, axis=0), sem(all410, axis=0)

    ax1.plot(rel_time_peth, y470_m, color=COLOR_YOUNG, lw=2.2, label="Young 470 nm (n=8)")
    ax1.fill_between(rel_time_peth, y470_m - y470_s, y470_m + y470_s, color=COLOR_YOUNG, alpha=0.2)
    ax1.plot(rel_time_peth, o470_m, color=COLOR_OLD, lw=2.2, label="Old 470 nm (n=10)")
    ax1.fill_between(rel_time_peth, o470_m - o470_s, o470_m + o470_s, color=COLOR_OLD, alpha=0.2)
    ax1.plot(rel_time_peth, m410, color="#4d4d4d", lw=2.0, linestyle="--", label="410 nm Control (n=18)")
    ax1.fill_between(rel_time_peth, m410 - s410, m410 + s410, color="#999999", alpha=0.25)
    ax1.axvline(0, color="black", linestyle=":", lw=1.2)
    ax1.axhline(0, color="gray", linestyle="-", lw=0.6, alpha=0.5)
    ax1.set_xlabel("Time Relative to Social Onset (s)")
    ax1.set_ylabel("Baseline-Corrected Z-Score")
    ax1.set_title("A | Calcium (470 nm) vs Control (410 nm) PETH", fontweight="bold")
    ax1.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.4)

    for _, row in df_ext.iterrows():
        c = COLOR_YOUNG if row["Age_Group"] == "Young" else COLOR_OLD
        ax2.plot([1, 2], [row["Speed_r_Raw470"], row["Speed_r_CorrectedZ"]], color=c, alpha=0.45, lw=1.2)
    y_raw = df_ext[df_ext["Age_Group"] == "Young"]["Speed_r_Raw470"]
    o_raw = df_ext[df_ext["Age_Group"] == "Old"]["Speed_r_Raw470"]
    y_cor = df_ext[df_ext["Age_Group"] == "Young"]["Speed_r_CorrectedZ"]
    o_cor = df_ext[df_ext["Age_Group"] == "Old"]["Speed_r_CorrectedZ"]
    ax2.scatter(np.ones(len(y_raw)), y_raw, color=COLOR_YOUNG, s=45, zorder=3, label="Young")
    ax2.scatter(np.ones(len(o_raw)), o_raw, color=COLOR_OLD, s=45, zorder=3, label="Old")
    ax2.scatter(2 * np.ones(len(y_cor)), y_cor, color=COLOR_YOUNG, s=45, zorder=3)
    ax2.scatter(2 * np.ones(len(o_cor)), o_cor, color=COLOR_OLD, s=45, zorder=3)
    ax2.axhline(0, color="gray", linestyle="--", lw=1.0)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Raw 470 nm\nvs Speed", "Corrected Z\nvs Speed"], fontweight="bold")
    ax2.set_xlim(0.5, 2.5)
    ax2.set_ylabel("Pearson Correlation with Speed (r)")
    _, p_scrub = ttest_rel(df_ext["Speed_r_Raw470"], df_ext["Speed_r_CorrectedZ"])
    ax2.set_title(f"B | Motion-Artifact Correction (Speed Correlation)\n(Paired Reduction p = {p_scrub:.4f}{sig_stars(p_scrub)})", fontweight="bold")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

    y_snr = df_ext[df_ext["Age_Group"] == "Young"]["SNR"]
    o_snr = df_ext[df_ext["Age_Group"] == "Old"]["SNR"]
    _, p_snr = ttest_ind(y_snr, o_snr, equal_var=False)

    df_sum["Obj_Peak_Bilat"] = (df_sum["NonSocial_Peak_Left"] + df_sum["NonSocial_Peak_Right"]) / 2.0
    y_obj = df_sum[df_sum["Age_Group"] == "Young"]["Obj_Peak_Bilat"]
    o_obj = df_sum[df_sum["Age_Group"] == "Old"]["Obj_Peak_Bilat"]
    _, p_obj = ttest_ind(y_obj, o_obj, equal_var=False)

    ax3_twin = ax3.twinx()
    ax3.boxplot([y_snr, o_snr], positions=[0.8, 1.4], widths=0.35, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="#f0f0f0", lw=1.1), medianprops=dict(color="black", lw=1.6))
    ax3.scatter(0.8 + rng.normal(0, 0.03, len(y_snr)), y_snr, color=COLOR_YOUNG, s=40, alpha=0.85)
    ax3.scatter(1.4 + rng.normal(0, 0.03, len(o_snr)), o_snr, color=COLOR_OLD, s=40, alpha=0.85)

    ax3_twin.boxplot([y_obj, o_obj], positions=[2.4, 3.0], widths=0.35, patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor="#e8f4f8", lw=1.1), medianprops=dict(color="black", lw=1.6))
    ax3_twin.scatter(2.4 + rng.normal(0, 0.03, len(y_obj)), y_obj, color=COLOR_YOUNG, s=40, alpha=0.85)
    ax3_twin.scatter(3.0 + rng.normal(0, 0.03, len(o_obj)), o_obj, color=COLOR_OLD, s=40, alpha=0.85)

    ax3.axvline(1.9, color="gray", linestyle=":", lw=1.2)
    ax3.set_xticks([0.8, 1.4, 2.4, 3.0])
    ax3.set_xticklabels(["Young\nSNR", "Old\nSNR", "Young\nObject", "Old\nObject"], fontsize=8.5, fontweight="bold")
    ax3.set_ylabel("Session SNR")
    ax3_twin.set_ylabel("Object Peak Z-Score")
    ax3_twin.spines["top"].set_visible(False)
    ax3.set_title(f"C | Signal Quality & Viral Gain Controls\n(SNR p = {p_snr:.2f} | Object Peak p = {p_obj:.2f}{', n.s.' if p_obj >= 0.05 else sig_stars(p_obj)})", fontweight="bold")
    ax3.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig7.suptitle("Figure 7 | Optical Controls: Isosbestic Reference, Motion Correction & Signal Quality", fontweight="bold")
    plt.tight_layout()
    save_figure(fig7, os.path.join(figs_dir, "Figure7_Optical_QC_and_Isosbestic_Controls.png"), dpi=300)
    save_figure(fig7, os.path.join(figs_dir, "Figure7_Optical_QC_and_Isosbestic_Controls.pdf"))
    plt.close(fig7)

    # Figure 8: Mixed-effects modeling, jackknife stability & time-window controls
    fig8, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13.2, 9.6))

    ax1.plot(rel_time_peth, peth_div["y_mean"], color=COLOR_YOUNG, lw=2.2, label="Young Mean +/- 95% Boot CI")
    ax1.fill_between(rel_time_peth, peth_div["y_ci_lo"], peth_div["y_ci_hi"], color=COLOR_YOUNG, alpha=0.22)
    ax1.plot(rel_time_peth, peth_div["o_mean"], color=COLOR_OLD, lw=2.2, label="Old Mean +/- 95% Boot CI")
    ax1.fill_between(rel_time_peth, peth_div["o_ci_lo"], peth_div["o_ci_hi"], color=COLOR_OLD, alpha=0.22)
    for idx_i, (t0, t1) in enumerate(peth_div["sig_intervals_s"]):
        lbl = "p < 0.05 (>= 0.2 s consec.)" if idx_i == 0 else None
        ax1.axvspan(t0, t1, color="#fdae61", alpha=0.35, label=lbl)
    ax1.axvline(0, color="black", linestyle=":", lw=1.2)
    ax1.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax1.set_xlabel("Time Relative to Social Onset (s)")
    ax1.set_ylabel("Baseline-Corrected Z-Score")
    all_ints = [(round(t0, 2), round(t1, 2)) for t0, t1 in peth_div["sig_intervals_s"]]
    int_txt = ", ".join([f"{a:+.2f} to {b:+.2f}s" for a, b in all_ints]) if all_ints else "None"
    ax1.set_title(f"A | Bootstrapped PETH & Consecutive Divergence\n(Significant Windows: {int_txt})", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.88, edgecolor="none", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.4)

    y_soc_tr = df_trials[(df_trials["Stimulus"] == "Social") & (df_trials["Age_Group"] == "Young")].copy()
    y_soc_tr["AUC_Bilat"] = (y_soc_tr["Left_MeA_AUC"] + y_soc_tr["Right_MeA_AUC"]) / 2.0
    ax2.scatter(y_soc_tr["Duration_s"], y_soc_tr["AUC_Bilat"], color="#7b3294", alpha=0.5, s=28,
                label=f"Raw AUC (r={dur_checks['Young_r_Dur_AUC']}, p={dur_checks['Young_p_Dur_AUC']})")
    ax2.scatter(y_soc_tr["Duration_s"], y_soc_tr["Whole_Bout_Mean_Z_Bilateral"], color="#008837", alpha=0.65, s=28,
                label=f"Whole-Bout Mean Z (r={dur_checks['Young_r_Dur_WB_MeanZ']}, p={dur_checks['Young_p_Dur_WB_MeanZ']}{' n.s.' if dur_checks['Young_p_Dur_WB_MeanZ'] >= 0.05 else '*'})")
    m1, b1 = np.polyfit(y_soc_tr["Duration_s"], y_soc_tr["AUC_Bilat"], 1)
    m2, b2 = np.polyfit(y_soc_tr["Duration_s"], y_soc_tr["Whole_Bout_Mean_Z_Bilateral"], 1)
    xs_line = np.linspace(0.2, y_soc_tr["Duration_s"].max(), 50)
    ax2.plot(xs_line, m1 * xs_line + b1, color="#7b3294", lw=2.0)
    ax2.plot(xs_line, m2 * xs_line + b2, color="#008837", lw=2.0)
    ax2.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax2.set_xlabel("Social Bout Duration (s)")
    ax2.set_ylabel("Neural Response (AUC or Mean Z)")
    ax2.set_title(f"B | Duration Normalization via Whole-Bout Mean Z\n(Young Cohort, n = {len(y_soc_tr)} Social Bouts)", fontweight="bold")
    ax2.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.4)

    lomo_metrics = ["Peak MeA Z (0-4 s)", "Whole-Bout Mean Z", "Fixed 0-2 s Mean Z"]
    lomo_short_names = ["Peak Z", "Whole-Bout Mean", "0-2 s Window"]
    lomo_colors = ["#1b9e77", "#2c7bb6", "#d95f02"]
    lomo_xtick_labels = []
    for i, (mname, sname, col) in enumerate(zip(lomo_metrics, lomo_short_names, lomo_colors)):
        sub_l = df_lomo_drops[df_lomo_drops["Metric"] == mname]
        sig_cnt = int((sub_l["Welch_p"] < 0.05).sum())
        lomo_xtick_labels.append(f"{sname}\n({sig_cnt}/{len(sub_l)} sig.)")
        ax3.scatter(np.full(len(sub_l), i + 1) + rng.normal(0, 0.06, len(sub_l)), sub_l["Welch_p"], color=col, s=45, alpha=0.8)
        ax3.hlines(sub_l["Welch_p"].max(), i + 0.8, i + 1.2, color=col, lw=2.2)
    ax3.axhline(0.05, color="red", linestyle="--", lw=1.3, label="p = 0.05 Threshold")
    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(lomo_xtick_labels, fontweight="bold")
    ax3.set_ylabel("Welch's p-value Across Single-Mouse Drops")
    ax3.set_title("C | 18-Fold Leave-One-Mouse-Out Jackknife Stability", fontweight="bold")
    ax3.legend(loc="upper left", frameon=False)
    ax3.grid(True, axis="y", linestyle=":", alpha=0.4)

    n_soc_trials = int((df_trials["Stimulus"] == "Social").sum())
    lmm_age = df_lmm[df_lmm["Predictor"] == "Is_Old"]
    y_pos = np.arange(len(lmm_age))
    betas = lmm_age["Beta"].values
    errs_lo = betas - lmm_age["CI_Low"].values
    errs_hi = lmm_age["CI_High"].values - betas
    ax4.errorbar(betas, y_pos, xerr=[errs_lo, errs_hi], fmt="o", color=COLOR_OLD, ecolor="black", elinewidth=1.8, capsize=5, markersize=8)
    ax4.axvline(0, color="gray", linestyle="--", lw=1.2)
    for idx_y, (_, r_lmm) in enumerate(lmm_age.iterrows()):
        p_txt = f"p = {r_lmm['p_value']:.4f}{sig_stars(r_lmm['p_value'])}"
        ax4.text(betas[idx_y], idx_y + 0.16, f"Beta = {betas[idx_y]:.3f} ({p_txt})", ha="center", fontsize=9, fontweight="bold")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(lmm_age["Outcome"].values, fontweight="bold")
    ax4.invert_yaxis()
    x_min_lim = min(-1.35, float(lmm_age["CI_Low"].min()) - 0.1)
    ax4.set_xlim(x_min_lim, 0.08)
    ax4.set_ylim(len(lmm_age) - 0.3, -0.5)
    ax4.set_xlabel("Mixed-Effects Age Coefficient (Old vs Young, 95% CI)")
    ax4.set_title(f"D | Hierarchical LMM Controlling for Mouse, Bout Order & Duration\n(n = {n_soc_trials} Social Bouts)", fontweight="bold")
    ax4.grid(True, axis="x", linestyle=":", alpha=0.4)

    fig8.suptitle("Figure 8 | Mixed-Effects Modeling, Jackknife Stability & Time-Window Controls", fontweight="bold")
    plt.tight_layout()
    save_figure(fig8, os.path.join(figs_dir, "Figure8_Mixed_Effects_Jackknife_and_PETH_Windows.png"), dpi=300)
    save_figure(fig8, os.path.join(figs_dir, "Figure8_Mixed_Effects_Jackknife_and_PETH_Windows.pdf"))
    plt.close(fig8)

    # Figure 9: Transients, auROC selectivity & cross-correlation
    fig9, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    y_basal = df_ext[df_ext["Age_Group"] == "Young"]["Transient_Rate_Basal_per_min"]
    o_basal = df_ext[df_ext["Age_Group"] == "Old"]["Transient_Rate_Basal_per_min"]
    y_soc_r = df_ext[df_ext["Age_Group"] == "Young"]["Transient_Rate_Social_per_min"].dropna()
    o_soc_r = df_ext[df_ext["Age_Group"] == "Old"]["Transient_Rate_Social_per_min"].dropna()
    _, p_bas = ttest_ind(y_basal, o_basal, equal_var=False)
    _, p_evk = ttest_ind(y_soc_r, o_soc_r, equal_var=False)

    ax1.boxplot([y_basal, o_basal, y_soc_r, o_soc_r], positions=[1, 1.7, 3, 3.7], widths=0.45, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="#f0f0f0", lw=1.1), medianprops=dict(color="black", lw=1.6))
    ax1.scatter(1 + rng.normal(0, 0.04, len(y_basal)), y_basal, color=COLOR_YOUNG, s=40, label="Young")
    ax1.scatter(1.7 + rng.normal(0, 0.04, len(o_basal)), o_basal, color=COLOR_OLD, s=40, label="Old")
    ax1.scatter(3 + rng.normal(0, 0.04, len(y_soc_r)), y_soc_r, color=COLOR_YOUNG, s=40)
    ax1.scatter(3.7 + rng.normal(0, 0.04, len(o_soc_r)), o_soc_r, color=COLOR_OLD, s=40)
    ax1.set_xticks([1, 1.7, 3, 3.7])
    ax1.set_xticklabels(["Young\nBasal", "Old\nBasal", "Young\nSocial", "Old\nSocial"], fontweight="bold")
    ax1.set_ylabel("Calcium Transients (Events / min)")
    ax1.set_title(f"A | Basal & Evoked Burst Rate (MAD Threshold)\n(Basal p = {p_bas:.4f}{sig_stars(p_bas)} | Social p = {p_evk:.4f})", fontweight="bold")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)

    y_roc = df_ext[df_ext["Age_Group"] == "Young"]["auROC_Peak"].dropna()
    o_roc = df_ext[df_ext["Age_Group"] == "Old"]["auROC_Peak"].dropna()
    t_roc, p_roc = ttest_ind(y_roc, o_roc, equal_var=False)
    _, p_roc_mw = mannwhitneyu(y_roc, o_roc, alternative="two-sided")

    ax2.boxplot([y_roc, o_roc], positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="#f0f0f0", lw=1.2), medianprops=dict(color="black", lw=1.8))
    ax2.scatter(1 + rng.normal(0, 0.04, len(y_roc)), y_roc, color=COLOR_YOUNG, s=50, alpha=0.85)
    ax2.scatter(2 + rng.normal(0, 0.04, len(o_roc)), o_roc, color=COLOR_OLD, s=50, alpha=0.85)
    ax2.axhline(0.5, color="gray", linestyle="--", lw=1.2, label="Chance (0.50)")
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Young", "Old"], fontweight="bold")
    ax2.set_ylabel("Single-Trial Social vs Object auROC")
    ax2.set_ylim(0.05, 1.05)
    ax2.set_title(f"B | Social vs Object Discrimination (auROC)\n(Welch t = {t_roc:.2f}, p = {p_roc:.4f}{sig_stars(p_roc)} | MW p = {p_roc_mw:.4f}{sig_stars(p_roc_mw)})", fontweight="bold")
    ax2.legend(loc="lower left", frameon=False)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

    y_xc_m, y_xc_s = np.mean(xcorr_waves["Young"], axis=0), sem(xcorr_waves["Young"], axis=0)
    o_xc_m, o_xc_s = np.mean(xcorr_waves["Old"], axis=0), sem(xcorr_waves["Old"], axis=0)
    _, p_xc0 = ttest_ind(
        df_ext[df_ext["Age_Group"] == "Young"]["CrossCorr_ZeroLag_r"],
        df_ext[df_ext["Age_Group"] == "Old"]["CrossCorr_ZeroLag_r"],
        equal_var=False,
    )
    ax3.plot(lags_s, y_xc_m, color=COLOR_YOUNG, lw=2.2, label="Young (n=8)")
    ax3.fill_between(lags_s, y_xc_m - y_xc_s, y_xc_m + y_xc_s, color=COLOR_YOUNG, alpha=0.2)
    ax3.plot(lags_s, o_xc_m, color=COLOR_OLD, lw=2.2, label="Old (n=10)")
    ax3.fill_between(lags_s, o_xc_m - o_xc_s, o_xc_m + o_xc_s, color=COLOR_OLD, alpha=0.2)
    ax3.axvline(0, color="black", linestyle=":", lw=1.2)
    ax3.set_xlabel("Inter-Hemispheric Time Lag (s)")
    ax3.set_ylabel("Cross-Correlation (r)")
    ax3.set_title(f"C | Bilateral Time-Lagged Cross-Correlation\n(Zero-Lag Synchrony {'Intact' if p_xc0 >= 0.05 else 'Shifted'}, p = {p_xc0:.3f})", fontweight="bold")
    ax3.legend(loc="lower center", frameon=False)
    ax3.grid(True, linestyle=":", alpha=0.4)

    fig9.suptitle("Figure 9 | Spontaneous Excitability, Social Discrimination (auROC) & Cross-Correlation", fontweight="bold")
    plt.tight_layout()
    save_figure(fig9, os.path.join(figs_dir, "Figure9_Spontaneous_Transients_and_Social_Selectivity.png"), dpi=300)
    save_figure(fig9, os.path.join(figs_dir, "Figure9_Spontaneous_Transients_and_Social_Selectivity.pdf"))
    plt.close(fig9)

    # Figure 10: Bout stratification, zone entry, offset & behavioral coupling
    fig10, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13.2, 9.6))

    y_sh = df_ext[df_ext["Age_Group"] == "Young"]["Short_Peak"].dropna()
    o_sh = df_ext[df_ext["Age_Group"] == "Old"]["Short_Peak"].dropna()
    y_lg = df_ext[df_ext["Age_Group"] == "Young"]["Long_Peak"].dropna()
    o_lg = df_ext[df_ext["Age_Group"] == "Old"]["Long_Peak"].dropna()
    _, p_sh = ttest_ind(y_sh, o_sh, equal_var=False)
    _, p_lg = ttest_ind(y_lg, o_lg, equal_var=False)

    ax1.boxplot([y_sh, o_sh, y_lg, o_lg], positions=[1, 1.7, 3, 3.7], widths=0.45, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="#f0f0f0", lw=1.1), medianprops=dict(color="black", lw=1.6))
    ax1.scatter(1 + rng.normal(0, 0.04, len(y_sh)), y_sh, color=COLOR_YOUNG, s=40, label="Young")
    ax1.scatter(1.7 + rng.normal(0, 0.04, len(o_sh)), o_sh, color=COLOR_OLD, s=40, label="Old")
    ax1.scatter(3 + rng.normal(0, 0.04, len(y_lg)), y_lg, color=COLOR_YOUNG, s=40)
    ax1.scatter(3.7 + rng.normal(0, 0.04, len(o_lg)), o_lg, color=COLOR_OLD, s=40)
    ax1.set_xticks([1, 1.7, 3, 3.7])
    ax1.set_xticklabels(["Young\nShort <3s", "Old\nShort <3s", "Young\nLong >=3s", "Old\nLong >=3s"], fontweight="bold")
    ax1.set_ylabel("Peak MeA Z-Score")
    top_sh = float(max(y_sh.max(), o_sh.max(), y_lg.max(), o_lg.max()))
    bot_sh = float(min(y_sh.min(), o_sh.min(), y_lg.min(), o_lg.min()))
    ax1.set_ylim(bot_sh - 0.25, top_sh + 0.78)
    ax1.set_title(f"A | Duration-Stratified Social Deficit\n(Short p = {p_sh:.4f}{sig_stars(p_sh)} | Sustained >=3s p = {p_lg:.4f}{sig_stars(p_lg)})", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.88, edgecolor="none")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)

    y_ze = df_ext[df_ext["Age_Group"] == "Young"]["Zone_Entry_Social_Delta_Z"].dropna()
    o_ze = df_ext[df_ext["Age_Group"] == "Old"]["Zone_Entry_Social_Delta_Z"].dropna()
    y_of = df_ext[df_ext["Age_Group"] == "Young"]["Offset_Disengagement_Delta_Z"].dropna()
    o_of = df_ext[df_ext["Age_Group"] == "Old"]["Offset_Disengagement_Delta_Z"].dropna()
    _, p_ze = ttest_ind(y_ze, o_ze, equal_var=False)
    _, p_of = ttest_ind(y_of, o_of, equal_var=False)

    ax2.boxplot([y_ze, o_ze, y_of, o_of], positions=[1, 1.7, 3, 3.7], widths=0.45, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="#f0f0f0", lw=1.1), medianprops=dict(color="black", lw=1.6))
    ax2.scatter(1 + rng.normal(0, 0.04, len(y_ze)), y_ze, color=COLOR_YOUNG, s=40)
    ax2.scatter(1.7 + rng.normal(0, 0.04, len(o_ze)), o_ze, color=COLOR_OLD, s=40)
    ax2.scatter(3 + rng.normal(0, 0.04, len(y_of)), y_of, color=COLOR_YOUNG, s=40)
    ax2.scatter(3.7 + rng.normal(0, 0.04, len(o_of)), o_of, color=COLOR_OLD, s=40)
    ax2.axhline(0, color="gray", linestyle="--", lw=0.9)
    ax2.set_xticks([1, 1.7, 3, 3.7])
    ax2.set_xticklabels(["Young\nZone Entry", "Old\nZone Entry", "Young\nBout Offset", "Old\nBout Offset"], fontweight="bold")
    ax2.set_ylabel("Delta Z-Score (Post - Pre)")
    ze_tag = sig_stars(p_ze) if p_ze < 0.05 else " n.s."
    ax2.set_title(f"B | Chamber Entry vs Disengagement Offset\n(Zone Entry p = {p_ze:.3f}{ze_tag} | Offset p = {p_of:.4f}{sig_stars(p_of)})", fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

    y_hab = df_ext[df_ext["Age_Group"] == "Young"][["Early_Peak", "Late_Peak"]].dropna()
    o_hab = df_ext[df_ext["Age_Group"] == "Old"][["Early_Peak", "Late_Peak"]].dropna()
    _, p_hy = ttest_rel(y_hab["Early_Peak"], y_hab["Late_Peak"])
    _, p_ho = ttest_rel(o_hab["Early_Peak"], o_hab["Late_Peak"])
    hy_tag = sig_stars(p_hy) if p_hy < 0.05 else " n.s."
    ho_tag = sig_stars(p_ho) if p_ho < 0.05 else (" trend" if p_ho < 0.10 else " n.s.")

    ax3.errorbar([1, 2], [y_hab["Early_Peak"].mean(), y_hab["Late_Peak"].mean()],
                 yerr=[sem(y_hab["Early_Peak"]), sem(y_hab["Late_Peak"])],
                 color=COLOR_YOUNG, marker="o", markersize=8, lw=2.5, capsize=5, label=f"Young (p = {p_hy:.3f}{hy_tag})")
    ax3.errorbar([1, 2], [o_hab["Early_Peak"].mean(), o_hab["Late_Peak"].mean()],
                 yerr=[sem(o_hab["Early_Peak"]), sem(o_hab["Late_Peak"])],
                 color=COLOR_OLD, marker="s", markersize=8, lw=2.5, capsize=5, label=f"Old (p = {p_ho:.3f}{ho_tag})")
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(["Early Bouts (1-5)", "Late Bouts (>5)"], fontweight="bold")
    ax3.set_xlim(0.5, 2.5)
    top_hab = max(y_hab["Early_Peak"].mean() + sem(y_hab["Early_Peak"]), y_hab["Late_Peak"].mean() + sem(y_hab["Late_Peak"]))
    bot_hab = min(o_hab["Early_Peak"].mean() - sem(o_hab["Early_Peak"]), o_hab["Late_Peak"].mean() - sem(o_hab["Late_Peak"]))
    ax3.set_ylim(bot_hab - 0.18, top_hab + 0.35)
    ax3.set_ylabel("Peak MeA Z-Score")
    ax3.set_title("C | Within-Session Adaptation Across Repeated Bouts", fontweight="bold")
    ax3.legend(loc="upper right", frameon=False)
    ax3.grid(True, axis="y", linestyle=":", alpha=0.4)

    df_y_bb = df_sum[df_sum["Age_Group"] == "Young"]
    df_o_bb = df_sum[df_sum["Age_Group"] == "Old"]
    ry_bb, py_bb = pearsonr(df_y_bb["Behavioral_Social_Preference_Index"], df_y_bb["Social_Mean_Z_0_2s_Bilateral"])
    ro_bb, po_bb = pearsonr(df_o_bb["Behavioral_Social_Preference_Index"], df_o_bb["Social_Mean_Z_0_2s_Bilateral"])

    ax4.scatter(df_y_bb["Behavioral_Social_Preference_Index"], df_y_bb["Social_Mean_Z_0_2s_Bilateral"], color=COLOR_YOUNG, s=55, alpha=0.85,
                label=f"Young (r = {ry_bb:+.2f}, p = {py_bb:.3f})")
    ax4.scatter(df_o_bb["Behavioral_Social_Preference_Index"], df_o_bb["Social_Mean_Z_0_2s_Bilateral"], color=COLOR_OLD, s=55, alpha=0.85,
                label=f"Old (r = {ro_bb:+.2f}, p = {po_bb:.3f})")
    my_b, by_b = np.polyfit(df_y_bb["Behavioral_Social_Preference_Index"], df_y_bb["Social_Mean_Z_0_2s_Bilateral"], 1)
    mo_b, bo_b = np.polyfit(df_o_bb["Behavioral_Social_Preference_Index"], df_o_bb["Social_Mean_Z_0_2s_Bilateral"], 1)
    xs_y = np.linspace(df_y_bb["Behavioral_Social_Preference_Index"].min(), df_y_bb["Behavioral_Social_Preference_Index"].max(), 30)
    xs_o = np.linspace(df_o_bb["Behavioral_Social_Preference_Index"].min(), df_o_bb["Behavioral_Social_Preference_Index"].max(), 30)
    ax4.plot(xs_y, my_b * xs_y + by_b, color=COLOR_YOUNG, lw=2.0)
    ax4.plot(xs_o, mo_b * xs_o + bo_b, color=COLOR_OLD, lw=2.0, linestyle="--")
    ax4.axhline(0, color="gray", linestyle=":", lw=0.9)
    ax4.axvline(0, color="gray", linestyle=":", lw=0.9)
    max_m02 = float(df_sum["Social_Mean_Z_0_2s_Bilateral"].max())
    min_m02 = float(df_sum["Social_Mean_Z_0_2s_Bilateral"].min())
    ax4.set_ylim(min_m02 - 0.15, max_m02 + 0.35)
    ax4.set_xlabel("Behavioral Social Preference Index (SPI)")
    ax4.set_ylabel("Mean MeA Z-Score (0 to 2 s)")
    ax4.set_title("D | Behavioral Preference vs Early MeA Activation", fontweight="bold")
    ax4.legend(loc="upper right", frameon=False)
    ax4.grid(True, linestyle=":", alpha=0.4)

    fig10.suptitle("Figure 10 | Bout Stratification, Disengagement Dynamics & Behavioral Coupling", fontweight="bold")
    plt.tight_layout()
    save_figure(fig10, os.path.join(figs_dir, "Figure10_Bout_Stratification_Offset_ZoneEntry_and_Behavior.png"), dpi=300)
    save_figure(fig10, os.path.join(figs_dir, "Figure10_Bout_Stratification_Offset_ZoneEntry_and_Behavior.pdf"))
    plt.close(fig10)

    # Extended statistical report
    y_sub = df_sum[df_sum["Age_Group"] == "Young"]
    o_sub = df_sum[df_sum["Age_Group"] == "Old"]
    t_yr, p_yr = ttest_rel(y_sub["Social_Peak_Right"], y_sub["NonSocial_Peak_Right"])
    t_or, p_or = ttest_rel(o_sub["Social_Peak_Right"], o_sub["NonSocial_Peak_Right"])

    y_tot = df_ext[df_ext["Age_Group"] == "Young"]["Transient_Rate_Total_per_min"]
    o_tot = df_ext[df_ext["Age_Group"] == "Old"]["Transient_Rate_Total_per_min"]
    _, p_tot = ttest_ind(y_tot, o_tot, equal_var=False)
    y_amp = df_ext[df_ext["Age_Group"] == "Young"]["Transient_Peak_Amp_Mean"]
    o_amp = df_ext[df_ext["Age_Group"] == "Old"]["Transient_Peak_Amp_Mean"]
    _, p_amp = ttest_ind(y_amp, o_amp, equal_var=False)

    rep_fp = os.path.join(tables_dir, "extended_statistical_report.txt")
    with open(rep_fp, "w", encoding="utf-8") as rf:
        rf.write("Extended Statistical & Quality Control Report\n")
        rf.write("---------------------------------------------\n\n")

        rf.write("1. OPTICAL QUALITY & ARTIFACT CONTROLS:\n")
        rf.write(f"   - Session SNR:           Young {y_snr.mean():.2f} +/- {sem(y_snr):.2f} vs Old {o_snr.mean():.2f} +/- {sem(o_snr):.2f} (p = {p_snr:.4f})\n")
        rf.write(f"   - 410 nm PETH (0-2 s):   Young {df_ext[df_ext['Age_Group']=='Young']['Iso_410_0_2s_Mean_Z'].mean():.4f} vs Old {df_ext[df_ext['Age_Group']=='Old']['Iso_410_0_2s_Mean_Z'].mean():.4f}\n")
        rf.write(f"   - Speed-Signal r (Raw):  Young {y_raw.mean():.4f} | Old {o_raw.mean():.4f}\n")
        rf.write(f"   - Speed-Signal r (Corr): Young {y_cor.mean():.4f} | Old {o_cor.mean():.4f} (Paired drop p = {p_scrub:.4f})\n")
        rf.write(f"   - Object Peak Control:   Young {y_obj.mean():.3f} +/- {sem(y_obj):.3f} vs Old {o_obj.mean():.3f} +/- {sem(o_obj):.3f} (p = {p_obj:.4f})\n\n")

        rf.write("2. NON-PARAMETRIC, HEDGES' g, & 18-FOLD LEAVE-ONE-MOUSE-OUT (JACKKNIFE):\n")
        for _, r in df_nonparam_lomo.iterrows():
            rf.write(
                f"   - {r['Metric']}: Mann-Whitney p = {r['MannWhitney_p']:.4f} | Hedges' g = {r['Hedges_g']:.3f} | "
                f"LOMO p in [{r['LOMO_p_Min']:.4f}, {r['LOMO_p_Max']:.4f}] ({r['LOMO_Significant_Count']}/{r['LOMO_Total']} drops p<0.05)\n"
            )
        rf.write("\n")

        rf.write(f"3. HIERARCHICAL LINEAR MIXED-EFFECTS MODEL ({n_soc_trials} SOCIAL TRIALS, RANDOM MOUSE INTERCEPT):\n")
        for _, r in df_lmm[df_lmm["Predictor"] == "Is_Old"].iterrows():
            rf.write(f"   - {r['Outcome']}: Age Beta = {r['Beta']:.4f}, SE = {r['SE']:.4f}, z = {r['z_stat']:.3f}, p = {r['p_value']:.4f}\n")
        rf.write(f"   - Young AUC vs Duration:             r = {dur_checks['Young_r_Dur_AUC']}, p = {dur_checks['Young_p_Dur_AUC']}\n")
        rf.write(f"   - Young Whole-Bout Mean vs Duration: r = {dur_checks['Young_r_Dur_WB_MeanZ']}, p = {dur_checks['Young_p_Dur_WB_MeanZ']}\n")
        rf.write(f"   - PETH Consecutive Divergence (>=0.2 s, p<0.05): {peth_div['sig_intervals_s']}\n\n")

        rf.write("4. SPONTANEOUS vs EVOKED CALCIUM TRANSIENT RATES (MAD THRESHOLD):\n")
        rf.write(f"   - Total Session Rate (/min):  Young {y_tot.mean():.2f} +/- {sem(y_tot):.2f} vs Old {o_tot.mean():.2f} +/- {sem(o_tot):.2f} (p = {p_tot:.4f})\n")
        rf.write(f"   - Basal Outside-Bout (/min):  Young {y_basal.mean():.2f} +/- {sem(y_basal):.2f} vs Old {o_basal.mean():.2f} +/- {sem(o_basal):.2f} (p = {p_bas:.4f})\n")
        rf.write(f"   - In-Social Evoked (/min):    Young {y_soc_r.mean():.2f} +/- {sem(y_soc_r):.2f} vs Old {o_soc_r.mean():.2f} +/- {sem(o_soc_r):.2f} (p = {p_evk:.4f})\n")
        rf.write(f"   - Peak Transient Amplitude:   Young {y_amp.mean():.2f} vs Old {o_amp.mean():.2f} (p = {p_amp:.4f})\n\n")

        rf.write("5. STIMULUS SELECTIVITY (auROC), DURATION STRATIFICATION & BEHAVIORAL DYNAMICS:\n")
        rf.write(f"   - Right MeA Paired Social vs Object: Young t = {t_yr:.3f}, p = {p_yr:.4f} | Old t = {t_or:.3f}, p = {p_or:.4f}\n")
        rf.write(f"   - Single-Trial auROC (Social vs Obj): Young {y_roc.mean():.3f} +/- {sem(y_roc):.3f} vs Old {o_roc.mean():.3f} +/- {sem(o_roc):.3f} (Welch p = {p_roc:.4f}, MW p = {p_roc_mw:.4f})\n")
        rf.write(f"   - Short (<3s) Bouts Peak Z:          Young {y_sh.mean():.3f} vs Old {o_sh.mean():.3f} (p = {p_sh:.4f})\n")
        rf.write(f"   - Long (>=3s) Bouts Peak Z:          Young {y_lg.mean():.3f} vs Old {o_lg.mean():.3f} (p = {p_lg:.4f})\n")
        rf.write(f"   - Chamber Entry (Novel Mouse 0-2s):  Young {y_ze.mean():.3f} vs Old {o_ze.mean():.3f} (p = {p_ze:.4f})\n")
        rf.write(f"   - Bout Offset Disengagement Delta:   Young {y_of.mean():.3f} vs Old {o_of.mean():.3f} (p = {p_of:.4f})\n")
        rf.write(f"   - Early vs Late Habituation (Peak):  Young p = {p_hy:.4f} | Old p = {p_ho:.4f}\n")
        rf.write(f"   - Brain-Behavior (0-2s vs SPI):      Young r = {ry_bb:.3f} (p = {py_bb:.4f}) | Old r = {ro_bb:.3f} (p = {po_bb:.4f})\n")

    print("Extended analysis complete. Saved tables and Figures 7 to 10.")


def main():
    parser = argparse.ArgumentParser(description="Extended Photometry & Behavioral Analysis Pipeline")
    parser.add_argument("--raw_zip", type=str, default=None, help="Path to Data_2.zip or extracted Data_2 folder")
    parser.add_argument("--tracking_dir", type=str, default=None, help="Path to results_social tracking directory")
    parser.add_argument("--results_dir", type=str, default=None, help="Path to results directory")
    args = parser.parse_args()

    run_extended(raw_zip=args.raw_zip, tracking_dir=args.tracking_dir, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
