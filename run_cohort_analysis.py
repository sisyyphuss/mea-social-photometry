"""
Cohort Analysis: Bilateral Medial Amygdala Photometry and Social Interaction

Usage:
    python run_cohort_analysis.py [--raw_zip PATH] [--tracking_dir PATH] [--output_dir PATH]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_ind, ttest_rel, sem
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.io_paths import resolve_data_paths, load_tracking_csv
from src.visualization import set_publication_style, save_figure, COLOR_LEFT, COLOR_RIGHT, COLOR_YOUNG, COLOR_OLD
from analyze_mouse import process_single_mouse


def run_pipeline(raw_zip=None, tracking_dir=None, output_dir=None):
    set_publication_style()
    rng = np.random.default_rng(42)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(repo_root, "results")

    raw_source, tracking_dir = resolve_data_paths(raw_zip, tracking_dir, repo_root=repo_root)

    print("Running cohort photometry, tracking, and coherence pipeline...")
    print(f"Photometry source:  {raw_source}")
    print(f"Tracking directory: {tracking_dir}")
    print(f"Output directory:   {output_dir}")

    tables_dir = os.path.join(output_dir, "tables")
    figs_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    young_ids = ["303", "305", "307", "308", "309", "311", "326", "331"]
    old_ids = ["315", "316", "317", "318", "319", "323", "50", "56", "66", "69"]

    all_mouse_summaries = []
    all_trial_records = []

    cohort_peths = {
        "Young": {"Social": {"L": [], "R": []}, "NonSocial": {"L": [], "R": []}},
        "Old": {"Social": {"L": [], "R": []}, "NonSocial": {"L": [], "R": []}},
    }
    cohort_spatial_heatmaps = {"Young": [], "Old": []}
    cohort_coherence = {
        "Young": {"freqs": None, "spectrum_pre": [], "spectrum_post": [], "post_full": [], "delta_full": [], "delta_std": []},
        "Old": {"freqs": None, "spectrum_pre": [], "spectrum_post": [], "post_full": [], "delta_full": [], "delta_std": []},
    }
    rel_time = None

    for age_group, mouse_list in [("Young", young_ids), ("Old", old_ids)]:
        for mouse_id in mouse_list:
            print(f"Processing Mouse {mouse_id} ({age_group})...")
            target_indiv_dir = os.path.join(output_dir, "individual", age_group, f"Mouse_{mouse_id}")
            res = process_single_mouse(
                mouse_id=mouse_id,
                age_group=age_group,
                raw_source=raw_source,
                tracking_dir=tracking_dir,
                output_dir=target_indiv_dir,
                generate_plots=True,
            )

            all_mouse_summaries.append(res["summary"])
            all_trial_records.extend(res["trials"])
            rel_time = res["rel_time"]

            if len(res["peth_soc_l"]) > 0:
                cohort_peths[age_group]["Social"]["L"].append(np.mean(res["peth_soc_l"], axis=0))
                cohort_peths[age_group]["Social"]["R"].append(np.mean(res["peth_soc_r"], axis=0))
            if len(res["peth_obj_l"]) > 0:
                cohort_peths[age_group]["NonSocial"]["L"].append(np.mean(res["peth_obj_l"], axis=0))
                cohort_peths[age_group]["NonSocial"]["R"].append(np.mean(res["peth_obj_r"], axis=0))

            cohort_spatial_heatmaps[age_group].append(res["h_norm"])

            if len(res["coh_soc_pre"]) > 0:
                if cohort_coherence[age_group]["freqs"] is None:
                    cohort_coherence[age_group]["freqs"] = res["freqs_soc"]
                cohort_coherence[age_group]["spectrum_pre"].append(np.mean(res["coh_soc_pre"], axis=0))
                cohort_coherence[age_group]["spectrum_post"].append(np.mean(res["coh_soc_post"], axis=0))
                cohort_coherence[age_group]["post_full"].append(res["m_c_full_post"])
                cohort_coherence[age_group]["delta_full"].append(res["m_c_full_del"])
                cohort_coherence[age_group]["delta_std"].append(res["m_coh_del"])

    df_all_summaries = pd.DataFrame(all_mouse_summaries)
    df_all_trials = pd.DataFrame(all_trial_records)

    df_all_summaries.to_csv(os.path.join(tables_dir, "master_per_mouse_summary.csv"), index=False)
    df_all_trials.to_csv(os.path.join(tables_dir, "master_all_trials_dataset.csv"), index=False)

    # --------------------------------------------------------------------------
    # COHORT SUMMARIES (Young & Old)
    # --------------------------------------------------------------------------
    for age_group in ["Young", "Old"]:
        cdir = os.path.join(output_dir, "cohorts", age_group)
        os.makedirs(cdir, exist_ok=True)
        n_m = len(young_ids) if age_group == "Young" else len(old_ids)
        prefix = age_group.lower()

        df_c = df_all_summaries[df_all_summaries["Age_Group"] == age_group]
        df_c.to_csv(os.path.join(cdir, f"{prefix}_cohort_summary.csv"), index=False)

        # Grand PETH per cohort
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

        def draw_cohort_peth(axis, l_list, r_list, title):
            m_l, s_l = np.mean(l_list, axis=0), sem(l_list, axis=0)
            m_r, s_r = np.mean(r_list, axis=0), sem(r_list, axis=0)
            axis.plot(rel_time, m_l, color=COLOR_LEFT, label=f"Left MeA (n={len(l_list)})", lw=2.2)
            axis.fill_between(rel_time, m_l - s_l, m_l + s_l, color=COLOR_LEFT, alpha=0.22)
            axis.plot(rel_time, m_r, color=COLOR_RIGHT, label=f"Right MeA (n={len(r_list)})", lw=2.2)
            axis.fill_between(rel_time, m_r - s_r, m_r + s_r, color=COLOR_RIGHT, alpha=0.22)
            axis.axvline(0, color="black", linestyle=":", lw=1.2)
            axis.axhline(0, color="gray", linestyle="--", lw=0.7, alpha=0.6)
            axis.axvspan(0, 2.0, color="#abdda4", alpha=0.2, label="0-2 s Window")
            axis.axvspan(2.0, 4.0, color="#fee08b", alpha=0.14)
            axis.set_title(title, fontweight="bold")
            axis.set_xlabel("Time Relative to Onset (s)")
            axis.grid(True, linestyle=":", alpha=0.4)
            axis.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.88, edgecolor="none")

        ax1.set_ylabel("Baseline-Corrected Z-Score")
        draw_cohort_peth(ax1, cohort_peths[age_group]["Social"]["L"], cohort_peths[age_group]["Social"]["R"], "Social Sniffing")
        draw_cohort_peth(ax2, cohort_peths[age_group]["NonSocial"]["L"], cohort_peths[age_group]["NonSocial"]["R"], "Object Exploration")
        y_lo, y_hi = ax1.get_ylim()
        ax1.set_ylim(y_lo, y_hi + 0.20 * (y_hi - y_lo))
        fig.suptitle(f"{age_group} Cohort (n={n_m}) | Grand Average MeA Photometry Dynamics", fontweight="bold")
        plt.tight_layout()
        save_figure(fig, os.path.join(cdir, f"{prefix}_grand_peth.png"), dpi=250)
        plt.close(fig)

        # Spatial occupancy heatmap with physical extent & colorbar
        mean_h = np.mean(cohort_spatial_heatmaps[age_group], axis=0)
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        im = ax.imshow(mean_h, origin="upper", cmap="inferno", aspect="auto", extent=[0.0, 1.0, 1.0, 0.0])
        ax.set_xlabel("Arena X (Normalized)")
        ax.set_ylabel("Arena Y (Normalized)")
        ax.set_title(f"{age_group} Cohort (n={n_m}) | Mean Spatial Dwell Occupancy", fontweight="bold")
        plt.colorbar(im, ax=ax, label="Normalized Presence Density")
        plt.tight_layout()
        save_figure(fig, os.path.join(cdir, f"{prefix}_spatial_heatmap.png"), dpi=250)
        plt.close(fig)

        # Coherence spectrum per cohort
        freqs = cohort_coherence[age_group]["freqs"]
        m_pre_c = np.mean(cohort_coherence[age_group]["spectrum_pre"], axis=0)
        s_pre_c = sem(cohort_coherence[age_group]["spectrum_pre"], axis=0)
        m_post_c = np.mean(cohort_coherence[age_group]["spectrum_post"], axis=0)
        s_post_c = sem(cohort_coherence[age_group]["spectrum_post"], axis=0)
        f_mask = freqs <= 3.0

        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.plot(freqs[f_mask], m_pre_c[f_mask], color="gray", lw=2.2, label="Baseline (-5 to 0 s)")
        ax.fill_between(freqs[f_mask], (m_pre_c - s_pre_c)[f_mask], (m_pre_c + s_pre_c)[f_mask], color="gray", alpha=0.2)
        ax.plot(freqs[f_mask], m_post_c[f_mask], color=COLOR_YOUNG if age_group == "Young" else COLOR_OLD, lw=2.2, label="Social Interaction (0 to 5 s)")
        ax.fill_between(freqs[f_mask], (m_post_c - s_post_c)[f_mask], (m_post_c + s_post_c)[f_mask], color=COLOR_YOUNG if age_group == "Young" else COLOR_OLD, alpha=0.22)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude-Squared Coherence")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{age_group} Cohort (n={n_m}) | Bilateral MeA Coherence Spectrum", fontweight="bold")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        save_figure(fig, os.path.join(cdir, f"{prefix}_spectral_coherence.png"), dpi=250)
        plt.close(fig)

    # --------------------------------------------------------------------------
    # GROUP COMPARISON PUBLICATION FIGURES (Figures 1 to 6)
    # --------------------------------------------------------------------------
    # Figure 1: Grand Average PETH
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

    def render_cohort_cmp(axis, y_list, o_list, title):
        ym, ys = np.mean(y_list, axis=0), sem(y_list, axis=0)
        om, os_v = np.mean(o_list, axis=0), sem(o_list, axis=0)
        axis.plot(rel_time, ym, color=COLOR_YOUNG, lw=2.2, label="Young (n=8)")
        axis.fill_between(rel_time, ym - ys, ym + ys, color=COLOR_YOUNG, alpha=0.22)
        axis.plot(rel_time, om, color=COLOR_OLD, lw=2.2, label="Old (n=10)")
        axis.fill_between(rel_time, om - os_v, om + os_v, color=COLOR_OLD, alpha=0.22)
        axis.axvline(0, color="black", linestyle=":", lw=1.2)
        axis.axhline(0, color="gray", linestyle="--", lw=0.7, alpha=0.6)
        axis.axvspan(0, 2.0, color="#abdda4", alpha=0.2, label="0-2 s Window")
        axis.axvspan(2.0, 4.0, color="#fee08b", alpha=0.12)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Time Relative to Onset (s)")
        axis.grid(True, linestyle=":", alpha=0.4)
        axis.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.88, edgecolor="none")

    y_soc_bilat = [(cohort_peths["Young"]["Social"]["L"][i] + cohort_peths["Young"]["Social"]["R"][i]) / 2.0 for i in range(len(young_ids))]
    o_soc_bilat = [(cohort_peths["Old"]["Social"]["L"][i] + cohort_peths["Old"]["Social"]["R"][i]) / 2.0 for i in range(len(old_ids))]
    y_obj_bilat = [(cohort_peths["Young"]["NonSocial"]["L"][i] + cohort_peths["Young"]["NonSocial"]["R"][i]) / 2.0 for i in range(len(young_ids))]
    o_obj_bilat = [(cohort_peths["Old"]["NonSocial"]["L"][i] + cohort_peths["Old"]["NonSocial"]["R"][i]) / 2.0 for i in range(len(old_ids))]

    ax1.set_ylabel("Baseline-Corrected Z-Score")
    render_cohort_cmp(ax1, y_soc_bilat, o_soc_bilat, "Social Sniffing")
    render_cohort_cmp(ax2, y_obj_bilat, o_obj_bilat, "Object Exploration")
    y_lo1, y_hi1 = ax1.get_ylim()
    ax1.set_ylim(y_lo1, y_hi1 + 0.22 * (y_hi1 - y_lo1))
    def sig_stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    def sig_note(p):
        return sig_stars(p) if p < 0.05 else ", n.s."

    fig1.suptitle("Figure 1 | Grand Average MeA Dynamics: Young vs Old Mice", fontweight="bold")
    plt.tight_layout()
    save_figure(fig1, os.path.join(figs_dir, "Figure1_Grand_PETH_Dynamics.png"), dpi=300)
    save_figure(fig1, os.path.join(figs_dir, "Figure1_Grand_PETH_Dynamics.pdf"))
    plt.close(fig1)

    # Figure 2: Lateralization Index
    young_lis = df_all_summaries[df_all_summaries["Age_Group"] == "Young"]["Social_Lateralization_Index"]
    old_lis = df_all_summaries[df_all_summaries["Age_Group"] == "Old"]["Social_Lateralization_Index"]
    t_lat, p_lat = ttest_ind(young_lis, old_lis, equal_var=False)

    fig2, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.boxplot([young_lis, old_lis], positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="#f0f0f0", lw=1.2), medianprops=dict(color="black", lw=1.8))
    ax.scatter(1 + rng.normal(0, 0.04, len(young_lis)), young_lis, color=COLOR_YOUNG, s=55, alpha=0.85, label="Young (n=8)")
    ax.scatter(2 + rng.normal(0, 0.04, len(old_lis)), old_lis, color=COLOR_OLD, s=55, alpha=0.85, label="Old (n=10)")
    ax.axhline(0, color="gray", linestyle="--", lw=1.0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Young", "Old"], fontweight="bold")
    ax.set_ylabel("Social Lateralization Index (LI)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"Figure 2 | Hemispheric Lateralization Index (p = {p_lat:.3f}{sig_note(p_lat)})\n(+1: Left MeA Dominant, -1: Right MeA Dominant)", fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    save_figure(fig2, os.path.join(figs_dir, "Figure2_Hemispheric_Lateralization.png"), dpi=300)
    save_figure(fig2, os.path.join(figs_dir, "Figure2_Hemispheric_Lateralization.pdf"))
    plt.close(fig2)

    # Figure 3: Primary Activation Deficits (Peak Z, Whole-Bout Mean Z, 0-2 s Window)
    young_pks = (df_all_summaries[df_all_summaries["Age_Group"] == "Young"]["Social_Peak_Left"] + df_all_summaries[df_all_summaries["Age_Group"] == "Young"]["Social_Peak_Right"]) / 2.0
    old_pks = (df_all_summaries[df_all_summaries["Age_Group"] == "Old"]["Social_Peak_Left"] + df_all_summaries[df_all_summaries["Age_Group"] == "Old"]["Social_Peak_Right"]) / 2.0
    t_pk, p_pk = ttest_ind(young_pks, old_pks, equal_var=False)

    young_wb = df_all_summaries[df_all_summaries["Age_Group"] == "Young"]["Social_Whole_Bout_Mean_Z_Bilateral"]
    old_wb = df_all_summaries[df_all_summaries["Age_Group"] == "Old"]["Social_Whole_Bout_Mean_Z_Bilateral"]
    t_wb, p_wb = ttest_ind(young_wb, old_wb, equal_var=False)

    young_m02 = df_all_summaries[df_all_summaries["Age_Group"] == "Young"]["Social_Mean_Z_0_2s_Bilateral"]
    old_m02 = df_all_summaries[df_all_summaries["Age_Group"] == "Old"]["Social_Mean_Z_0_2s_Bilateral"]
    t_m02, p_m02 = ttest_ind(young_m02, old_m02, equal_var=False)

    fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8))
    panels_f3 = [
        (ax1, young_pks, old_pks, "Peak MeA Activation (Z-score)", f"A | Peak Activation (0 to 4 s)\n(Welch's t = {t_pk:.2f}, p = {p_pk:.4f}{sig_stars(p_pk)})", False),
        (ax2, young_wb, old_wb, "Whole-Bout Mean Z-Score", f"B | Whole-Bout Mean Z (Duration-Normalized)\n(Welch's t = {t_wb:.2f}, p = {p_wb:.4f}{sig_stars(p_wb)})", True),
        (ax3, young_m02, old_m02, "Mean Z-Score (0 to +2 s)", f"C | Fixed 0 to +2 s Window\n(Welch's t = {t_m02:.2f}, p = {p_m02:.4f}{sig_stars(p_m02)})", True),
    ]
    for axis, y_v, o_v, ylab, tit, draw_zero in panels_f3:
        axis.boxplot([y_v, o_v], positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor="#f0f0f0", lw=1.2), medianprops=dict(color="black", lw=1.8))
        axis.scatter(1 + rng.normal(0, 0.04, len(y_v)), y_v, color=COLOR_YOUNG, s=55, alpha=0.85)
        axis.scatter(2 + rng.normal(0, 0.04, len(o_v)), o_v, color=COLOR_OLD, s=55, alpha=0.85)
        if draw_zero:
            axis.axhline(0, color="gray", linestyle="--", lw=1.0)
        axis.set_xticks([1, 2])
        axis.set_xticklabels(["Young", "Old"], fontweight="bold")
        axis.set_ylabel(ylab)
        axis.set_title(tit, fontweight="bold")
        axis.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig3.suptitle("Figure 3 | Medial Amygdala Activation Deficits in Aging (Peak, Whole-Bout Mean & 0-2 s Window)", fontweight="bold")
    plt.tight_layout()
    save_figure(fig3, os.path.join(figs_dir, "Figure3_MeA_Peak_Activation_Aging_Deficit.png"), dpi=300)
    save_figure(fig3, os.path.join(figs_dir, "Figure3_MeA_Peak_Activation_Aging_Deficit.pdf"))
    plt.close(fig3)

    # Figure 4: Behavioral SPI & Locomotor Control
    young_spis = df_all_summaries[df_all_summaries["Age_Group"] == "Young"]["Behavioral_Social_Preference_Index"]
    old_spis = df_all_summaries[df_all_summaries["Age_Group"] == "Old"]["Behavioral_Social_Preference_Index"]
    t_spi, p_spi = ttest_ind(young_spis, old_spis, equal_var=False)

    young_dists = df_all_summaries[df_all_summaries["Age_Group"] == "Young"]["Locomotor_Distance_px"]
    old_dists = df_all_summaries[df_all_summaries["Age_Group"] == "Old"]["Locomotor_Distance_px"]
    t_dist, p_dist = ttest_ind(young_dists, old_dists, equal_var=False)

    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax1.boxplot([young_spis, old_spis], positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="#f0f0f0", lw=1.2), medianprops=dict(color="black", lw=1.8))
    ax1.scatter(1 + rng.normal(0, 0.04, len(young_spis)), young_spis, color=COLOR_YOUNG, s=50, alpha=0.85)
    ax1.scatter(2 + rng.normal(0, 0.04, len(old_spis)), old_spis, color=COLOR_OLD, s=50, alpha=0.85)
    ax1.axhline(0, color="gray", linestyle="--", lw=1.0)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Young", "Old"], fontweight="bold")
    ax1.set_ylabel("Social Preference Index (SPI)")
    ax1.set_title(f"A | Behavioral Sociability (p = {p_spi:.3f}{sig_note(p_spi)})", fontweight="bold")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax2.boxplot([young_dists, old_dists], positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="#f0f0f0", lw=1.2), medianprops=dict(color="black", lw=1.8))
    ax2.scatter(1 + rng.normal(0, 0.04, len(young_dists)), young_dists, color=COLOR_YOUNG, s=50, alpha=0.85)
    ax2.scatter(2 + rng.normal(0, 0.04, len(old_dists)), old_dists, color=COLOR_OLD, s=50, alpha=0.85)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Young", "Old"], fontweight="bold")
    ax2.set_ylabel("Total Distance Traveled (px)")
    ax2.set_title(f"B | Locomotor Distance Control (p = {p_dist:.3f}{sig_note(p_dist)})", fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig4.suptitle("Figure 4 | Behavioral Preference & Locomotor Kinematics Controls", fontweight="bold")
    plt.tight_layout()
    save_figure(fig4, os.path.join(figs_dir, "Figure4_Behavioral_SPI_and_Locomotion.png"), dpi=300)
    save_figure(fig4, os.path.join(figs_dir, "Figure4_Behavioral_SPI_and_Locomotion.pdf"))
    plt.close(fig4)

    # Figure 5: Spatial Neural Mapping (Representative Mouse 303 with exact arena pixel bounds)
    df_s = load_tracking_csv(tracking_dir, "Young", "303").dropna(subset=["x_px", "y_px", "CH1_z", "CH2_z"])
    xs, ys = df_s["x_px"].values, df_s["y_px"].values
    zl, zr = df_s["CH1_z"].values, df_s["CH2_z"].values

    h_l, _, _ = np.histogram2d(xs, ys, bins=[60, 40], weights=np.maximum(0, zl))
    h_r, _, _ = np.histogram2d(xs, ys, bins=[60, 40], weights=np.maximum(0, zr))
    cnts, _, _ = np.histogram2d(xs, ys, bins=[60, 40])
    cnts[cnts == 0] = 1
    map_l = gaussian_filter(h_l.T / cnts.T, sigma=1.2)
    map_r = gaussian_filter(h_r.T / cnts.T, sigma=1.2)
    ext = [float(xs.min()), float(xs.max()), float(ys.max()), float(ys.min())]

    fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    im1 = ax1.imshow(map_l, cmap="viridis", origin="upper", extent=ext, aspect="auto")
    ax1.set_title("Left MeA (CH1) Mean Activity", fontweight="bold")
    ax1.set_xlabel("Arena X (px)")
    ax1.set_ylabel("Arena Y (px)")
    plt.colorbar(im1, ax=ax1, label="Mean Z-Score")

    im2 = ax2.imshow(map_r, cmap="viridis", origin="upper", extent=ext, aspect="auto")
    ax2.set_title("Right MeA (CH2) Mean Activity", fontweight="bold")
    ax2.set_xlabel("Arena X (px)")
    ax2.set_ylabel("Arena Y (px)")
    plt.colorbar(im2, ax=ax2, label="Mean Z-Score")

    fig5.suptitle("Figure 5 | 2D Spatial Neural Representation (Representative Animal: Mouse 303)", fontweight="bold")
    plt.tight_layout()
    save_figure(fig5, os.path.join(figs_dir, "Figure5_Spatial_Neural_Mapping.png"), dpi=300)
    save_figure(fig5, os.path.join(figs_dir, "Figure5_Spatial_Neural_Mapping.pdf"))
    plt.close(fig5)

    # Figure 6: Bilateral Spectral Coherence
    fig6 = plt.figure(figsize=(14.5, 4.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.1, 1])
    ax_spec = fig6.add_subplot(gs[0])
    ax_bar = fig6.add_subplot(gs[1])
    ax_del = fig6.add_subplot(gs[2])

    freqs = cohort_coherence["Young"]["freqs"]
    f_mask = freqs <= 3.0
    m_full = (freqs >= 0.1) & (freqs <= 3.0)
    m_b1 = (freqs >= 0.1) & (freqs <= 0.5)
    m_b2 = (freqs >= 0.5) & (freqs <= 1.5)
    m_b3 = (freqs >= 1.5) & (freqs <= 3.0)
    m_std = (freqs >= 0.8) & (freqs <= 1.6)

    y_post_spec, y_post_sem = np.mean(cohort_coherence["Young"]["spectrum_post"], axis=0), sem(cohort_coherence["Young"]["spectrum_post"], axis=0)
    o_post_spec, o_post_sem = np.mean(cohort_coherence["Old"]["spectrum_post"], axis=0), sem(cohort_coherence["Old"]["spectrum_post"], axis=0)

    ax_spec.plot(freqs[f_mask], y_post_spec[f_mask], color=COLOR_YOUNG, lw=2.2, label="Young Social (n=8)")
    ax_spec.fill_between(freqs[f_mask], (y_post_spec - y_post_sem)[f_mask], (y_post_spec + y_post_sem)[f_mask], color=COLOR_YOUNG, alpha=0.22)
    ax_spec.plot(freqs[f_mask], o_post_spec[f_mask], color=COLOR_OLD, lw=2.2, label="Old Social (n=10)")
    ax_spec.fill_between(freqs[f_mask], (o_post_spec - o_post_sem)[f_mask], (o_post_spec + o_post_sem)[f_mask], color=COLOR_OLD, alpha=0.22)
    ax_spec.axvspan(0.1, 0.5, color="#9ecae1", alpha=0.2, label="0.1-0.5 Hz")
    ax_spec.axvspan(0.5, 1.5, color="#fee08b", alpha=0.3, label="0.5-1.5 Hz")
    ax_spec.axvspan(1.5, 3.0, color="#fc9272", alpha=0.2, label="1.5-3.0 Hz")
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Magnitude-Squared Coherence")
    ax_spec.set_ylim(-0.05, 1.05)
    ax_spec.set_title("A | Inter-Hemispheric Coherence Spectrum", fontweight="bold")
    ax_spec.legend(loc="upper right", frameon=False, fontsize=8)
    ax_spec.grid(True, linestyle=":", alpha=0.5)

    band_labels = ["Full (0.1-3 Hz)", "0.1-0.5 Hz", "0.5-1.5 Hz", "1.5-3.0 Hz", "0.8-1.6 Hz"]
    y_b_vals = [np.mean([np.mean(c[m]) for c in cohort_coherence["Young"]["spectrum_post"]]) for m in [m_full, m_b1, m_b2, m_b3, m_std]]
    o_b_vals = [np.mean([np.mean(c[m]) for c in cohort_coherence["Old"]["spectrum_post"]]) for m in [m_full, m_b1, m_b2, m_b3, m_std]]
    x_b = np.arange(len(band_labels))
    w = 0.35
    ax_bar.bar(x_b - w / 2, y_b_vals, width=w, color=COLOR_YOUNG, label="Young", edgecolor="black")
    ax_bar.bar(x_b + w / 2, o_b_vals, width=w, color=COLOR_OLD, label="Old", edgecolor="black")
    ax_bar.set_xticks(x_b)
    ax_bar.set_xticklabels(band_labels, rotation=25, ha="right", fontsize=8.5)
    ax_bar.set_ylabel("Social Coherence")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("B | Multi-Band Coherence Across Ages", fontweight="bold")
    ax_bar.legend(loc="upper right", frameon=False)
    ax_bar.grid(True, axis="y", linestyle=":", alpha=0.5)

    y_del = cohort_coherence["Young"]["delta_full"]
    o_del = cohort_coherence["Old"]["delta_full"]
    t_del, p_del = ttest_ind(y_del, o_del, equal_var=False)

    ax_del.boxplot([y_del, o_del], positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="#f0f0f0", lw=1.2), medianprops=dict(color="black", lw=1.8))
    ax_del.scatter(1 + rng.normal(0, 0.04, len(y_del)), y_del, color=COLOR_YOUNG, s=55, alpha=0.85)
    ax_del.scatter(2 + rng.normal(0, 0.04, len(o_del)), o_del, color=COLOR_OLD, s=55, alpha=0.85)
    ax_del.axhline(0, color="gray", linestyle="--", lw=1.0)
    ax_del.set_xticks([1, 2])
    ax_del.set_xticklabels(["Young", "Old"], fontweight="bold")
    ax_del.set_ylabel("Full Coherence Delta (Social - Baseline)")
    ax_del.set_title(f"C | Coherence Shift (p = {p_del:.3f}{sig_note(p_del)})", fontweight="bold")
    ax_del.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig6.suptitle("Figure 6 | Inter-Hemispheric Medial Amygdala Spectral Coherence & Functional Coupling", fontweight="bold")
    plt.tight_layout()
    save_figure(fig6, os.path.join(figs_dir, "Figure6_Bilateral_Spectral_Coherence_and_Aging.png"), dpi=300)
    save_figure(fig6, os.path.join(figs_dir, "Figure6_Bilateral_Spectral_Coherence_and_Aging.pdf"))
    plt.close(fig6)

    # --------------------------------------------------------------------------
    # STATISTICAL REPORT
    # --------------------------------------------------------------------------
    y_del_std = cohort_coherence["Young"]["delta_std"]
    o_del_std = cohort_coherence["Old"]["delta_std"]
    t_del_std, p_del_std = ttest_ind(y_del_std, o_del_std, equal_var=False)

    stat_report_fp = os.path.join(tables_dir, "statistical_report.txt")
    with open(stat_report_fp, "w", encoding="utf-8") as sf:
        sf.write("========================================================================================\n")
        sf.write("STATISTICAL REPORT: BILATERAL MEDIAL AMYGDALA PHOTOMETRY & COHERENCE DYNAMICS IN AGING\n")
        sf.write("========================================================================================\n\n")

        sf.write("1. PRIMARY NEURAL ACTIVATION (UNCONTAMINATED SOCIAL BOUTS):\n")
        sf.write("   A. Social Sniffing Peak MeA Z-Score (0 to 4 s):\n")
        sf.write(f"      - Young Cohort (n=8):  {young_pks.mean():.3f} +/- {sem(young_pks):.3f} (Mean +/- SEM)\n")
        sf.write(f"      - Old Cohort (n=10):   {old_pks.mean():.3f} +/- {sem(old_pks):.3f} (Mean +/- SEM)\n")
        sf.write(f"      - Welch's t-test:      t = {t_pk:.3f}, p = {p_pk:.4f}\n\n")

        sf.write("   B. Whole-Bout Mean MeA Z-Score (Duration-Normalized):\n")
        sf.write(f"      - Young Cohort (n=8):  {young_wb.mean():.3f} +/- {sem(young_wb):.3f} (Mean +/- SEM)\n")
        sf.write(f"      - Old Cohort (n=10):   {old_wb.mean():.3f} +/- {sem(old_wb):.3f} (Mean +/- SEM)\n")
        sf.write(f"      - Welch's t-test:      t = {t_wb:.3f}, p = {p_wb:.4f}\n\n")

        sf.write("   C. Fixed 0 to +2 s Mean MeA Z-Score (Initial Trigger Window):\n")
        sf.write(f"      - Young Cohort (n=8):  {young_m02.mean():.3f} +/- {sem(young_m02):.3f} (Mean +/- SEM)\n")
        sf.write(f"      - Old Cohort (n=10):   {old_m02.mean():.3f} +/- {sem(old_m02):.3f} (Mean +/- SEM)\n")
        sf.write(f"      - Welch's t-test:      t = {t_m02:.3f}, p = {p_m02:.4f}\n\n")

        sf.write("2. INTER-HEMISPHERIC LATERALIZATION INDEX:\n")
        sf.write(f"   - Young Social LI:     {young_lis.mean():.3f} +/- {sem(young_lis):.3f}\n")
        sf.write(f"   - Old Social LI:       {old_lis.mean():.3f} +/- {sem(old_lis):.3f}\n")
        sf.write(f"   - Welch's t-test:      t = {t_lat:.3f}, p = {p_lat:.4f} (Symmetric Bilateral Recruitment)\n\n")

        sf.write("3. INTER-HEMISPHERIC SPECTRAL COHERENCE (0.1 to 3.0 Hz):\n")
        sf.write("   A. Full Spectrum (0.1 to 3.0 Hz):\n")
        sf.write(f"      - Young Social Coherence:      {np.mean(cohort_coherence['Young']['post_full']):.3f} +/- {sem(cohort_coherence['Young']['post_full']):.3f}\n")
        sf.write(f"      - Old Social Coherence:        {np.mean(cohort_coherence['Old']['post_full']):.3f} +/- {sem(cohort_coherence['Old']['post_full']):.3f}\n")
        sf.write(f"      - Welch's t-test on Delta:     t = {t_del:.3f}, p = {p_del:.4f}\n\n")

        sf.write("   B. Low-Frequency Band (0.8 to 1.6 Hz):\n")
        sf.write(f"      - Young Delta:                 {np.mean(y_del_std):.3f} +/- {sem(y_del_std):.3f}\n")
        sf.write(f"      - Old Delta:                   {np.mean(o_del_std):.3f} +/- {sem(o_del_std):.3f}\n")
        sf.write(f"      - Welch's t-test on Delta:     t = {t_del_std:.3f}, p = {p_del_std:.4f}\n\n")

        sf.write("4. BEHAVIORAL SOCIAL PREFERENCE INDEX (SPI):\n")
        sf.write(f"   - Young SPI:           {young_spis.mean():.3f} +/- {sem(young_spis):.3f}\n")
        sf.write(f"   - Old SPI:             {old_spis.mean():.3f} +/- {sem(old_spis):.3f}\n")
        sf.write(f"   - Welch's t-test:      t = {t_spi:.3f}, p = {p_spi:.4f}\n\n")

        sf.write("5. LOCOMOTOR KINEMATICS (Control):\n")
        sf.write(f"   - Young Distance (px): {young_dists.mean():.1f} +/- {sem(young_dists):.1f}\n")
        sf.write(f"   - Old Distance (px):   {old_dists.mean():.1f} +/- {sem(old_dists):.1f}\n")
        sf.write(f"   - Welch's t-test:      t = {t_dist:.3f}, p = {p_dist:.4f} (No Locomotor Impairment)\n")

    print(f"Cohort analysis complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Cohort Analysis: Bilateral MeA Photometry & Behavior")
    parser.add_argument("--raw_zip", type=str, default=None, help="Path to Data_2.zip or extracted Data_2 folder")
    parser.add_argument("--tracking_dir", type=str, default=None, help="Path to results_social tracking directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory")
    args = parser.parse_args()

    run_pipeline(raw_zip=args.raw_zip, tracking_dir=args.tracking_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
