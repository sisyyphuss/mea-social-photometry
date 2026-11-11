"""
Publication-ready visualization styles and figure generators.
"""

import os
import time as _time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def save_figure(fig, filepath, dpi=None):
    """Save figure atomically to avoid transient OS file-lock errors."""
    ext = os.path.splitext(filepath)[1].lstrip(".").lower() or "png"
    tmp_path = f"{filepath}.tmp"
    kwargs = {"format": ext}
    if dpi is not None:
        kwargs["dpi"] = dpi
    fig.savefig(tmp_path, **kwargs)
    for attempt in range(5):
        try:
            os.replace(tmp_path, filepath)
            return
        except OSError:
            _time.sleep(0.2 * (attempt + 1))
    os.replace(tmp_path, filepath)

COLOR_LEFT = "#1b9e77"   # Left MeA (Teal)
COLOR_RIGHT = "#7570b3"  # Right MeA (Purple)
COLOR_YOUNG = "#2c7bb6"  # Young Cohort (Blue)
COLOR_OLD = "#d7191c"    # Old Cohort (Red)


def set_publication_style():
    """Apply clean publication styling parameters to Matplotlib."""
    plt.rcParams.update({
        "font.sans-serif": "Arial",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def compute_normalized_occupancy(x_pts, y_pts, bins=(80, 50), sigma=1.8):
    """Compute normalized 2D Gaussian-smoothed spatial occupancy density over [0, 1] x [0, 1]."""
    x_min, x_max = float(np.min(x_pts)), float(np.max(x_pts))
    y_min, y_max = float(np.min(y_pts)), float(np.max(y_pts))
    x_norm = np.clip((x_pts - x_min) / (x_max - x_min + 1e-9), 0.0, 1.0)
    y_norm = np.clip((y_pts - y_min) / (y_max - y_min + 1e-9), 0.0, 1.0)
    h, _, _ = np.histogram2d(x_norm, y_norm, bins=list(bins), range=[[0, 1], [0, 1]])
    h_smooth = gaussian_filter(h.T, sigma=sigma)
    return h_smooth / (np.sum(h_smooth) + 1e-9), h_smooth


def plot_individual_mouse_figures(
    out_dir,
    mouse_id,
    age_group,
    time,
    ch1_470_f,
    ch1_fit,
    ch2_470_f,
    ch2_fit,
    ch1_z,
    ch2_z,
    soc_on,
    soc_dur,
    obj_on,
    obj_dur,
    rel_time,
    peth_soc_l,
    peth_soc_r,
    peth_obj_l,
    peth_obj_r,
    mean_soc_wb_bilat,
    mean_obj_wb_bilat,
    soc_lis,
    obj_lis,
    x_pts,
    y_pts,
    dist_px,
    spi_beh,
    h_smooth_counts,
    freqs_soc,
    coh_soc_pre,
    coh_soc_post,
    c_del_bands,
    m_c_full_del,
):
    """Generate the five individual diagnostic plots for a single mouse."""
    rng = np.random.default_rng(42)

    # 1. Continuous traces with exact [onset, onset + duration] shading
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(time, ch1_470_f, color=COLOR_LEFT, label="Left MeA 470 nm", lw=1.0)
    ax[0].plot(time, ch1_fit, color="gray", linestyle="--", label="Fitted 410 nm", lw=1.0)
    ax[0].set_ylabel("Fluor. (a.u.)")
    ax[0].set_title(f"Mouse {mouse_id} ({age_group}) | Channel 1: Left MeA Signal & Fitted Reference", fontweight="bold")
    ax[0].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.88, edgecolor="none")

    ax[1].plot(time, ch2_470_f, color=COLOR_RIGHT, label="Right MeA 470 nm", lw=1.0)
    ax[1].plot(time, ch2_fit, color="gray", linestyle="--", label="Fitted 410 nm", lw=1.0)
    ax[1].set_ylabel("Fluor. (a.u.)")
    ax[1].set_title(f"Mouse {mouse_id} ({age_group}) | Channel 2: Right MeA Signal & Fitted Reference", fontweight="bold")
    ax[1].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.88, edgecolor="none")

    ax[2].plot(time, ch1_z, color=COLOR_LEFT, label="Left MeA (Z-score)", lw=1.0)
    ax[2].plot(time, ch2_z, color=COLOR_RIGHT, label="Right MeA (Z-score)", lw=1.0, alpha=0.85)
    for idx_s, (o, d) in enumerate(zip(soc_on, soc_dur)):
        ax[2].axvspan(o, o + d, color="#fee08b", alpha=0.35, label="Social" if idx_s == 0 else "")
    for idx_o, (o, d) in enumerate(zip(obj_on, obj_dur)):
        ax[2].axvspan(o, o + d, color="#d9d9d9", alpha=0.45, label="Object" if idx_o == 0 else "")
    z_lo = float(min(np.min(ch1_z), np.min(ch2_z)))
    z_hi = float(max(np.max(ch1_z), np.max(ch2_z)))
    ax[2].set_ylim(z_lo - 0.5, z_hi + 0.22 * (z_hi - z_lo))
    ax[2].set_ylabel("Standardized Z")
    ax[2].set_xlabel("Session Time (s)")
    ax[2].set_title("Continuous Motion-Corrected Traces with Exact Investigation Epochs Shaded", fontweight="bold")
    ax[2].legend(loc="upper right", ncol=4, frameon=True, facecolor="white", framealpha=0.88, edgecolor="none")
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "continuous_traces.png"), dpi=250)
    plt.close()

    # 2. Event-aligned PETH
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)

    def render_peth(axis, l_mat, r_mat, title):
        if len(l_mat) > 0:
            m_l = np.mean(l_mat, axis=0)
            s_l = np.std(l_mat, axis=0) / np.sqrt(len(l_mat))
            axis.plot(rel_time, m_l, color=COLOR_LEFT, label=f"Left MeA (n={len(l_mat)})", lw=2)
            axis.fill_between(rel_time, m_l - s_l, m_l + s_l, color=COLOR_LEFT, alpha=0.25)
        if len(r_mat) > 0:
            m_r = np.mean(r_mat, axis=0)
            s_r = np.std(r_mat, axis=0) / np.sqrt(len(r_mat))
            axis.plot(rel_time, m_r, color=COLOR_RIGHT, label=f"Right MeA (n={len(r_mat)})", lw=2)
            axis.fill_between(rel_time, m_r - s_r, m_r + s_r, color=COLOR_RIGHT, alpha=0.25)
        axis.axvline(0, color="black", linestyle=":", lw=1.2)
        axis.axhline(0, color="gray", linestyle="--", lw=0.7, alpha=0.6)
        axis.axvspan(0, 2.0, color="#abdda4", alpha=0.22, label="0-2 s Window")
        axis.axvspan(2.0, 4.0, color="#fee08b", alpha=0.14)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Time Relative to Onset (s)")
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.88, edgecolor="none")

    ax[0].set_ylabel("Baseline-Corrected Z-Score")
    render_peth(ax[0], peth_soc_l, peth_soc_r, f"Social Bouts (Whole-Bout Mean Z = {mean_soc_wb_bilat:+.2f})")
    render_peth(ax[1], peth_obj_l, peth_obj_r, f"Non-Social Bouts (Whole-Bout Mean Z = {mean_obj_wb_bilat:+.2f})")
    y_lo, y_hi = ax[0].get_ylim()
    ax[0].set_ylim(y_lo, y_hi + 0.22 * (y_hi - y_lo))
    fig.suptitle(f"Mouse {mouse_id} ({age_group}) | Event-Triggered Average Transients", fontweight="bold")
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "peth_social_vs_nonsocial.png"), dpi=250)
    plt.close()

    # 3. Trial-by-trial lateralization distribution
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.boxplot([soc_lis, obj_lis], positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="#f0f0f0", lw=1.2), medianprops=dict(color="black", lw=1.5))
    ax.scatter(np.ones(len(soc_lis)) + rng.normal(0, 0.04, len(soc_lis)), soc_lis, color="#e41a1c", s=38, alpha=0.75)
    ax.scatter(2 * np.ones(len(obj_lis)) + rng.normal(0, 0.04, len(obj_lis)), obj_lis, color="gray", s=38, alpha=0.75)
    ax.axhline(0, color="gray", linestyle="--", lw=1)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Social", "Non-Social"], fontweight="bold")
    ax.set_ylabel("Lateralization Index")
    ax.set_ylim(-1.08, 1.08)
    ax.set_title(f"Mouse {mouse_id} | Trial-by-Trial Lateralization\n(+1: Left MeA, -1: Right MeA)", fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "lateralization_distribution.png"), dpi=250)
    plt.close()

    # 4. Spatial trajectory and normalized dwell heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    ax1.plot(x_pts[::3], y_pts[::3], color="#3182bd", lw=0.65, alpha=0.6)
    ax1.set_title(f"Mouse Trajectory ({dist_px:.0f} px)", fontweight="bold")
    ax1.set_xlabel("Arena X (px)")
    ax1.set_ylabel("Arena Y (px)")
    ax1.invert_yaxis()

    im = ax2.imshow(h_smooth_counts, origin="upper", cmap="inferno", aspect="auto", extent=[0.0, 1.0, 1.0, 0.0])
    ax2.set_title(f"Spatial Presence Heatmap (SPI = {spi_beh:+.2f})", fontweight="bold")
    ax2.set_xlabel("Arena X (Normalized)")
    ax2.set_ylabel("Arena Y (Normalized)")
    plt.colorbar(im, ax=ax2, label="Occupancy Density")
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "spatial_trajectory_and_occupancy.png"), dpi=250)
    plt.close()

    # 5. Bilateral spectral coherence
    if len(coh_soc_pre) > 0 and len(freqs_soc) > 0:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.4))
        m_pre = np.mean(coh_soc_pre, axis=0)
        s_pre = np.std(coh_soc_pre, axis=0) / np.sqrt(len(coh_soc_pre))
        m_post = np.mean(coh_soc_post, axis=0)
        s_post = np.std(coh_soc_post, axis=0) / np.sqrt(len(coh_soc_post))

        f_mask = freqs_soc <= 3.0
        ax1.plot(freqs_soc[f_mask], m_pre[f_mask], color="gray", lw=2.0, label="Baseline (-5 to 0 s)")
        ax1.fill_between(freqs_soc[f_mask], (m_pre - s_pre)[f_mask], (m_pre + s_pre)[f_mask], color="gray", alpha=0.2)
        ax1.plot(freqs_soc[f_mask], m_post[f_mask], color="#e41a1c", lw=2.0, label="Social Interaction (0 to 5 s)")
        ax1.fill_between(freqs_soc[f_mask], (m_post - s_post)[f_mask], (m_post + s_post)[f_mask], color="#e41a1c", alpha=0.25)
        ax1.axvspan(0.1, 0.5, color="#9ecae1", alpha=0.2, label="0.1-0.5 Hz")
        ax1.axvspan(0.5, 1.5, color="#fee08b", alpha=0.3, label="0.5-1.5 Hz")
        ax1.axvspan(1.5, 3.0, color="#fc9272", alpha=0.2, label="1.5-3.0 Hz")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude-Squared Coherence")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_title("Left-Right MeA Coherence Spectrum", fontweight="bold")
        ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.88, edgecolor="none", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.5)

        mask_full = (freqs_soc >= 0.1) & (freqs_soc <= 3.0)
        mask_b1 = (freqs_soc >= 0.1) & (freqs_soc <= 0.5)
        mask_b2 = (freqs_soc >= 0.5) & (freqs_soc <= 1.5)
        mask_b3 = (freqs_soc >= 1.5) & (freqs_soc <= 3.0)
        mask_std = (freqs_soc >= 0.8) & (freqs_soc <= 1.6)
        bands_names = ["Full (0.1-3 Hz)", "0.1-0.5 Hz", "0.5-1.5 Hz", "1.5-3.0 Hz", "0.8-1.6 Hz"]
        c_pre_b = [np.mean(m_pre[m]) for m in [mask_full, mask_b1, mask_b2, mask_b3, mask_std]]
        c_post_b = [np.mean(m_post[m]) for m in [mask_full, mask_b1, mask_b2, mask_b3, mask_std]]
        x_idx = np.arange(len(bands_names))
        w = 0.35
        ax2.bar(x_idx - w / 2, c_pre_b, width=w, color="#999999", label="Baseline", edgecolor="black")
        ax2.bar(x_idx + w / 2, c_post_b, width=w, color="#e41a1c", label="Social", edgecolor="black")
        ax2.set_xticks(x_idx)
        ax2.set_xticklabels(bands_names, rotation=25, ha="right", fontsize=8.5)
        ax2.set_ylabel("Magnitude-Squared Coherence")
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Multi-Band Coherence Comparison", fontweight="bold")
        ax2.legend(loc="upper right", frameon=False)
        ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

        b_labels = ["0.1-0.5 Hz", "0.5-1.5 Hz", "1.5-3.0 Hz", "Full (0.1-3 Hz)"]
        ax3.boxplot(c_del_bands, positions=[1, 2, 3, 4], widths=0.45, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor="#f0f0f0", lw=1.2), medianprops=dict(color="black", lw=1.5))
        for p_i, arr in enumerate(c_del_bands):
            ax3.scatter(np.full(len(arr), p_i + 1) + rng.normal(0, 0.04, len(arr)), arr, color="#d7191c", s=28, alpha=0.65)
        ax3.axhline(0, color="gray", linestyle="--", lw=1)
        ax3.set_xticks([1, 2, 3, 4])
        ax3.set_xticklabels(b_labels, rotation=20, ha="right", fontsize=8.5)
        ax3.set_ylabel("Coherence Shift (Delta C)")
        ax3.set_title(f"Coupling Shifts (Full Delta: {m_c_full_del:+.3f})", fontweight="bold")
        ax3.grid(True, axis="y", linestyle=":", alpha=0.5)

        fig.suptitle(f"Mouse {mouse_id} ({age_group}) | Bilateral MeA Spectral Coherence", fontweight="bold")
        plt.tight_layout()
        save_figure(fig, os.path.join(out_dir, "bilateral_spectral_coherence.png"), dpi=250)
        plt.close()
