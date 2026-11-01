"""
Behavioral trajectory tracking, chamber dwell times, and spatial heatmaps.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def load_tracking_data(tracking_csv, fps=60.0):
    """Load tracking CSV and compute total locomotor distance, compartment dwell times, and SPI."""
    df_tr = pd.read_csv(tracking_csv)
    x_pts = df_tr["x_px"].dropna().values
    y_pts = df_tr["y_px"].dropna().values

    total_dist_px = float(np.sum(np.sqrt(np.diff(x_pts) ** 2 + np.diff(y_pts) ** 2)))
    stim = df_tr["compartment_stimulus"].values

    t_soc = float(np.sum(stim == "Novel Mouse") / fps)
    t_obj = float(np.sum(stim == "Object") / fps)
    spi = (t_soc - t_obj) / (t_soc + t_obj + 1e-9)

    return {
        "x_px": x_pts,
        "y_px": y_pts,
        "total_distance_px": total_dist_px,
        "time_social_s": t_soc,
        "time_object_s": t_obj,
        "social_preference_index": spi,
        "raw_dataframe": df_tr,
    }


def compute_spatial_occupancy(x_pts, y_pts, x_bins=60, y_bins=30, sigma=1.5):
    """Compute 2D Gaussian-filtered spatial dwell occupancy heatmap."""
    valid = ~(np.isnan(x_pts) | np.isnan(y_pts))
    h, xedges, yedges = np.histogram2d(x_pts[valid], y_pts[valid], bins=[x_bins, y_bins])
    h_smooth = gaussian_filter(h.T, sigma=sigma)
    return h_smooth, xedges, yedges


def compute_spatial_neural_map(x_pts, y_pts, neural_z, x_bins=60, y_bins=30, sigma=1.5):
    """Compute 2D spatial distribution of mean neural Z-score across the arena."""
    valid = ~(np.isnan(x_pts) | np.isnan(y_pts) | np.isnan(neural_z))
    count, xedges, yedges = np.histogram2d(x_pts[valid], y_pts[valid], bins=[x_bins, y_bins])
    z_sum, _, _ = np.histogram2d(x_pts[valid], y_pts[valid], bins=[x_bins, y_bins], weights=neural_z[valid])
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_z = np.where(count > 0, z_sum / count, np.nan)
    mean_z_smooth = gaussian_filter(np.nan_to_num(mean_z.T), sigma=sigma)
    return mean_z_smooth, xedges, yedges
