"""
Bilateral Medial Amygdala (MeA) Photometry and Social Interaction Analysis
Author: Saad Yusuf
"""

from .preprocessing import butter_lowpass_filter, motion_correct_and_zscore
from .peth import extract_bouts_from_events, slice_peth, compute_trial_metrics, calc_lateralization_index
from .coherence import compute_spectral_coherence_bouts
from .tracking import load_tracking_data, compute_spatial_occupancy, compute_spatial_neural_map

__all__ = [
    "butter_lowpass_filter",
    "motion_correct_and_zscore",
    "extract_bouts_from_events",
    "slice_peth",
    "compute_trial_metrics",
    "calc_lateralization_index",
    "compute_spectral_coherence_bouts",
    "load_tracking_data",
    "compute_spatial_occupancy",
    "compute_spatial_neural_map",
]
