"""
Bilateral Medial Amygdala (MeA) Photometry and Social Interaction Analysis
"""

from .io_paths import resolve_data_paths, load_photometry_csv, load_tracking_csv
from .preprocessing import butter_lowpass_filter, motion_correct_and_zscore
from .peth import extract_bouts_from_events, slice_peth, compute_trial_metrics, calc_lateralization_index
from .coherence import compute_spectral_coherence_bouts
from .tracking import load_tracking_data, compute_spatial_occupancy, compute_spatial_neural_map
from .qc import compute_optical_qc, compute_speed_coupling
from .transients import detect_transients
from .modeling import hedges_g, run_nonparametric_and_lomo, fit_trial_lmm, find_consecutive_peth_windows
from .behavior_neural import (
    compute_social_object_auroc,
    compute_bout_stratification,
    compute_offset_and_zone_entry,
    compute_lagged_crosscorr,
    compute_brain_behavior_coupling,
)
from .visualization import (
    set_publication_style,
    compute_normalized_occupancy,
    plot_individual_mouse_figures,
    COLOR_LEFT,
    COLOR_RIGHT,
    COLOR_YOUNG,
    COLOR_OLD,
)

__all__ = [
    "resolve_data_paths",
    "load_photometry_csv",
    "load_tracking_csv",
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
    "compute_optical_qc",
    "compute_speed_coupling",
    "detect_transients",
    "hedges_g",
    "run_nonparametric_and_lomo",
    "fit_trial_lmm",
    "find_consecutive_peth_windows",
    "compute_social_object_auroc",
    "compute_bout_stratification",
    "compute_offset_and_zone_entry",
    "compute_lagged_crosscorr",
    "compute_brain_behavior_coupling",
    "set_publication_style",
    "compute_normalized_occupancy",
    "plot_individual_mouse_figures",
    "COLOR_LEFT",
    "COLOR_RIGHT",
    "COLOR_YOUNG",
    "COLOR_OLD",
]
