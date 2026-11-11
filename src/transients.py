"""
Discrete calcium transient detection via Median Absolute Deviation (MAD) thresholding.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths


def detect_transients(time, z_left, z_right, soc_onsets, soc_durs, fs=30.0, mad_mult=2.5):
    """Detect calcium events and separate basal (out-of-bout) vs social-evoked rates."""
    z_bi = (z_left + z_right) / 2.0
    med = np.median(z_bi)
    mad = np.median(np.abs(z_bi - med))
    thresh = med + mad_mult * (mad * 1.4826)

    pks, props = find_peaks(
        z_bi,
        height=thresh,
        prominence=0.8,
        distance=max(1, int(fs * 0.5)),
    )

    in_soc = np.zeros(len(time), dtype=bool)
    for o, d in zip(soc_onsets, soc_durs):
        in_soc |= (time >= o) & (time <= o + d)

    total_min = (time[-1] - time[0]) / 60.0
    soc_min = np.sum(in_soc) / fs / 60.0
    non_soc_min = np.sum(~in_soc) / fs / 60.0

    pk_in_soc = int(np.sum(in_soc[pks])) if len(pks) > 0 else 0
    pk_out_soc = int(len(pks) - pk_in_soc)

    amp_mean = float(np.mean(props["peak_heights"])) if len(pks) > 0 else np.nan
    fwhm_s = float(np.mean(peak_widths(z_bi, pks, rel_height=0.5)[0] / fs)) if len(pks) > 0 else np.nan

    rate_total = len(pks) / total_min if total_min > 0 else np.nan
    rate_basal = pk_out_soc / non_soc_min if non_soc_min > 0 else np.nan
    rate_social = pk_in_soc / soc_min if soc_min > 0 else np.nan

    return {
        "transient_count_total": int(len(pks)),
        "transient_rate_total_per_min": round(float(rate_total), 3),
        "transient_rate_basal_per_min": round(float(rate_basal), 3),
        "transient_rate_social_per_min": round(float(rate_social), 3) if not np.isnan(rate_social) else np.nan,
        "transient_peak_amp_mean": round(amp_mean, 3) if not np.isnan(amp_mean) else np.nan,
        "transient_fwhm_s": round(fwhm_s, 3) if not np.isnan(fwhm_s) else np.nan,
    }
