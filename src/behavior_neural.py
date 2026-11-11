"""
Stimulus selectivity (auROC), bout stratification, offset/zone-entry dynamics, and brain-behavior coupling.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr


def compute_social_object_auroc(df_trials):
    """Compute per-mouse single-trial auROC decoding of Social vs NonSocial bouts."""
    df_t = df_trials.copy()
    df_t["Peak_Bilat"] = (df_t["Left_MeA_Peak_Z"] + df_t["Right_MeA_Peak_Z"]) / 2.0

    rows = []
    for mid, g in df_t.groupby("Mouse_ID"):
        age = g["Age_Group"].iloc[0]
        y_true = (g["Stimulus"] == "Social").astype(int).values
        if len(np.unique(y_true)) < 2:
            continue
        rows.append({
            "Mouse_ID": str(mid),
            "Age_Group": age,
            "auROC_Peak": round(float(roc_auc_score(y_true, g["Peak_Bilat"].values)), 4),
            "auROC_WB_Mean_Z": round(float(roc_auc_score(y_true, g["Whole_Bout_Mean_Z_Bilateral"].values)), 4),
            "auROC_0_2s": round(float(roc_auc_score(y_true, g["Mean_Z_0_2s_Bilateral"].values)), 4),
        })
    return pd.DataFrame(rows)


def compute_bout_stratification(df_trials):
    """Stratify social bouts by duration (<3 s vs >=3 s) and sequence order (Bouts 1-5 vs >5)."""
    soc = df_trials[df_trials["Stimulus"] == "Social"].copy()
    soc["Peak_Bilat"] = (soc["Left_MeA_Peak_Z"] + soc["Right_MeA_Peak_Z"]) / 2.0

    strat_rows = []
    for mid, g in soc.groupby("Mouse_ID"):
        age = g["Age_Group"].iloc[0]
        short_g = g[g["Duration_s"] < 3.0]
        long_g = g[g["Duration_s"] >= 3.0]
        early_g = g[g["Bout"] <= 5]
        late_g = g[g["Bout"] > 5]

        strat_rows.append({
            "Mouse_ID": str(mid),
            "Age_Group": age,
            "Short_Peak": round(float(short_g["Peak_Bilat"].mean()), 3) if len(short_g) > 0 else np.nan,
            "Long_Peak": round(float(long_g["Peak_Bilat"].mean()), 3) if len(long_g) > 0 else np.nan,
            "Short_WB_Mean_Z": round(float(short_g["Whole_Bout_Mean_Z_Bilateral"].mean()), 3) if len(short_g) > 0 else np.nan,
            "Long_WB_Mean_Z": round(float(long_g["Whole_Bout_Mean_Z_Bilateral"].mean()), 3) if len(long_g) > 0 else np.nan,
            "Early_Peak": round(float(early_g["Peak_Bilat"].mean()), 3) if len(early_g) > 0 else np.nan,
            "Late_Peak": round(float(late_g["Peak_Bilat"].mean()), 3) if len(late_g) > 0 else np.nan,
            "Bout1_Peak": round(float(g[g["Bout"] == 1]["Peak_Bilat"].mean()), 3) if (g["Bout"] == 1).any() else np.nan,
        })
    return pd.DataFrame(strat_rows)


def compute_offset_and_zone_entry(time_phot, z_l, z_r, soc_offsets, df_tr):
    """Compute bout-offset disengagement drop and 60 fps chamber-entry responses."""
    rel_off = np.linspace(-2.0, 2.0, 121)
    z_bi = (z_l + z_r) / 2.0

    off_deltas = []
    off_traces = []
    for off_t in soc_offsets:
        if off_t - 2.0 >= time_phot[0] and off_t + 2.0 <= time_phot[-1]:
            iz = np.interp(off_t + rel_off, time_phot, z_bi)
            pre = np.mean(iz[rel_off < 0])
            post = np.mean(iz[rel_off >= 0])
            off_deltas.append(post - pre)
            off_traces.append(iz - pre)

    t_tr = df_tr["video_time_sec"].values
    z_tr = np.interp(t_tr, time_phot, z_bi)
    comp = df_tr["compartment_stimulus"].values
    rel_z = np.linspace(-2.0, 4.0, 361)

    soc_entry_d, obj_entry_d = [], []
    last_s, last_o = -10.0, -10.0
    for i in range(1, len(comp)):
        if comp[i] == "Novel Mouse" and comp[i - 1] != "Novel Mouse":
            if t_tr[i] - last_s > 3.0 and t_tr[i] - 2.0 >= t_tr[0] and t_tr[i] + 4.0 <= t_tr[-1]:
                iz = np.interp(t_tr[i] + rel_z, t_tr, z_tr)
                soc_entry_d.append(np.mean(iz[(rel_z >= 0) & (rel_z <= 2.0)]) - np.mean(iz[rel_z < 0]))
                last_s = t_tr[i]
        elif comp[i] == "Object" and comp[i - 1] != "Object":
            if t_tr[i] - last_o > 3.0 and t_tr[i] - 2.0 >= t_tr[0] and t_tr[i] + 4.0 <= t_tr[-1]:
                iz = np.interp(t_tr[i] + rel_z, t_tr, z_tr)
                obj_entry_d.append(np.mean(iz[(rel_z >= 0) & (rel_z <= 2.0)]) - np.mean(iz[rel_z < 0]))
                last_o = t_tr[i]

    return {
        "offset_delta": round(float(np.mean(off_deltas)), 4) if len(off_deltas) > 0 else np.nan,
        "offset_rel_time": rel_off,
        "offset_mean_trace": np.mean(off_traces, axis=0) if len(off_traces) > 0 else np.zeros(len(rel_off)),
        "zone_entry_soc_delta": round(float(np.mean(soc_entry_d)), 4) if len(soc_entry_d) > 0 else np.nan,
        "zone_entry_obj_delta": round(float(np.mean(obj_entry_d)), 4) if len(obj_entry_d) > 0 else np.nan,
    }


def compute_lagged_crosscorr(time_phot, z_l, z_r, df_tr, max_lag_s=2.0, fps=60.0):
    """Compute non-wrapping time-lagged Pearson cross-correlation between Left and Right MeA."""
    max_frames = int(round(max_lag_s * fps))
    lags_s = np.arange(-max_frames, max_frames + 1) / fps

    t_tr = df_tr["video_time_sec"].values
    l = np.interp(t_tr, time_phot, z_l)
    r = np.interp(t_tr, time_phot, z_r)

    xc = []
    for k in range(-max_frames, max_frames + 1):
        if k > 0:
            x, y = l[:-k], r[k:]
        elif k < 0:
            x, y = l[-k:], r[:k]
        else:
            x, y = l, r
        xc.append(float(np.corrcoef(x, y)[0, 1]))
    xc = np.array(xc)
    pk_idx = int(np.argmax(xc))

    return {
        "lags_s": lags_s,
        "xcorr": xc,
        "zero_lag_r": round(float(xc[max_frames]), 4),
        "max_r": round(float(xc[pk_idx]), 4),
        "peak_lag_s": round(float(lags_s[pk_idx]), 4),
    }


def compute_brain_behavior_coupling(df_sum):
    """Compute within-cohort Pearson correlations between MeA reactivity and behavioral metrics."""
    df_s = df_sum.copy()
    df_s["Peak_Bilat"] = (df_s["Social_Peak_Left"] + df_s["Social_Peak_Right"]) / 2.0

    rows = []
    pairs = [
        ("Peak_Bilat", "Behavioral_Social_Preference_Index", "Peak Z vs SPI"),
        ("Peak_Bilat", "Chamber_Time_Social_s", "Peak Z vs Social Chamber Time"),
        ("Social_Whole_Bout_Mean_Z_Bilateral", "Behavioral_Social_Preference_Index", "Whole-Bout Mean Z vs SPI"),
        ("Social_Mean_Z_0_2s_Bilateral", "Chamber_Time_Social_s", "0-2 s Mean Z vs Social Chamber Time"),
    ]

    for n_col, b_col, label in pairs:
        for grp in ["Young", "Old"]:
            sub = df_s[df_s["Age_Group"] == grp]
            r_val, p_val = pearsonr(sub[n_col], sub[b_col])
            rows.append({
                "Relationship": label,
                "Cohort": grp,
                "Pearson_r": round(float(r_val), 3),
                "p_value": round(float(p_val), 4),
            })
    return pd.DataFrame(rows)
