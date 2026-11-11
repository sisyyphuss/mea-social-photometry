"""
Hierarchical mixed-effects modeling, small-sample validation, and PETH divergence windows.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, pearsonr
import statsmodels.formula.api as smf


def hedges_g(x, y):
    """Compute bias-corrected Hedges' g effect size for small independent samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    s_pool = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    d = (np.mean(x) - np.mean(y)) / (s_pool + 1e-9)
    correction = 1.0 - (3.0 / (4.0 * dof - 1.0))
    return float(d * correction)


def run_nonparametric_and_lomo(df_sum):
    """Run Shapiro-Wilk, Mann-Whitney U, Hedges' g, and 18-fold Leave-One-Mouse-Out jackknife."""
    df_eval = df_sum.copy()
    df_eval["Peak_Bilat"] = (df_eval["Social_Peak_Left"] + df_eval["Social_Peak_Right"]) / 2.0

    metrics = [
        ("Peak_Bilat", "Peak MeA Z (0-4 s)"),
        ("Social_Whole_Bout_Mean_Z_Bilateral", "Whole-Bout Mean Z"),
        ("Social_Mean_Z_0_2s_Bilateral", "Fixed 0-2 s Mean Z"),
    ]

    summary_rows = []
    lomo_rows = []

    for col, label in metrics:
        y = df_eval[df_eval["Age_Group"] == "Young"][col].values
        o = df_eval[df_eval["Age_Group"] == "Old"][col].values

        _, p_shap_y = shapiro(y)
        _, p_shap_o = shapiro(o)
        t_welch, p_welch = ttest_ind(y, o, equal_var=False)
        u_stat, p_mw = mannwhitneyu(y, o, alternative="two-sided")
        g_val = hedges_g(y, o)

        p_drops = []
        for idx, row in df_eval.iterrows():
            sub = df_eval.drop(idx)
            yy = sub[sub["Age_Group"] == "Young"][col].values
            oo = sub[sub["Age_Group"] == "Old"][col].values
            _, pv = ttest_ind(yy, oo, equal_var=False)
            p_drops.append(pv)
            lomo_rows.append({
                "Metric": label,
                "Dropped_Mouse": str(row["Mouse_ID"]),
                "Dropped_Group": row["Age_Group"],
                "Welch_p": round(float(pv), 5),
            })

        summary_rows.append({
            "Metric": label,
            "Young_Mean": round(float(np.mean(y)), 3),
            "Old_Mean": round(float(np.mean(o)), 3),
            "Welch_t": round(float(t_welch), 3),
            "Welch_p": round(float(p_welch), 4),
            "Shapiro_Young_p": round(float(p_shap_y), 4),
            "Shapiro_Old_p": round(float(p_shap_o), 4),
            "MannWhitney_U": round(float(u_stat), 1),
            "MannWhitney_p": round(float(p_mw), 4),
            "Hedges_g": round(g_val, 3),
            "LOMO_p_Min": round(float(np.min(p_drops)), 4),
            "LOMO_p_Max": round(float(np.max(p_drops)), 4),
            "LOMO_Significant_Count": int(np.sum(np.array(p_drops) < 0.05)),
            "LOMO_Total": len(p_drops),
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(lomo_rows)


def fit_trial_lmm(df_trials):
    """Fit hierarchical LMM across all social bouts and test AUC vs duration dependence."""
    soc = df_trials[df_trials["Stimulus"] == "Social"].copy()
    soc["Peak_Bilat"] = (soc["Left_MeA_Peak_Z"] + soc["Right_MeA_Peak_Z"]) / 2.0
    soc["AUC_Bilat"] = (soc["Left_MeA_AUC"] + soc["Right_MeA_AUC"]) / 2.0
    soc["Is_Old"] = (soc["Age_Group"] == "Old").astype(int)

    targets = [
        ("Peak_Bilat", "Peak MeA Z (0-4 s)"),
        ("Whole_Bout_Mean_Z_Bilateral", "Whole-Bout Mean Z"),
        ("Mean_Z_0_2s_Bilateral", "Fixed 0-2 s Mean Z"),
    ]

    lmm_rows = []
    for col, label in targets:
        fit = smf.mixedlm(f"{col} ~ Is_Old + Bout + Duration_s", soc, groups=soc["Mouse_ID"]).fit()
        ci = fit.conf_int()
        for param in ["Is_Old", "Bout", "Duration_s"]:
            lmm_rows.append({
                "Outcome": label,
                "Predictor": param,
                "Beta": round(float(fit.params[param]), 4),
                "SE": round(float(fit.bse[param]), 4),
                "CI_Low": round(float(ci.loc[param, 0]), 4),
                "CI_High": round(float(ci.loc[param, 1]), 4),
                "z_stat": round(float(fit.tvalues[param]), 3),
                "p_value": round(float(fit.pvalues[param]), 4),
            })

    y_soc = soc[soc["Age_Group"] == "Young"]
    r_auc, p_auc = pearsonr(y_soc["Duration_s"], y_soc["AUC_Bilat"])
    r_wb, p_wb = pearsonr(y_soc["Duration_s"], y_soc["Whole_Bout_Mean_Z_Bilateral"])
    r_m02, p_m02 = pearsonr(y_soc["Duration_s"], y_soc["Mean_Z_0_2s_Bilateral"])

    dur_checks = {
        "Young_r_Dur_AUC": round(float(r_auc), 3),
        "Young_p_Dur_AUC": round(float(p_auc), 4),
        "Young_r_Dur_WB_MeanZ": round(float(r_wb), 3),
        "Young_p_Dur_WB_MeanZ": round(float(p_wb), 4),
        "Young_r_Dur_0_2s": round(float(r_m02), 3),
        "Young_p_Dur_0_2s": round(float(p_m02), 4),
    }

    return pd.DataFrame(lmm_rows), dur_checks


def find_consecutive_peth_windows(rel_time, young_mat, old_mat, min_consec_s=0.20, fs=30.0, n_boot=2000):
    """Compute bootstrapped 95% CIs and consecutive-threshold Young vs Old divergence bands."""
    rng = np.random.default_rng(42)

    def boot_ci(mat):
        n = len(mat)
        boots = np.empty((n_boot, mat.shape[1]))
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots[b] = np.mean(mat[idx], axis=0)
        return np.mean(mat, axis=0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)

    y_mean, y_lo, y_hi = boot_ci(young_mat)
    o_mean, o_lo, o_hi = boot_ci(old_mat)

    p_vals = np.array([ttest_ind(young_mat[:, i], old_mat[:, i], equal_var=False)[1] for i in range(len(rel_time))])
    sig = p_vals < 0.05
    min_frames = max(1, int(round(min_consec_s * fs)))

    intervals = []
    start = None
    for i, s in enumerate(sig):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_frames:
                intervals.append((round(float(rel_time[start]), 3), round(float(rel_time[i - 1]), 3)))
            start = None
    if start is not None and len(sig) - start >= min_frames:
        intervals.append((round(float(rel_time[start]), 3), round(float(rel_time[-1]), 3)))

    return {
        "y_mean": y_mean, "y_ci_lo": y_lo, "y_ci_hi": y_hi,
        "o_mean": o_mean, "o_ci_lo": o_lo, "o_ci_hi": o_hi,
        "p_vals": p_vals,
        "sig_intervals_s": intervals,
    }
