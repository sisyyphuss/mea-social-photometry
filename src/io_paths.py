"""
Cross-platform path resolution and raw data loading utilities.
"""

import os
import io
import zipfile
import pandas as pd


def resolve_data_paths(raw_path=None, tracking_dir=None, repo_root=None):
    """Locate photometry source (zip or directory) and tracking directory across standard locations."""
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_candidates = [
        raw_path,
        os.path.join(repo_root, "data", "Data_2.zip"),
        os.path.join(repo_root, "data", "Data_2"),
        os.path.join(repo_root, "data"),
        os.path.expanduser("~/Downloads/Data_2.zip"),
        os.path.expanduser("~/Downloads/Data_2"),
    ]
    resolved_raw = None
    for cand in raw_candidates:
        if cand and os.path.exists(cand):
            if os.path.isfile(cand) and cand.lower().endswith(".zip"):
                resolved_raw = cand
                break
            if os.path.isdir(cand):
                if os.path.exists(os.path.join(cand, "Social")) or os.path.exists(os.path.join(cand, "Data_2", "Social")):
                    resolved_raw = cand
                    break

    if resolved_raw is None:
        raise FileNotFoundError(
            "Photometry data not found. Pass --raw_zip PATH or place Data_2.zip (or extracted Data_2/) in ./data/ or ~/Downloads/."
        )

    trk_candidates = [
        tracking_dir,
        os.path.join(repo_root, "data", "results_social", "results_social"),
        os.path.join(repo_root, "data", "results_social"),
        os.path.expanduser("~/Downloads/results_social/results_social"),
        os.path.expanduser("~/Downloads/results_social"),
    ]
    resolved_trk = None
    for cand in trk_candidates:
        if cand and os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "individual", "Social")):
            resolved_trk = cand
            break

    if resolved_trk is None:
        raise FileNotFoundError(
            "Tracking directory not found. Pass --tracking_dir PATH or place results_social/ in ./data/ or ~/Downloads/."
        )

    return resolved_raw, resolved_trk


def load_photometry_csv(raw_source, age_group, mouse_id):
    """Load raw dual-channel fluorescence CSV from either Data_2.zip or an extracted folder."""
    folder_age = age_group.lower()
    rel_sub = os.path.join("Social", folder_age, f"Fluorescence_{mouse_id}.csv")

    if os.path.isfile(raw_source) and raw_source.lower().endswith(".zip"):
        with zipfile.ZipFile(raw_source, "r") as z:
            candidates = [f"Data_2/Social/{folder_age}/Fluorescence_{mouse_id}.csv", f"Social/{folder_age}/Fluorescence_{mouse_id}.csv"]
            match_name = next((c for c in candidates if c in z.namelist()), None)
            if match_name is None:
                raise FileNotFoundError(f"Fluorescence_{mouse_id}.csv not found inside {raw_source}")
            with z.open(match_name) as f:
                content = f.read().decode("utf-8", errors="ignore")
                df_raw = pd.read_csv(io.StringIO(content), skiprows=1)
    else:
        cand_direct = os.path.join(raw_source, rel_sub)
        cand_nested = os.path.join(raw_source, "Data_2", rel_sub)
        target_fp = cand_direct if os.path.exists(cand_direct) else cand_nested
        if not os.path.exists(target_fp):
            raise FileNotFoundError(f"Photometry CSV not found: {target_fp}")
        with open(target_fp, "r", encoding="utf-8", errors="ignore") as f:
            df_raw = pd.read_csv(f, skiprows=1)

    df_raw.columns = [c.strip() for c in df_raw.columns]
    return df_raw


def load_tracking_csv(tracking_dir, age_group, mouse_id):
    """Load synchronized 60 fps behavioral tracking CSV for one animal."""
    fp = os.path.join(tracking_dir, "individual", "Social", age_group, str(mouse_id), "synchronized_tracking_photometry.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Tracking CSV not found: {fp}")
    return pd.read_csv(fp)
