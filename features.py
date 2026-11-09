import os
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 25 common frequency bands in order
BANDS = [
    '31.6', '39.8', '50.1', '63.1', '79.4', '100', '125.9', '158.5', '199.5',
    '251.2', '316.2', '398.1', '501.2', '631', '794.3', '1000', '1258.9',
    '1584.9', '1995.3', '2511.9', '3162.3', '3981.1', '5011.9', '6309.6', '7943.3'
]
TOL_COLS = [f'TOL_{b}' for b in BANDS]

def extract_features_from_window(db_matrix):
    """
    db_matrix: shape (15, 25) with dB values, may contain NaNs
    Returns: 1D numpy array of engineered acoustic features
    """
    # Impute NaNs along time per band
    mat = db_matrix.copy()
    for b in range(25):
        vals = mat[:, b]
        mask = np.isnan(vals)
        if mask.any():
            valid = vals[~mask]
            fill_val = np.median(valid) if len(valid) > 0 else 75.0
            mat[mask, b] = fill_val

    # Convert dB to linear energy: E = 10^(dB / 10)
    lin_energy = 10.0 ** (mat / 10.0) # (15, 25)

    # Total energy per minute (sum across 25 bands)
    minute_total_e = lin_energy.sum(axis=1) # (15,)
    sum_total_e = minute_total_e.sum() + 1e-12

    # Baseline estimates at various quantiles
    q10_e = np.percentile(minute_total_e, 10)
    q15_e = np.percentile(minute_total_e, 15)
    q20_e = np.percentile(minute_total_e, 20)
    min_e = np.min(minute_total_e)
    max_e = np.max(minute_total_e)
    median_e = np.median(minute_total_e)

    # Direct physical share approximations
    share_q10 = np.maximum(0, minute_total_e - q10_e).sum() / sum_total_e
    share_q15 = np.maximum(0, minute_total_e - q15_e).sum() / sum_total_e
    share_q20 = np.maximum(0, minute_total_e - q20_e).sum() / sum_total_e
    share_min = np.maximum(0, minute_total_e - min_e).sum() / sum_total_e
    share_med = np.maximum(0, minute_total_e - median_e).sum() / sum_total_e

    # Ratio contrasts
    ratio_max_min = max_e / (min_e + 1e-9)
    ratio_max_q15 = max_e / (q15_e + 1e-9)
    ratio_max_med = max_e / (median_e + 1e-9)

    # Log contrasts
    log_ratio_max_min = np.log10(ratio_max_min + 1e-9)
    log_ratio_max_q15 = np.log10(ratio_max_q15 + 1e-9)

    # Temporal dynamics across 15 minutes
    # Time of maximum energy (0 to 14)
    peak_idx = np.argmax(minute_total_e)
    peak_dist_center = abs(peak_idx - 7.0)

    # Standard deviation and CV of minute energy
    std_e = np.std(minute_total_e)
    cv_e = std_e / (np.mean(minute_total_e) + 1e-12)

    # Top 3 minutes energy fraction
    top3_e = np.sort(minute_total_e)[-3:].sum()
    share_top3 = top3_e / sum_total_e

    # Frequency band group energies:
    # Low: bands 0-8 (31.6 to 199.5 Hz)
    # Mid: bands 9-16 (251.2 to 1258.9 Hz)
    # High: bands 17-24 (1584.9 to 7943.3 Hz)
    low_e = lin_energy[:, 0:9].sum(axis=1)
    mid_e = lin_energy[:, 9:17].sum(axis=1)
    high_e = lin_energy[:, 17:25].sum(axis=1)

    low_share = np.maximum(0, low_e - np.percentile(low_e, 15)).sum() / (low_e.sum() + 1e-12)
    mid_share = np.maximum(0, mid_e - np.percentile(mid_e, 15)).sum() / (mid_e.sum() + 1e-12)
    high_share = np.maximum(0, high_e - np.percentile(high_e, 15)).sum() / (high_e.sum() + 1e-12)

    low_to_high_ratio = (low_e.sum() + 1e-12) / (high_e.sum() + 1e-12)

    # dB statistics across the entire 15x25 window
    mean_db = np.mean(mat)
    std_db = np.std(mat)
    max_db = np.max(mat)
    min_db = np.min(mat)
    db_range = max_db - min_db

    # Bell-curve CPA fit score: correlation with Gaussian peak centered at peak_idx
    t = np.arange(15)
    sigma = 2.5
    gaussian_template = np.exp(-0.5 * ((t - peak_idx) / sigma) ** 2)
    # Pearson correlation with normalized minute energy
    norm_e = minute_total_e - np.mean(minute_total_e)
    norm_g = gaussian_template - np.mean(gaussian_template)
    cov = np.sum(norm_e * norm_g)
    denom = np.sqrt(np.sum(norm_e ** 2) * np.sum(norm_g ** 2)) + 1e-12
    cpa_corr = cov / denom

    features = [
        share_q10, share_q15, share_q20, share_min, share_med,
        ratio_max_min, ratio_max_q15, ratio_max_med,
        log_ratio_max_min, log_ratio_max_q15,
        peak_idx, peak_dist_center, std_e, cv_e, share_top3,
        low_share, mid_share, high_share, low_to_high_ratio,
        mean_db, std_db, max_db, min_db, db_range, cpa_corr
    ]
    return np.array(features, dtype=np.float32)

print("Feature extractor defined successfully.")
