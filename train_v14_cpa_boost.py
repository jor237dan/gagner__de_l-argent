"""
train_v14_cpa_boost.py
======================
Physics-Informed Acoustic CPA Feature Extraction & LightGBM Pipeline
- Real South African background windows from parquet files
- Physical Lorentzian CPA passage modeling & inter-band temporal synchrony
- Distribution-calibrated target share (50% negative samples, mean ~0.10)
- 5-Fold Stratified/K-Fold LightGBM regressor
- Evaluates out-of-fold RMSE and generates submission_v14_cpa_boost.csv
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# 25 third-octave bands
BANDS = ['31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5',
         '251.2','316.2','398.1','501.2','631','794.3','1000','1258.9',
         '1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3']
TOL_COLS = [f'TOL_{b}' for b in BANDS]

# -------------------------------------------------------------
# 1. Physics-based CPA Feature Extractor
# -------------------------------------------------------------
def lorentzian(t, A, t0, sigma, B):
    return A / (1.0 + ((t - t0) / (sigma + 1e-4)) ** 2) + B

def extract_physics_cpa_features(w_lin):
    """
    w_lin: shape (15, 25) in linear acoustic energy (10^(dB/10)), NaNs imputed.
    Returns: 1D array of 25 highly discriminative acoustic features.
    """
    t = np.arange(15, dtype=np.float64)
    min_e = w_lin.sum(axis=1) # (15,)
    tot_e = np.sum(min_e) + 1e-12
    
    # 1. Energy Quantiles & Simple Subtraction Proxies
    q10 = np.percentile(min_e, 10)
    q20 = np.percentile(min_e, 20)
    med_val = np.median(min_e)
    min_val = np.min(min_e)
    max_val = np.max(min_e)
    
    share_min = np.sum(np.maximum(0.0, min_e - min_val)) / tot_e
    share_q10 = np.sum(np.maximum(0.0, min_e - q10)) / tot_e
    share_q20 = np.sum(np.maximum(0.0, min_e - q20)) / tot_e
    share_med = np.sum(np.maximum(0.0, min_e - med_val)) / tot_e
    
    # 2. Dynamic Contrast Features
    ratio_max_min = max_val / (min_val + 1e-12)
    ratio_max_q10 = max_val / (q10 + 1e-12)
    log_dyn_range = np.log10(ratio_max_min + 1.0)
    top2_share = np.sum(np.sort(min_e)[-2:]) / tot_e
    top3_share = np.sum(np.sort(min_e)[-3:]) / tot_e
    cv_energy = np.std(min_e) / (np.mean(min_e) + 1e-12)
    
    # 3. CPA Lorentzian Curve Fit
    B0 = q20
    A0 = max(0.0, max_val - B0)
    t0_0 = float(np.argmax(min_e))
    try:
        popt, _ = curve_fit(lorentzian, t, min_e, p0=[A0, t0_0, 2.0, B0],
                            bounds=([0, 0, 0.4, 0], [np.inf, 14, 8, np.inf]),
                            maxfev=150)
        A, t0, sigma, B = popt
        pred = lorentzian(t, A, t0, sigma, B)
        ss_res = np.sum((min_e - pred) ** 2)
        ss_tot = np.sum((min_e - np.mean(min_e)) ** 2) + 1e-12
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        vessel_e = np.sum(np.maximum(0.0, pred - B))
        fitted_share = min(1.0, vessel_e / tot_e)
        snr = A / (B + 1e-9)
        cpa_dist_center = abs(t0 - 7.0)
    except Exception:
        r2, snr, t0, sigma, fitted_share, cpa_dist_center = 0.0, 0.0, t0_0, 2.0, 0.0, abs(t0_0 - 7.0)
        
    # 4. Multi-Band CPA Synchrony
    band_peaks = np.argmax(w_lin, axis=0) # (25,)
    peak_std = float(np.std(band_peaks))
    peak_diff_cpa = float(np.mean(np.abs(band_peaks - t0)))
    
    corrs = []
    for b in range(25):
        s = np.std(w_lin[:, b])
        if s > 1e-9:
            c = np.corrcoef(min_e, w_lin[:, b])[0, 1]
            if np.isfinite(c):
                corrs.append(c)
    mean_band_corr = float(np.mean(corrs)) if corrs else 0.0
    
    # 5. Band-wise Stationary Background Subtraction
    quiet_idx = np.argsort(min_e)[:3]
    bg_spectrum = w_lin[quiet_idx].mean(axis=0, keepdims=True) # (1, 25)
    excess_matrix = np.maximum(0.0, w_lin - bg_spectrum)
    band_excess_share = float(excess_matrix.sum() / tot_e)
    
    # Low / Mid / High group shares
    low_excess_share = float(excess_matrix[:, :13].sum() / tot_e)
    mid_excess_share = float(excess_matrix[:, 13:17].sum() / tot_e)
    high_excess_share = float(excess_matrix[:, 17:].sum() / tot_e)
    ratio_low_high = (low_excess_share + 1e-6) / (high_excess_share + 1e-6)
    
    # 6. Peak-Minute Spectral Tilt & Centroid
    peak_idx = int(np.clip(round(t0), 0, 14))
    peak_spectrum = w_lin[peak_idx]
    tilt_peak = (peak_spectrum[:8].sum() + 1e-9) / (peak_spectrum[17:].sum() + 1e-9)
    quiet_spectrum = bg_spectrum[0]
    tilt_quiet = (quiet_spectrum[:8].sum() + 1e-9) / (quiet_spectrum[17:].sum() + 1e-9)
    tilt_contrast = np.log10(tilt_peak / (tilt_quiet + 1e-12) + 1.0)
    
    return [
        share_min, share_q10, share_q20, share_med,
        ratio_max_min, ratio_max_q10, log_dyn_range, top2_share, top3_share, cv_energy,
        r2, snr, t0, sigma, fitted_share, cpa_dist_center,
        peak_std, peak_diff_cpa, mean_band_corr,
        band_excess_share, low_excess_share, mid_excess_share, high_excess_share,
        ratio_low_high, tilt_contrast
    ]

FEATURE_NAMES = [
    'share_min', 'share_q10', 'share_q20', 'share_med',
    'ratio_max_min', 'ratio_max_q10', 'log_dyn_range', 'top2_share', 'top3_share', 'cv_energy',
    'r2_cpa', 'snr_cpa', 't0_cpa', 'sigma_cpa', 'fitted_share', 'cpa_dist_center',
    'peak_std', 'peak_diff_cpa', 'mean_band_corr',
    'band_excess_share', 'low_excess_share', 'mid_excess_share', 'high_excess_share',
    'ratio_low_high', 'tilt_contrast'
]

# -------------------------------------------------------------
# 2. Load Real Background Windows from Parquet Files
# -------------------------------------------------------------
print("=== 1. Loading diverse South African background measurements ===")
parquet_files = sorted(glob.glob('acoustics/*.parquet'))
raw_background_windows = []

for pf in parquet_files:
    station_name = os.path.basename(pf).replace('.parquet', '')
    t = pq.read_table(pf, columns=TOL_COLS)
    df_vals = t.to_pandas().to_numpy(dtype=np.float32)
    n_rows = len(df_vals)
    print(f"Station {station_name:25s}: {n_rows:,} rows")
    
    # Sample consecutive 15-min windows with valid data
    station_windows = 0
    step = 25
    for start in range(0, min(n_rows - 15, 120000), step):
        chunk = df_vals[start:start+15]
        if np.isnan(chunk).mean() < 0.15:
            lin_chunk = 10.0 ** (chunk / 10.0)
            if np.isfinite(lin_chunk).all() and (lin_chunk > 0).all():
                raw_background_windows.append(lin_chunk)
                station_windows += 1
                if station_windows >= 600:
                    break
    print(f"  -> Extracted {station_windows} clean windows")

print(f"\nTotal collected real background windows: {len(raw_background_windows):,}")

# -------------------------------------------------------------
# 3. Generate Realistic Simulation Dataset
# -------------------------------------------------------------
print("\n=== 2. Generating realistic calibrated simulation dataset ===")
np.random.seed(42)

X_train_list = []
y_train_list = []
y_class_list = [] # 1 if vessel added, 0 otherwise

num_samples = len(raw_background_windows)
# 50% clean background (y = 0.0), 50% with added vessel
t_vec = np.arange(15, dtype=np.float64)[:, None] # (15, 1)

for idx, bg_lin in enumerate(raw_background_windows):
    w_bg = bg_lin.copy()
    for b in range(25):
        vals = w_bg[:, b]
        m = np.isnan(vals)
        if m.any():
            w_bg[m, b] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 1e4
            
    bg_tot = w_bg.sum()
    
    # 50% probability of positive vessel addition
    has_vessel = (np.random.rand() < 0.50)
    
    if not has_vessel:
        mixture_lin = w_bg.copy()
        target_share = 0.0
        vessel_class = 0
    else:
        # Simulate realistic vessel passage
        # 1. Time trajectory: CPA Lorentzian or Gaussian
        t0 = np.random.uniform(1.5, 13.5)
        sigma = np.random.uniform(0.8, 3.5)
        if np.random.rand() < 0.6:
            # Lorentzian
            time_profile = 1.0 / (1.0 + ((t_vec - t0) / sigma) ** 2)
        else:
            # Gaussian
            time_profile = np.exp(-0.5 * ((t_vec - t0) / sigma) ** 2)
            
        # 2. Vessel frequency spectrum S(b)
        # Machinery blade harmonics + cavitation hump (peaking in bands 3 to 11)
        freq_idx = np.arange(25, dtype=np.float64)
        center_band = np.random.uniform(3.0, 10.0)
        band_w = np.random.uniform(2.0, 5.0)
        f_spectrum = np.exp(-0.5 * ((freq_idx - center_band) / band_w) ** 2)
        
        # Heavy low-frequency rumble (bands 0-6)
        f_spectrum[:7] += np.random.uniform(0.3, 1.2)
        # Cavitation broadband tail (high frequencies)
        f_spectrum[12:] += np.random.uniform(0.01, 0.15)
        f_spectrum = f_spectrum / f_spectrum.sum()
        
        vessel_pattern = time_profile * f_spectrum[None, :] # (15, 25)
        
        # 3. Sample realistic target share:
        # Beta(1.0, 3.5) has mean = 1/(1+3.5) = 0.22, range [0.01, 0.70]
        # Overall mixture mean with 50% zeros will be ~ 0.5 * 0.22 = 0.11
        sampled_share = float(np.random.beta(1.0, 3.5))
        sampled_share = np.clip(sampled_share, 0.005, 0.85)
        
        # Calculate required vessel energy
        desired_vessel_e = bg_tot * sampled_share / (1.0 - sampled_share + 1e-9)
        vessel_lin = vessel_pattern * (desired_vessel_e / (vessel_pattern.sum() + 1e-9))
        
        mixture_lin = w_bg + vessel_lin
        actual_v_tot = vessel_lin.sum()
        actual_m_tot = mixture_lin.sum()
        target_share = float(actual_v_tot / actual_m_tot)
        vessel_class = 1
        
    # Convert to dB, add ~2.5% realistic NaNs
    mixture_db = 10.0 * np.log10(mixture_lin + 1e-12)
    nan_mask = np.random.rand(*mixture_db.shape) < 0.025
    mixture_db[nan_mask] = np.nan
    
    # Re-impute and linearize for feature extraction (matching test pipeline)
    mixture_lin_clean = 10.0 ** (mixture_db / 10.0)
    for b in range(25):
        vals = mixture_lin_clean[:, b]
        m = np.isnan(vals)
        if m.any():
            med = np.nanmedian(vals)
            mixture_lin_clean[m, b] = med if np.isfinite(med) else 1e4
            
    feats = extract_physics_cpa_features(mixture_lin_clean)
    X_train_list.append(feats)
    y_train_list.append(target_share)
    y_class_list.append(vessel_class)

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)
y_class = np.array(y_class_list, dtype=np.int32)

print(f"X_train shape: {X_train.shape}")
print(f"y_train statistics: mean={y_train.mean():.4f}, std={y_train.std():.4f}, min={y_train.min():.4f}, max={y_train.max():.4f}")
print(f"Zero targets (clean background): {(y_train == 0).mean()*100:.1f}%")
print(f"Positive targets mean: {y_train[y_train > 0].mean():.4f}")

# -------------------------------------------------------------
# 4. Extract Features on Test.csv
# -------------------------------------------------------------
print("\n=== 3. Extracting physics CPA features on Test.csv ===")
test_df = pd.read_csv('Test.csv')
test_feat_cols = [c for c in test_df.columns if c != 'ID']
test_arr = test_df[test_feat_cols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
test_lin = 10.0 ** (test_arr / 10.0)

X_test_list = []
for i in range(len(test_lin)):
    w = test_lin[i].copy()
    for b in range(25):
        vals = w[:, b]
        m = np.isnan(vals)
        if m.any():
            med = np.nanmedian(vals)
            w[m, b] = med if np.isfinite(med) else 1e4
    feats = extract_physics_cpa_features(w)
    X_test_list.append(feats)

X_test = np.array(X_test_list, dtype=np.float32)
print(f"X_test shape: {X_test.shape}")

# -------------------------------------------------------------
# 5. Train LightGBM with 5-Fold Cross Validation
# -------------------------------------------------------------
print("\n=== 4. Training LightGBM 5-Fold Ensemble ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y_train))
oof_class_preds = np.zeros(len(y_train))
test_preds = np.zeros(len(test_df))
test_class_preds = np.zeros(len(test_df))

lgb_reg_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 600,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

lgb_clf_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 400,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_val = X_train[val_idx], y_train[val_idx]
    y_tr_c, y_val_c = y_class[train_idx], y_class[val_idx]
    
    # 1. Regressor for share_added_vessel
    reg = lgb.LGBMRegressor(**lgb_reg_params)
    reg.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    val_pred = reg.predict(X_va)
    oof_preds[val_idx] = val_pred
    test_preds += reg.predict(X_test) / kf.n_splits
    
    # 2. Classifier for score_added_vessel
    clf = lgb.LGBMClassifier(**lgb_clf_params)
    clf.fit(X_tr, y_tr_c, eval_set=[(X_va, y_val_c)], callbacks=[lgb.early_stopping(50, verbose=False)])
    val_c_pred = clf.predict_proba(X_va)[:, 1]
    oof_class_preds[val_idx] = val_c_pred
    test_class_preds += clf.predict_proba(X_test)[:, 1] / kf.n_splits
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, np.clip(val_pred, 0.0, 1.0)))
    print(f"Fold {fold+1}: RMSE={fold_rmse:.4f} | 1-RMSE={1.0 - fold_rmse:.4f}")

oof_rmse = np.sqrt(mean_squared_error(y_train, np.clip(oof_preds, 0.0, 1.0)))
oof_score = 1.0 - oof_rmse
print(f"\n--- Out-of-Fold Evaluation ---")
print(f"OOF RMSE  : {oof_rmse:.5f}")
print(f"OOF Score (1 - RMSE): {oof_score:.5f}")

# Feature importance
importances = reg.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print("\nTop 10 Most Important Features:")
for i in top_idx:
    print(f"  {FEATURE_NAMES[i]:20s}: {importances[i]}")

# -------------------------------------------------------------
# 6. Post-Processing & Calibration
# -------------------------------------------------------------
print("\n=== 5. Predictions Calibration and Validation ===")
# Clip predictions
test_share = np.clip(test_preds, 0.0, 1.0)
test_score = np.clip(test_class_preds, 0.0, 1.0)

print(f"Raw test_share: mean={test_share.mean():.4f}, std={test_share.std():.4f}, min={test_share.min():.4f}, max={test_share.max():.4f}")
print(f"Test score (evidence): mean={test_score.mean():.4f}, std={test_score.std():.4f}")

# Correlation with our previous best v4
v4 = pd.read_csv('submission_v4.csv')['share_added_vessel'].to_numpy()
corr_v4 = np.corrcoef(v4, test_share)[0, 1]
print(f"\nCorrelation between new model and v4 (best 0.79018): {corr_v4:.4f}")

# -------------------------------------------------------------
# 7. Generate Submission File
# -------------------------------------------------------------
output_file = 'submission_v14_cpa_boost.csv'
sub = pd.DataFrame({
    'ID': test_df['ID'],
    'share_added_vessel': np.round(test_share, 5),
    'score_added_vessel': np.round(test_score, 5)
})
sub.to_csv(output_file, index=False)
print(f"\nSaved {output_file} successfully ({len(sub)} rows)!")
print(sub.head(10))
