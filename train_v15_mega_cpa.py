"""
train_v15_mega_cpa.py
======================
Next-Generation Physics-Informed Pipeline:
1. 10,000+ real background windows from South African acoustics
2. Two-Stage Hurdle Architecture (Binary Classifier + Energy Regressor)
3. 30 High-Value Physics Acoustic Features (CPA Lorentzian & Gaussian, Multi-band Synchrony, Kurtosis)
4. Multi-Model 5-Fold Ensemble (LightGBM + HistGradientBoosting)
5. Zero-Noise Hurdle Suppression & Distribution Calibration
6. Generates submission_v15_mega_cpa.csv
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from scipy.stats import kurtosis
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import lightgbm as lgb

# 25 third-octave bands
BANDS = ['31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5',
         '251.2','316.2','398.1','501.2','631','794.3','1000','1258.9',
         '1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3']
TOL_COLS = [f'TOL_{b}' for b in BANDS]

# -------------------------------------------------------------
# 1. Advanced Physics-based Acoustic Feature Extractor (30 Features)
# -------------------------------------------------------------
def lorentzian(t, A, t0, sigma, B):
    return A / (1.0 + ((t - t0) / (sigma + 1e-4)) ** 2) + B

def gaussian_fn(t, A, t0, sigma, B):
    return A * np.exp(-0.5 * ((t - t0) / (sigma + 1e-4)) ** 2) + B

def extract_physics_cpa_features(w_lin):
    """
    w_lin: shape (15, 25) in linear acoustic energy (10^(dB/10)), NaNs imputed.
    Returns: 1D array of 30 discriminative acoustic features.
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
    
    share_min = float(np.sum(np.maximum(0.0, min_e - min_val)) / tot_e)
    share_q10 = float(np.sum(np.maximum(0.0, min_e - q10)) / tot_e)
    share_q20 = float(np.sum(np.maximum(0.0, min_e - q20)) / tot_e)
    share_med = float(np.sum(np.maximum(0.0, min_e - med_val)) / tot_e)
    
    # 2. Dynamic Contrast Features
    ratio_max_min = float(max_val / (min_val + 1e-12))
    ratio_max_q10 = float(max_val / (q10 + 1e-12))
    log_dyn_range = float(np.log10(ratio_max_min + 1.0))
    top2_share = float(np.sum(np.sort(min_e)[-2:]) / tot_e)
    top3_share = float(np.sum(np.sort(min_e)[-3:]) / tot_e)
    cv_energy = float(np.std(min_e) / (np.mean(min_e) + 1e-12))
    
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
        r2_cpa = float(max(0.0, 1.0 - ss_res / ss_tot))
        vessel_e = np.sum(np.maximum(0.0, pred - B))
        fitted_share = float(min(1.0, vessel_e / tot_e))
        snr_cpa = float(A / (B + 1e-9))
        cpa_dist_center = float(abs(t0 - 7.0))
    except Exception:
        r2_cpa, snr_cpa, t0, sigma, fitted_share, cpa_dist_center = 0.0, 0.0, t0_0, 2.0, 0.0, float(abs(t0_0 - 7.0))
        
    # Gaussian Fit
    try:
        popt_g, _ = curve_fit(gaussian_fn, t, min_e, p0=[A0, t0_0, 1.8, B0],
                              bounds=([0, 0, 0.4, 0], [np.inf, 14, 8, np.inf]),
                              maxfev=120)
        pred_g = gaussian_fn(t, *popt_g)
        ss_res_g = np.sum((min_e - pred_g) ** 2)
        ss_tot_g = np.sum((min_e - np.mean(min_e)) ** 2) + 1e-12
        r2_gauss = float(max(0.0, 1.0 - ss_res_g / ss_tot_g))
        fitted_share_gauss = float(min(1.0, np.sum(np.maximum(0.0, pred_g - popt_g[3])) / tot_e))
    except Exception:
        r2_gauss, fitted_share_gauss = 0.0, 0.0
        
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
    ratio_low_high = float((low_excess_share + 1e-6) / (high_excess_share + 1e-6))
    
    # Lowest 3 bands share (propeller direct tones: 31.6, 39.8, 50.1 Hz)
    propeller_low3_share = float(excess_matrix[:, :3].sum() / tot_e)
    
    # 6. Peak-Minute Spectral Tilt & Centroid Contrast
    peak_idx = int(np.clip(round(t0), 0, 14))
    peak_spectrum = w_lin[peak_idx]
    tilt_peak = (peak_spectrum[:8].sum() + 1e-9) / (peak_spectrum[17:].sum() + 1e-9)
    quiet_spectrum = bg_spectrum[0]
    tilt_quiet = (quiet_spectrum[:8].sum() + 1e-9) / (quiet_spectrum[17:].sum() + 1e-9)
    tilt_contrast = float(np.log10(tilt_peak / (tilt_quiet + 1e-12) + 1.0))
    
    # 7. Higher-Order Shape Statistics (Kurtosis)
    kurt_time = float(kurtosis(min_e))
    kurt_spectrum = float(kurtosis(peak_spectrum))
    
    return [
        share_min, share_q10, share_q20, share_med,
        ratio_max_min, ratio_max_q10, log_dyn_range, top2_share, top3_share, cv_energy,
        r2_cpa, snr_cpa, t0, sigma, fitted_share, cpa_dist_center,
        r2_gauss, fitted_share_gauss,
        peak_std, peak_diff_cpa, mean_band_corr,
        band_excess_share, low_excess_share, mid_excess_share, high_excess_share,
        ratio_low_high, propeller_low3_share, tilt_contrast,
        kurt_time, kurt_spectrum
    ]

# -------------------------------------------------------------
# 2. Load 10,000+ Background Windows Across Stations
# -------------------------------------------------------------
print("=== 1. Loading 10,000+ South African background measurements ===")
parquet_files = sorted(glob.glob('acoustics/*.parquet'))
raw_background_windows = []

# Allocate per-station quota based on file size
quotas = {
    '01_Richards_Bay': 2200,
    '02_Durban': 1200,
    '03_Cape_Recife': 300,
    '04_Bird_Island': 1200,
    '05_St_Croix_Island': 1500,
    '06_Riy_Banks': 200,
    '07_St_Francis': 1500,
    '09_Fish_Hoek': 1200,
    '10_Sea_Point': 200,
    '13_Saldhana_Bay': 1500,
}

for pf in parquet_files:
    sname = os.path.basename(pf).replace('.parquet', '')
    max_q = quotas.get(sname, 1000)
    t = pq.read_table(pf, columns=TOL_COLS)
    df_vals = t.to_pandas().to_numpy(dtype=np.float32)
    n_rows = len(df_vals)
    
    station_count = 0
    step = 20
    for start in range(0, min(n_rows - 15, 200000), step):
        chunk = df_vals[start:start+15]
        if np.isnan(chunk).mean() < 0.15:
            lin_chunk = 10.0 ** (chunk / 10.0)
            if np.isfinite(lin_chunk).all() and (lin_chunk > 0).all():
                raw_background_windows.append(lin_chunk)
                station_count += 1
                if station_count >= max_q:
                    break
    print(f"Station {sname:25s}: extracted {station_count:,} windows (target {max_q})")

print(f"\nTotal background windows collected: {len(raw_background_windows):,}")

# -------------------------------------------------------------
# 3. Generate Calibrated Simulation Dataset (Hurdle Labels)
# -------------------------------------------------------------
print("\n=== 2. Generating simulation dataset with hurdle labels ===")
np.random.seed(42)

X_train_list = []
y_share_list = []
y_class_list = []

t_vec = np.arange(15, dtype=np.float64)[:, None]

for bg_lin in raw_background_windows:
    w_bg = bg_lin.copy()
    for b in range(25):
        vals = w_bg[:, b]
        m = np.isnan(vals)
        if m.any():
            w_bg[m, b] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 1e4
            
    bg_tot = w_bg.sum()
    
    # 52% clean background (y = 0.0), 48% vessel passage
    has_vessel = (np.random.rand() < 0.48)
    
    if not has_vessel:
        mixture_lin = w_bg.copy()
        target_share = 0.0
        vessel_class = 0
    else:
        # CPA Time profile (mix of Lorentzian and Gaussian)
        t0 = np.random.uniform(1.2, 13.8)
        sigma = np.random.uniform(0.7, 3.8)
        if np.random.rand() < 0.55:
            time_profile = 1.0 / (1.0 + ((t_vec - t0) / sigma) ** 2)
        else:
            time_profile = np.exp(-0.5 * ((t_vec - t0) / sigma) ** 2)
            
        # Realistic multi-band vessel spectrum
        freq_idx = np.arange(25, dtype=np.float64)
        center_band = np.random.uniform(2.5, 10.5)
        band_w = np.random.uniform(1.8, 5.2)
        f_spectrum = np.exp(-0.5 * ((freq_idx - center_band) / band_w) ** 2)
        
        # Low frequency diesel machinery lines
        f_spectrum[:7] += np.random.uniform(0.25, 1.4)
        # Cavitation broadband tail
        f_spectrum[12:] += np.random.uniform(0.01, 0.16)
        f_spectrum = f_spectrum / f_spectrum.sum()
        
        vessel_pattern = time_profile * f_spectrum[None, :]
        
        # Sample realistic target share (Beta(1.0, 3.8) -> mean ~ 0.208 for positive)
        sampled_share = float(np.random.beta(1.0, 3.8))
        sampled_share = np.clip(sampled_share, 0.005, 0.82)
        
        desired_vessel_e = bg_tot * sampled_share / (1.0 - sampled_share + 1e-9)
        vessel_lin = vessel_pattern * (desired_vessel_e / (vessel_pattern.sum() + 1e-9))
        
        mixture_lin = w_bg + vessel_lin
        target_share = float(vessel_lin.sum() / mixture_lin.sum())
        vessel_class = 1
        
    # Convert to dB with realistic missing values
    mixture_db = 10.0 * np.log10(mixture_lin + 1e-12)
    nan_mask = np.random.rand(*mixture_db.shape) < 0.025
    mixture_db[nan_mask] = np.nan
    
    # Re-impute
    mixture_clean = 10.0 ** (mixture_db / 10.0)
    for b in range(25):
        vals = mixture_clean[:, b]
        m = np.isnan(vals)
        if m.any():
            med = np.nanmedian(vals)
            mixture_clean[m, b] = med if np.isfinite(med) else 1e4
            
    feats = extract_physics_cpa_features(mixture_clean)
    X_train_list.append(feats)
    y_share_list.append(target_share)
    y_class_list.append(vessel_class)

X_train = np.array(X_train_list, dtype=np.float32)
y_share = np.array(y_share_list, dtype=np.float32)
y_class = np.array(y_class_list, dtype=np.int32)

print(f"X_train shape: {X_train.shape}")
print(f"y_share: mean={y_share.mean():.4f}, std={y_share.std():.4f}, zeros={(y_share == 0).mean()*100:.1f}%")

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
# 5. Multi-Model 5-Fold Training (LightGBM + HistGBR)
# -------------------------------------------------------------
print("\n=== 4. Training 5-Fold Multi-Model Ensemble ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(y_share))
oof_preds_hgb = np.zeros(len(y_share))
oof_prob_lgb = np.zeros(len(y_share))
oof_prob_hgb = np.zeros(len(y_share))

test_preds_lgb = np.zeros(len(test_df))
test_preds_hgb = np.zeros(len(test_df))
test_prob_lgb = np.zeros(len(test_df))
test_prob_hgb = np.zeros(len(test_df))

lgb_reg_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 800,
    'learning_rate': 0.025,
    'num_leaves': 35,
    'min_child_samples': 25,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

lgb_clf_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 500,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'min_child_samples': 25,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_share)):
    X_tr, y_tr_s, y_tr_c = X_train[train_idx], y_share[train_idx], y_class[train_idx]
    X_va, y_va_s, y_va_c = X_train[val_idx], y_share[val_idx], y_class[val_idx]
    
    # 1. LightGBM Regressor
    reg_lgb = lgb.LGBMRegressor(**lgb_reg_params)
    reg_lgb.fit(X_tr, y_tr_s, eval_set=[(X_va, y_va_s)], callbacks=[lgb.early_stopping(50, verbose=False)])
    val_pred_lgb = reg_lgb.predict(X_va)
    oof_preds_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += reg_lgb.predict(X_test) / kf.n_splits
    
    # 2. LightGBM Classifier
    clf_lgb = lgb.LGBMClassifier(**lgb_clf_params)
    clf_lgb.fit(X_tr, y_tr_c, eval_set=[(X_va, y_va_c)], callbacks=[lgb.early_stopping(50, verbose=False)])
    val_prob_lgb = clf_lgb.predict_proba(X_va)[:, 1]
    oof_prob_lgb[val_idx] = val_prob_lgb
    test_prob_lgb += clf_lgb.predict_proba(X_test)[:, 1] / kf.n_splits
    
    # 3. HistGradientBoosting Regressor
    reg_hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, min_samples_leaf=25, random_state=42)
    reg_hgb.fit(X_tr, y_tr_s)
    val_pred_hgb = reg_hgb.predict(X_va)
    oof_preds_hgb[val_idx] = val_pred_hgb
    test_preds_hgb += reg_hgb.predict(X_test) / kf.n_splits
    
    # 4. HistGradientBoosting Classifier
    clf_hgb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, min_samples_leaf=25, random_state=42)
    clf_hgb.fit(X_tr, y_tr_c)
    val_prob_hgb = clf_hgb.predict_proba(X_va)[:, 1]
    oof_prob_hgb[val_idx] = val_prob_hgb
    test_prob_hgb += clf_hgb.predict_proba(X_test)[:, 1] / kf.n_splits
    
    # Combined Fold Prediction (70% LightGBM + 30% HistGBR)
    val_combo = 0.70 * val_pred_lgb + 0.30 * val_pred_hgb
    val_rmse = np.sqrt(mean_squared_error(y_va_s, np.clip(val_combo, 0.0, 1.0)))
    print(f"Fold {fold+1}: RMSE={val_rmse:.4f} | Score={1.0 - val_rmse:.4f}")

# Overall OOF
oof_combo = 0.70 * oof_preds_lgb + 0.30 * oof_preds_hgb
oof_prob_combo = 0.70 * oof_prob_lgb + 0.30 * oof_prob_hgb
total_oof_rmse = np.sqrt(mean_squared_error(y_share, np.clip(oof_combo, 0.0, 1.0)))
print(f"\nTotal Multi-Model OOF Score (1 - RMSE): {1.0 - total_oof_rmse:.5f}")

# -------------------------------------------------------------
# 6. Test Predictions & Hurdle Zero-Suppression
# -------------------------------------------------------------
print("\n=== 5. Calibrating Predictions & Hurdle Zero-Suppression ===")
raw_test_pred = 0.70 * test_preds_lgb + 0.30 * test_preds_hgb
test_prob = 0.70 * test_prob_lgb + 0.30 * test_prob_hgb

# Apply Hurdle Zero-Suppression:
# If probability of vessel addition is low (< 0.18) or energy share is near zero (< 0.015), suppress to 0.0
hurdle_mask = (test_prob < 0.18) | (raw_test_pred < 0.015)
cleaned_test_pred = raw_test_pred.copy()
cleaned_test_pred[hurdle_mask] = 0.0

# Optimal calibration scale to match true E[y] ~ 0.108
scale_factor = 0.108 / cleaned_test_pred.mean()
calibrated_pred = np.clip(cleaned_test_pred * scale_factor, 0.0, 1.0)

print(f"Hurdle suppressed zeros: {hurdle_mask.sum():,}/{len(calibrated_pred)} ({hurdle_mask.mean()*100:.1f}%)")
print(f"Calibrated final share: mean={calibrated_pred.mean():.4f}, std={calibrated_pred.std():.4f}, max={calibrated_pred.max():.4f}")
print(f"Diagnostic score_added_vessel: mean={test_prob.mean():.4f}, std={test_prob.std():.4f}")

# Correlation with our newly proven best zbKNW3Cc (score 0.8079)
v14_cal = pd.read_csv('submission_v14_calibrated.csv')['share_added_vessel'].to_numpy()
corr_v14 = np.corrcoef(v14_cal, calibrated_pred)[0, 1]
print(f"\nCorrelation with v14 (our 0.8079 benchmark): {corr_v14:.4f}")

# -------------------------------------------------------------
# 7. Save Final Submission
# -------------------------------------------------------------
output_file = 'submission_v15_mega_cpa.csv'
sub = pd.DataFrame({
    'ID': test_df['ID'],
    'share_added_vessel': np.round(calibrated_pred, 5),
    'score_added_vessel': np.round(np.clip(test_prob, 0.0, 1.0), 5)
})
sub.to_csv(output_file, index=False)
print(f"\nSaved {output_file} successfully ({len(sub)} rows)!")
print(sub.head(10))
