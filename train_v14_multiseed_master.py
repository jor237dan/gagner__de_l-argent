"""
train_v14_multiseed_master.py
=============================
Pure Multi-Seed Ensemble of the proven v14 CPA Boost architecture (best score 0.807917815)
- Exact same 25 physics features as v14 (no noisy features added)
- Exact same 6,000 balanced South African background windows (600 per station)
- Exact same Beta(1.0, 3.5) vessel share and physical CPA simulation
- 5 independent simulation seeds (42, 100, 2024, 777, 999) to cancel Monte Carlo simulation noise
- Calibrated with the exact proven v14 scaling factor (0.601728)
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

def lorentzian(t, A, t0, sigma, B):
    return A / (1.0 + ((t - t0) / (sigma + 1e-4)) ** 2) + B

def extract_physics_cpa_features(w_lin):
    """Exact 25 features from v14."""
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

def main():
    start_time = time.time()
    print("=== Step 1: Loading Balanced Background Windows ===")
    parquet_files = sorted(glob.glob('acoustics/*.parquet'))
    raw_background_windows = []
    
    for pf in parquet_files:
        station_name = os.path.basename(pf).replace('.parquet', '')
        t = pq.read_table(pf, columns=TOL_COLS)
        df_vals = t.to_pandas().to_numpy(dtype=np.float32)
        n_rows = len(df_vals)
        
        station_windows = 0
        step = 25
        for s in range(0, min(n_rows - 15, 120000), step):
            chunk = df_vals[s:s+15]
            if np.isnan(chunk).mean() < 0.15:
                lin_chunk = 10.0 ** (chunk / 10.0)
                if np.isfinite(lin_chunk).all() and (lin_chunk > 0).all():
                    raw_background_windows.append(lin_chunk)
                    station_windows += 1
                    if station_windows >= 600:
                        break
        print(f"Station {station_name:25s}: {station_windows} windows")
        
    print(f"\nTotal raw background windows: {len(raw_background_windows):,}")

    # -------------------------------------------------------------
    # Step 2: Extract Test Features
    # -------------------------------------------------------------
    print("\n=== Step 2: Extracting Test Features ===")
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
        
    X_test = np.nan_to_num(np.array(X_test_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"X_test shape: {X_test.shape}")

    # -------------------------------------------------------------
    # Step 3: Multi-Seed Ensemble Training (5 Seeds)
    # -------------------------------------------------------------
    SEEDS = [42, 100, 2024, 777, 999]
    all_test_preds = []
    all_class_preds = []
    
    t_vec = np.arange(15, dtype=np.float64)[:, None]
    
    for s_idx, seed in enumerate(SEEDS):
        seed_t0 = time.time()
        np.random.seed(seed)
        print(f"\n--- Running Seed {seed} ({s_idx+1}/{len(SEEDS)}) ---")
        
        X_train_list = []
        y_train_list = []
        y_class_list = []
        
        for idx, bg_lin in enumerate(raw_background_windows):
            w_bg = bg_lin.copy()
            for b in range(25):
                vals = w_bg[:, b]
                m = np.isnan(vals)
                if m.any():
                    w_bg[m, b] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 1e4
                    
            bg_tot = w_bg.sum()
            has_vessel = (np.random.rand() < 0.50)
            
            if not has_vessel:
                mixture_lin = w_bg.copy()
                target_share = 0.0
                vessel_class = 0
            else:
                t0 = np.random.uniform(1.5, 13.5)
                sigma = np.random.uniform(0.8, 3.5)
                if np.random.rand() < 0.6:
                    time_profile = 1.0 / (1.0 + ((t_vec - t0) / sigma) ** 2)
                else:
                    time_profile = np.exp(-0.5 * ((t_vec - t0) / sigma) ** 2)
                    
                freq_idx = np.arange(25, dtype=np.float64)
                center_band = np.random.uniform(3.0, 10.0)
                band_w = np.random.uniform(2.0, 5.0)
                f_spectrum = np.exp(-0.5 * ((freq_idx - center_band) / band_w) ** 2)
                
                f_spectrum[:7] += np.random.uniform(0.3, 1.2)
                f_spectrum[12:] += np.random.uniform(0.01, 0.15)
                f_spectrum = f_spectrum / f_spectrum.sum()
                
                vessel_pattern = time_profile * f_spectrum[None, :]
                
                sampled_share = float(np.random.beta(1.0, 3.5))
                sampled_share = np.clip(sampled_share, 0.005, 0.85)
                
                desired_vessel_e = bg_tot * sampled_share / (1.0 - sampled_share + 1e-9)
                vessel_lin = vessel_pattern * (desired_vessel_e / (vessel_pattern.sum() + 1e-9))
                
                mixture_lin = w_bg + vessel_lin
                actual_v_tot = vessel_lin.sum()
                actual_m_tot = mixture_lin.sum()
                target_share = float(actual_v_tot / actual_m_tot)
                vessel_class = 1
                
            mixture_db = 10.0 * np.log10(mixture_lin + 1e-12)
            nan_mask = np.random.rand(*mixture_db.shape) < 0.025
            mixture_db[nan_mask] = np.nan
            
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
            
        X_train = np.nan_to_num(np.array(X_train_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.array(y_train_list, dtype=np.float32)
        y_class = np.array(y_class_list, dtype=np.int32)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        seed_test_preds = np.zeros(len(test_df), dtype=np.float32)
        seed_class_preds = np.zeros(len(test_df), dtype=np.float32)
        
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
            'random_state': seed,
            'verbose': -1,
            'n_jobs': -1
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
            'random_state': seed,
            'verbose': -1,
            'n_jobs': -1
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va, y_val = X_train[val_idx], y_train[val_idx]
            y_tr_c, y_val_c = y_class[train_idx], y_class[val_idx]
            
            reg = lgb.LGBMRegressor(**lgb_reg_params)
            reg.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
            seed_test_preds += reg.predict(X_test) / kf.n_splits
            
            clf = lgb.LGBMClassifier(**lgb_clf_params)
            clf.fit(X_tr, y_tr_c, eval_set=[(X_va, y_val_c)], callbacks=[lgb.early_stopping(50, verbose=False)])
            seed_class_preds += clf.predict_proba(X_test)[:, 1] / kf.n_splits
            
        all_test_preds.append(seed_test_preds)
        all_class_preds.append(seed_class_preds)
        print(f"Seed {seed} finished in {time.time() - seed_t0:.1f}s | Raw test mean={seed_test_preds.mean():.4f}, std={seed_test_preds.std():.4f}")
        
    # -------------------------------------------------------------
    # Step 4: Multi-Seed Averaging & Exact Calibration
    # -------------------------------------------------------------
    ens_test_preds = np.mean(all_test_preds, axis=0)
    ens_class_preds = np.mean(all_class_preds, axis=0)
    
    print(f"\nEnsemble raw mean: {ens_test_preds.mean():.4f}, std: {ens_test_preds.std():.4f}")
    
    # Exact calibration factor proven by v14_calibrated:
    scaling_k = 0.601728
    calibrated_share = np.clip(np.round(ens_test_preds * scaling_k, 5), 0.0, 1.0)
    calibrated_score = np.clip(np.round(ens_class_preds, 5), 0.0, 1.0)
    
    # Check correlation with the original v14_calibrated
    v14_orig = pd.read_csv('submission_v14_calibrated.csv')['share_added_vessel'].to_numpy()
    corr = np.corrcoef(v14_orig, calibrated_share)[0, 1]
    print(f"\nCorrelation with original v14_calibrated (0.8079): {corr:.5f}")
    print(f"Mean abs diff with v14_calibrated: {np.abs(v14_orig - calibrated_share).mean():.5f}")
    
    # Save standalone multi-seed ensemble
    sub_ens = pd.DataFrame({
        'ID': test_df['ID'],
        'share_added_vessel': calibrated_share,
        'score_added_vessel': calibrated_score
    })
    sub_ens.to_csv('submission_v14_multiseed_master.csv', index=False)
    print("Saved submission_v14_multiseed_master.csv")
    
    # Also save a 50/50 smooth blend with v14_calibrated (ultra-safe)
    smooth_share = np.clip(np.round(0.50 * v14_orig + 0.50 * calibrated_share, 5), 0.0, 1.0)
    sub_smooth = pd.DataFrame({
        'ID': test_df['ID'],
        'share_added_vessel': smooth_share,
        'score_added_vessel': calibrated_score
    })
    sub_smooth.to_csv('submission_v14_smooth_blend.csv', index=False)
    print("Saved submission_v14_smooth_blend.csv")
    
    print(f"\nTotal elapsed time: {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
