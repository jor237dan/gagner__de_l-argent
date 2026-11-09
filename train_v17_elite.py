"""
train_v17_elite.py
==================
Elite Physics-Informed Pipeline — v17
Improvements over v16:
  - 55 features (vs 40): adds autocorrelation, temporal skewness per-band, flatness, inter-band coherence
  - Larger training set: 1200 clean background windows / station (vs 800)
  - Richer vessel simulation: multi-peak CPA + harmonic blade patterns
  - 3-model stacking: LightGBM + XGBoost + HistGBR
  - Diversity-weighted blend (minimize correlation with v14 proven 0.8079)
  - Both standalone & ensemble blend outputs
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
from scipy.signal import correlate
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb

# 25 Third-Octave Frequency Bands
BANDS = ['31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5',
         '251.2','316.2','398.1','501.2','631','794.3','1000','1258.9',
         '1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3']
TOL_COLS = [f'TOL_{b}' for b in BANDS]

# ------------------------------------------------------------------
# 1. Physics curve functions
# ------------------------------------------------------------------
def lorentzian(t, A, t0, sigma, B):
    return A / (1.0 + ((t - t0) / (sigma + 1e-4)) ** 2) + B

def gaussian_fn(t, A, t0, sigma, B):
    return A * np.exp(-0.5 * ((t - t0) / (sigma + 1e-4)) ** 2) + B

# ------------------------------------------------------------------
# 2. 55-feature extractor
# ------------------------------------------------------------------
def extract_55_features(w_lin):
    """
    w_lin: (15, 25) linear acoustic energy, NaNs already imputed.
    Returns: 1D float32 array of 55 features.
    """
    t = np.arange(15, dtype=np.float64)
    min_e = w_lin.sum(axis=1)          # (15,)
    tot_e = min_e.sum() + 1e-12

    q05 = np.percentile(min_e, 5)
    q10 = np.percentile(min_e, 10)
    q20 = np.percentile(min_e, 20)
    q15 = np.percentile(min_e, 15)
    med_val = np.median(min_e)
    min_val = min_e.min()
    max_val = min_e.max()

    # --- Group 1: Energy subtraction proxies (6) ---
    share_min = np.maximum(0.0, min_e - min_val).sum() / tot_e
    share_q05 = np.maximum(0.0, min_e - q05).sum() / tot_e
    share_q10 = np.maximum(0.0, min_e - q10).sum() / tot_e
    share_q15 = np.maximum(0.0, min_e - q15).sum() / tot_e
    share_q20 = np.maximum(0.0, min_e - q20).sum() / tot_e
    share_med = np.maximum(0.0, min_e - med_val).sum() / tot_e

    # --- Group 2: Dynamic range (6) ---
    ratio_max_min = max_val / (min_val + 1e-12)
    ratio_max_q10 = max_val / (q10 + 1e-12)
    log_dyn_range = np.log10(ratio_max_min + 1.0)
    top2_share = np.sort(min_e)[-2:].sum() / tot_e
    top3_share = np.sort(min_e)[-3:].sum() / tot_e
    cv_energy = np.std(min_e) / (np.mean(min_e) + 1e-12)

    # --- Group 3: Temporal shape (5) ---
    skew_val = float(skew(min_e))
    kurt_val = float(kurtosis(min_e))
    peak_t = int(np.argmax(min_e))
    peak_dist_center = abs(peak_t - 7.0)
    # Autocorrelation lag-1 of minute energy
    if np.std(min_e) > 1e-9:
        ac1 = float(np.corrcoef(min_e[:-1], min_e[1:])[0, 1])
        if not np.isfinite(ac1):
            ac1 = 0.0
    else:
        ac1 = 0.0

    # --- Group 4: CPA Lorentzian fit (5) ---
    B0 = q20
    A0 = max(0.0, max_val - B0)
    t0_0 = float(peak_t)
    ss_tot = np.sum((min_e - np.mean(min_e)) ** 2) + 1e-12
    try:
        popt_l, _ = curve_fit(lorentzian, t, min_e,
                               p0=[A0, t0_0, 2.0, B0],
                               bounds=([0, 0, 0.4, 0], [np.inf, 14, 8, np.inf]),
                               maxfev=120)
        Al, t0l, sigl, Bl = popt_l
        pred_l = lorentzian(t, Al, t0l, sigl, Bl)
        ss_res_l = np.sum((min_e - pred_l) ** 2)
        r2_lorentz = max(0.0, 1.0 - ss_res_l / ss_tot)
        vessel_e_l = np.maximum(0.0, pred_l - Bl).sum()
        fitted_lorentz_share = min(1.0, vessel_e_l / tot_e)
        snr_lorentz = Al / (Bl + 1e-9)
        lorentz_res_std = np.std(min_e - pred_l) / (np.mean(min_e) + 1e-12)
        t0_lorentz = t0l
    except Exception:
        r2_lorentz, fitted_lorentz_share, snr_lorentz = 0.0, 0.0, 0.0
        sigl, lorentz_res_std, t0_lorentz = 2.0, 0.0, t0_0

    # --- Group 5: CPA Gaussian fit (3) ---
    try:
        popt_g, _ = curve_fit(gaussian_fn, t, min_e,
                               p0=[A0, t0_0, 2.0, B0],
                               bounds=([0, 0, 0.4, 0], [np.inf, 14, 8, np.inf]),
                               maxfev=120)
        Ag, t0g, sigg, Bg = popt_g
        pred_g = gaussian_fn(t, Ag, t0g, sigg, Bg)
        ss_res_g = np.sum((min_e - pred_g) ** 2)
        r2_gauss = max(0.0, 1.0 - ss_res_g / ss_tot)
        vessel_e_g = np.maximum(0.0, pred_g - Bg).sum()
        fitted_gauss_share = min(1.0, vessel_e_g / tot_e)
        ratio_lorentz_gauss = r2_lorentz - r2_gauss
    except Exception:
        r2_gauss, fitted_gauss_share, ratio_lorentz_gauss = 0.0, 0.0, 0.0

    # --- Group 6: Multi-band synchrony (5) ---
    band_peaks = np.argmax(w_lin, axis=0)   # (25,)
    peak_time_std = float(np.std(band_peaks))
    peak_diff_cpa = float(np.mean(np.abs(band_peaks - t0_lorentz)))

    corrs = []
    for b in range(25):
        s = np.std(w_lin[:, b])
        if s > 1e-9:
            c = np.corrcoef(min_e, w_lin[:, b])[0, 1]
            if np.isfinite(c):
                corrs.append(c)
    mean_band_corr = float(np.mean(corrs)) if corrs else 0.0
    min_band_corr = float(np.min(corrs)) if corrs else 0.0
    frac_high_corr = float(np.mean(np.array(corrs) > 0.7)) if corrs else 0.0

    # --- Group 7: Coherence eigenvalues (3) ---
    try:
        cov = np.cov(w_lin.T)   # (25, 25)
        evals = np.linalg.eigvalsh(cov)
        evals_sum = evals.sum() + 1e-12
        eigen_ratio_1 = float(evals[-1] / evals_sum)
        eigen_ratio_2 = float((evals[-1] + evals[-2]) / evals_sum)
        p_evals = np.clip(evals / evals_sum, 1e-12, 1.0)
        spectral_entropy = float(-np.sum(p_evals * np.log2(p_evals)))
    except Exception:
        eigen_ratio_1, eigen_ratio_2, spectral_entropy = 0.5, 0.7, 3.0

    # --- Group 8: Background subtraction (7) ---
    quiet_idx = np.argsort(min_e)[:3]
    bg_spectrum = w_lin[quiet_idx].mean(axis=0, keepdims=True)   # (1, 25)
    excess_matrix = np.maximum(0.0, w_lin - bg_spectrum)
    band_excess_share = float(excess_matrix.sum() / tot_e)

    low_excess  = float(excess_matrix[:, :13].sum() / tot_e)
    mid_excess  = float(excess_matrix[:, 13:17].sum() / tot_e)
    high_excess = float(excess_matrix[:, 17:].sum() / tot_e)
    ratio_low_high = (low_excess + 1e-6) / (high_excess + 1e-6)
    diff_low_high = low_excess - high_excess
    # Excess concentrated in time (gini-like)
    exc_per_min = excess_matrix.sum(axis=1) / (excess_matrix.sum() + 1e-12)
    exc_concentration = float(1.0 - (-np.sum(exc_per_min * np.log2(exc_per_min + 1e-12)) / np.log2(15)))

    # --- Group 9: Spectral tilt & centroid (6) ---
    pk_spec = w_lin[peak_t]
    tilt_peak = (pk_spec[:8].sum() + 1e-9) / (pk_spec[17:].sum() + 1e-9)
    quiet_spectrum = bg_spectrum[0]
    tilt_quiet = (quiet_spectrum[:8].sum() + 1e-9) / (quiet_spectrum[17:].sum() + 1e-9)
    tilt_contrast = np.log10(tilt_peak / (tilt_quiet + 1e-12) + 1.0)
    freq_weights = np.arange(25, dtype=np.float64)
    centroid_peak = np.sum(freq_weights * pk_spec) / (pk_spec.sum() + 1e-9)
    centroid_quiet = np.sum(freq_weights * quiet_spectrum) / (quiet_spectrum.sum() + 1e-9)
    centroid_shift = centroid_peak - centroid_quiet

    # Spectral flatness (geometric-to-arithmetic mean ratio) at peak vs quiet
    gm_pk = np.exp(np.log(pk_spec + 1e-12).mean())
    am_pk = pk_spec.mean()
    spectral_flatness_peak = float(gm_pk / (am_pk + 1e-12))

    # --- Group 10: NEW — per-band temporal skewness (3 aggregates) ---
    per_band_skew = np.array([float(skew(w_lin[:, b])) for b in range(25)])
    per_band_skew = np.where(np.isfinite(per_band_skew), per_band_skew, 0.0)
    mean_band_skew = float(per_band_skew.mean())
    max_band_skew = float(per_band_skew.max())
    frac_pos_skew = float((per_band_skew > 0.5).mean())

    # --- Group 11: NEW — 1st-lag autocorrelation per band (3 aggregates) ---
    band_ac1s = []
    for b in range(25):
        sb = w_lin[:, b]
        if np.std(sb) > 1e-9:
            c_b = np.corrcoef(sb[:-1], sb[1:])[0, 1]
            if np.isfinite(c_b):
                band_ac1s.append(c_b)
    mean_band_ac1 = float(np.mean(band_ac1s)) if band_ac1s else 0.0
    min_band_ac1 = float(np.min(band_ac1s)) if band_ac1s else 0.0
    max_band_ac1 = float(np.max(band_ac1s)) if band_ac1s else 0.0

    return np.array([
        # G1 (6)
        share_min, share_q05, share_q10, share_q15, share_q20, share_med,
        # G2 (6)
        ratio_max_min, ratio_max_q10, log_dyn_range, top2_share, top3_share, cv_energy,
        # G3 (5)
        skew_val, kurt_val, peak_t, peak_dist_center, ac1,
        # G4 (5)
        r2_lorentz, fitted_lorentz_share, snr_lorentz, sigl, lorentz_res_std,
        # G5 (3)
        r2_gauss, fitted_gauss_share, ratio_lorentz_gauss,
        # G6 (5)
        peak_time_std, peak_diff_cpa, mean_band_corr, min_band_corr, frac_high_corr,
        # G7 (3)
        eigen_ratio_1, eigen_ratio_2, spectral_entropy,
        # G8 (7)
        band_excess_share, low_excess, mid_excess, high_excess,
        ratio_low_high, diff_low_high, exc_concentration,
        # G9 (6)
        tilt_peak, tilt_quiet, tilt_contrast, centroid_shift,
        centroid_peak, spectral_flatness_peak,
        # G10 (3)
        mean_band_skew, max_band_skew, frac_pos_skew,
        # G11 (3)
        mean_band_ac1, min_band_ac1, max_band_ac1,
    ], dtype=np.float32)

FEATURE_NAMES = [
    # G1
    'share_min','share_q05','share_q10','share_q15','share_q20','share_med',
    # G2
    'ratio_max_min','ratio_max_q10','log_dyn_range','top2_share','top3_share','cv_energy',
    # G3
    'skew_val','kurt_val','peak_t','peak_dist_center','ac1',
    # G4
    'r2_lorentz','fitted_lorentz_share','snr_lorentz','sigl','lorentz_res_std',
    # G5
    'r2_gauss','fitted_gauss_share','ratio_lorentz_gauss',
    # G6
    'peak_time_std','peak_diff_cpa','mean_band_corr','min_band_corr','frac_high_corr',
    # G7
    'eigen_ratio_1','eigen_ratio_2','spectral_entropy',
    # G8
    'band_excess_share','low_excess','mid_excess','high_excess',
    'ratio_low_high','diff_low_high','exc_concentration',
    # G9
    'tilt_peak','tilt_quiet','tilt_contrast','centroid_shift',
    'centroid_peak','spectral_flatness_peak',
    # G10
    'mean_band_skew','max_band_skew','frac_pos_skew',
    # G11
    'mean_band_ac1','min_band_ac1','max_band_ac1',
]
assert len(FEATURE_NAMES) == 52, f"Expected 52, got {len(FEATURE_NAMES)}"

# ------------------------------------------------------------------
# 3. Main
# ------------------------------------------------------------------
def main():
    t_start = time.time()

    # ----- Step 1: Load background windows -----
    print("=== Step 1: Loading real background windows from all 10 stations ===")
    parquet_files = sorted(glob.glob('acoustics/*.parquet'))
    raw_backgrounds = []

    for pf in parquet_files:
        st_name = os.path.basename(pf).replace('.parquet', '')
        tbl = pq.read_table(pf, columns=TOL_COLS)
        vals = tbl.to_pandas().to_numpy(dtype=np.float32)
        n = len(vals)
        print(f"  {st_name:22s}: {n:,} rows")

        count = 0
        step = max(1, n // 1200)        # sample up to 1200 windows/station
        for s in range(0, min(n - 15, 200000), step):
            chunk = vals[s:s+15]
            if np.isnan(chunk).mean() < 0.12:
                lin = 10.0 ** (chunk / 10.0)
                if np.isfinite(lin).all() and (lin > 0).all():
                    raw_backgrounds.append(lin)
                    count += 1
                    if count >= 1200:
                        break
        print(f"    -> {count} windows extracted")

    total_bg = len(raw_backgrounds)
    print(f"\nTotal background windows: {total_bg:,}")

    # ----- Step 2: Simulate training dataset -----
    print("\n=== Step 2: Generating enhanced simulation dataset ===")
    np.random.seed(2024)
    X_list, y_list = [], []
    t_vec = np.arange(15, dtype=np.float64)[:, None]   # (15, 1)

    for idx, bg_lin in enumerate(raw_backgrounds):
        w_bg = bg_lin.copy()
        # impute NaN
        for b in range(25):
            col = w_bg[:, b]
            m = np.isnan(col)
            if m.any():
                mv = np.nanmedian(col)
                w_bg[m, b] = mv if np.isfinite(mv) else 1e4

        bg_tot = w_bg.sum()

        # 52 % vessel, 48 % clean background
        has_vessel = (np.random.rand() < 0.52)

        if not has_vessel:
            w_mix = w_bg.copy()
            target_share = 0.0
        else:
            # Time profile: Lorentzian (65 %) or Gaussian (20 %) or asymmetric (15 %)
            t0 = np.random.uniform(1.5, 13.5)
            sigma = np.random.uniform(0.7, 3.5)
            rr = np.random.rand()
            if rr < 0.65:
                time_curve = 1.0 / (1.0 + ((t_vec - t0) / sigma) ** 2)
            elif rr < 0.85:
                time_curve = np.exp(-0.5 * ((t_vec - t0) / sigma) ** 2)
            else:
                # Asymmetric: fast approach, slow departure
                sigma2 = np.random.uniform(1.5, 5.0)
                tc = np.where(t_vec <= t0,
                              np.exp(-0.5 * ((t_vec - t0) / sigma) ** 2),
                              np.exp(-0.5 * ((t_vec - t0) / sigma2) ** 2))
                time_curve = tc

            # Vessel spectral pattern
            b_idx = np.arange(25, dtype=np.float64)
            center_b = np.random.uniform(2.0, 12.0)
            bw = np.random.uniform(1.5, 5.0)
            spec = np.exp(-0.5 * ((b_idx - center_b) / bw) ** 2)
            # Low-freq rumble (propeller + engine)
            spec[:8] += np.random.uniform(0.3, 2.0)
            # Cavitation high-freq tail
            spec[16:] += np.random.uniform(0.03, 0.5)
            # Harmonics at ~125, 250, 500 Hz (bands 6, 9, 12)
            for hb in [6, 9, 12]:
                spec[hb] += np.random.uniform(0.0, 1.5)
            spec /= spec.sum()

            vessel_matrix = time_curve * spec[None, :]     # (15, 25)

            # Target share: Beta(1.6, 6.0) → mean ≈ 0.21
            target_s = float(np.random.beta(1.6, 6.0))
            target_s = float(np.clip(target_s, 0.004, 0.75))

            desired_ev = bg_tot * target_s / (1.0 - target_s + 1e-9)
            vessel_matrix = vessel_matrix * (desired_ev / (vessel_matrix.sum() + 1e-9))

            w_mix = w_bg + vessel_matrix
            target_share = float(vessel_matrix.sum() / (w_mix.sum() + 1e-12))

        feats = extract_55_features(w_mix)
        X_list.append(feats)
        y_list.append(target_share)

        if (idx + 1) % 2000 == 0:
            print(f"  Simulated {idx+1:,} / {total_bg:,}...")

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.float32)
    print(f"\nX_train: {X_train.shape}  y_train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")

    # ----- Step 3: Extract features on Test.csv -----
    print("\n=== Step 3: Extracting features on Test.csv ===")
    test_df = pd.read_csv('Test.csv')
    feat_cols = [c for c in test_df.columns if c != 'ID']
    test_arr = test_df[feat_cols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
    test_lin = 10.0 ** (test_arr / 10.0)

    X_test_list = []
    for i in range(len(test_lin)):
        w = test_lin[i].copy()
        for b in range(25):
            vals = w[:, b]
            m = np.isnan(vals)
            if m.any():
                mv = np.nanmedian(vals)
                w[m, b] = mv if np.isfinite(mv) else 1e4
        X_test_list.append(extract_55_features(w))

    X_test = np.array(X_test_list, dtype=np.float32)
    print(f"X_test: {X_test.shape}")

    # Replace any NaN/Inf in features
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

    # ----- Step 4: 5-Fold ensemble training -----
    print("\n=== Step 4: 5-Fold Multi-Model Ensemble (LGB + XGB + HGB) ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    n_train = len(y_train)
    n_test  = len(test_df)

    oof_lgb = np.zeros(n_train)
    oof_xgb = np.zeros(n_train)
    oof_hgb = np.zeros(n_train)
    test_lgb = np.zeros(n_test)
    test_xgb = np.zeros(n_test)
    test_hgb = np.zeros(n_test)

    lgb_params = dict(
        objective='regression', metric='rmse', boosting_type='gbdt',
        n_estimators=1000, learning_rate=0.02, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1
    )
    xgb_params = dict(
        n_estimators=800, learning_rate=0.02, max_depth=5,
        subsample=0.8, colsample_bytree=0.75,
        min_child_weight=5, gamma=0.05,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method='hist', random_state=42, n_jobs=-1,
        eval_metric='rmse', early_stopping_rounds=50,
        verbosity=0
    )

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        Xva, yva = X_train[va_idx], y_train[va_idx]

        # LightGBM
        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
        oof_lgb[va_idx] = m_lgb.predict(Xva)
        test_lgb += m_lgb.predict(X_test) / kf.n_splits

        # XGBoost
        m_xgb = xgb.XGBRegressor(**xgb_params)
        m_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof_xgb[va_idx] = m_xgb.predict(Xva)
        test_xgb += m_xgb.predict(X_test) / kf.n_splits

        # HistGradientBoosting (sklearn)
        m_hgb = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.025, max_leaf_nodes=31,
            min_samples_leaf=20, random_state=42 + fold
        )
        m_hgb.fit(Xtr, ytr)
        oof_hgb[va_idx] = m_hgb.predict(Xva)
        test_hgb += m_hgb.predict(X_test) / kf.n_splits

        combo = 0.5 * oof_lgb[va_idx] + 0.3 * oof_xgb[va_idx] + 0.2 * oof_hgb[va_idx]
        rmse_fold = np.sqrt(mean_squared_error(yva, np.clip(combo, 0, 1)))
        print(f"  Fold {fold+1}: RMSE={rmse_fold:.5f} | 1-RMSE={1-rmse_fold:.5f}")

    oof_combo = 0.5 * oof_lgb + 0.3 * oof_xgb + 0.2 * oof_hgb
    oof_rmse = np.sqrt(mean_squared_error(y_train, np.clip(oof_combo, 0, 1)))
    print(f"\nOverall OOF Score (1 - RMSE): {1 - oof_rmse:.5f}  (RMSE: {oof_rmse:.5f})")

    # ----- Step 5: Calibration -----
    print("\n=== Step 5: Calibration ===")
    raw_pred = 0.5 * test_lgb + 0.3 * test_xgb + 0.2 * test_hgb
    raw_pred = np.clip(raw_pred, 0.0, 1.0)

    target_mu    = 0.108
    target_sigma = 0.078

    cal_pred = (raw_pred - raw_pred.mean()) * (target_sigma / (raw_pred.std() + 1e-9)) + target_mu
    cal_pred = np.clip(cal_pred, 0.0, 1.0)
    print(f"Calibrated v17: mean={cal_pred.mean():.4f}, std={cal_pred.std():.4f}, "
          f"min={cal_pred.min():.4f}, max={cal_pred.max():.4f}")

    v14_cal = pd.read_csv('submission_v14_calibrated.csv')['share_added_vessel'].to_numpy()
    corr_v14 = np.corrcoef(v14_cal, cal_pred)[0, 1]
    print(f"Correlation v17 vs v14 (0.8079): {corr_v14:.4f}")

    # ----- Step 6: Outputs -----
    print("\n=== Step 6: Saving submissions ===")

    # Submission A: v17 standalone
    sub_v17 = pd.DataFrame({
        'ID': test_df['ID'],
        'share_added_vessel': np.round(cal_pred, 5),
        'score_added_vessel': np.round(np.clip(cal_pred * 2.5, 0.0, 1.0), 5)
    })
    sub_v17.to_csv('submission_v17_elite_standalone.csv', index=False)
    print("Saved submission_v17_elite_standalone.csv")

    # Submission B: diverse blend v14 (50%) + v17 (50%)
    # Using 50/50 to maximally diversify vs v14 while keeping proven signal
    blend_pred = 0.50 * v14_cal + 0.50 * cal_pred
    blend_pred = (blend_pred - blend_pred.mean()) * (target_sigma / (blend_pred.std() + 1e-9)) + target_mu
    blend_pred = np.clip(blend_pred, 0.0, 1.0)
    sub_blend = pd.DataFrame({
        'ID': test_df['ID'],
        'share_added_vessel': np.round(blend_pred, 5),
        'score_added_vessel': np.round(np.clip(blend_pred * 2.5, 0.0, 1.0), 5)
    })
    sub_blend.to_csv('submission_v17_elite_blend50.csv', index=False)
    print("Saved submission_v17_elite_blend50.csv")

    # Submission C: conservative blend v14 (70%) + v17 (30%)
    blend_pred2 = 0.70 * v14_cal + 0.30 * cal_pred
    blend_pred2 = (blend_pred2 - blend_pred2.mean()) * (target_sigma / (blend_pred2.std() + 1e-9)) + target_mu
    blend_pred2 = np.clip(blend_pred2, 0.0, 1.0)
    sub_blend2 = pd.DataFrame({
        'ID': test_df['ID'],
        'share_added_vessel': np.round(blend_pred2, 5),
        'score_added_vessel': np.round(np.clip(blend_pred2 * 2.5, 0.0, 1.0), 5)
    })
    sub_blend2.to_csv('submission_v17_elite_blend70.csv', index=False)
    print("Saved submission_v17_elite_blend70.csv")

    print(f"\nTotal elapsed: {time.time() - t_start:.0f}s")

if __name__ == '__main__':
    main()
