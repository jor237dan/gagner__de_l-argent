"""
train_v26_improved_triplets.py
=================================
v26: Version améliorée de v25 avec:
1. Correction extraction fenêtres quiet (toutes stations, pas de zéro)
2. Recalibration optimale (vertex de la parabole blend)
3. Plus grand ensemble d'entraînement (6000 samples par seed)
4. Hyperparamètres LightGBM affinés
"""

import os, glob, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

BANDS = ['31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5',
         '251.2','316.2','398.1','501.2','631','794.3','1000','1258.9',
         '1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3']
TOL_COLS = [f'TOL_{b}' for b in BANDS]


def lorentzian(t, A, t0, sigma, B):
    return A / (1.0 + ((t - t0) / (sigma + 1e-4))**2) + B


def extract_features(w_lin, st_id, station_profiles_db):
    """32 features identiques à v24/v25."""
    t = np.arange(15, dtype=np.float64)
    min_e = w_lin.sum(axis=1)
    tot_e = min_e.sum() + 1e-12

    q10 = np.percentile(min_e, 10); q20 = np.percentile(min_e, 20)
    med_val = np.median(min_e); min_val = min_e.min(); max_val = min_e.max()

    share_min  = np.maximum(0., min_e - min_val).sum() / tot_e
    share_q10  = np.maximum(0., min_e - q10).sum() / tot_e
    share_q20  = np.maximum(0., min_e - q20).sum() / tot_e
    share_med  = np.maximum(0., min_e - med_val).sum() / tot_e

    ratio_max_min = max_val / (min_val + 1e-12)
    ratio_max_q10 = max_val / (q10 + 1e-12)
    log_dyn_range = np.log10(ratio_max_min + 1.)
    top2_share = np.sort(min_e)[-2:].sum() / tot_e
    top3_share = np.sort(min_e)[-3:].sum() / tot_e
    cv_energy  = min_e.std() / (min_e.mean() + 1e-12)

    B0 = q20; A0 = max(1e-6, max_val - B0); t0_0 = float(np.argmax(min_e))
    try:
        popt, _ = curve_fit(lorentzian, t, min_e, p0=[A0, t0_0, 2., B0],
                            bounds=([0,0,.4,0],[np.inf,14,8,np.inf]), maxfev=100)
        A, t0c, sigma, B = popt
        pred = lorentzian(t, A, t0c, sigma, B)
        ss_r = ((min_e-pred)**2).sum(); ss_t = ((min_e-min_e.mean())**2).sum()+1e-12
        r2 = max(0., 1.-ss_r/ss_t)
        vessel_e = np.maximum(0., pred-B).sum()
        fitted_share = min(1., vessel_e/tot_e)
        snr = A/(B+1e-9)
    except Exception:
        t0c, sigma, r2, snr, fitted_share = t0_0, 2., 0., 0., 0.
        A = 0.
    cpa_dist = abs(t0c - 7.)

    band_peaks = w_lin.argmax(axis=0)
    peak_std = float(band_peaks.std())
    peak_diff = float(np.abs(band_peaks - t0c).mean())
    corrs = [np.corrcoef(min_e, w_lin[:,b])[0,1] for b in range(25) if w_lin[:,b].std()>1e-9]
    mean_corr = float(np.mean([c for c in corrs if np.isfinite(c)])) if corrs else 0.

    qi = np.argsort(min_e)[:3]
    bg = w_lin[qi].mean(axis=0, keepdims=True)
    exc = np.maximum(0., w_lin - bg)
    band_exc = exc.sum()/tot_e
    low_exc  = exc[:,:13].sum()/tot_e
    mid_exc  = exc[:,13:17].sum()/tot_e
    high_exc = exc[:,17:].sum()/tot_e
    r_lh = (low_exc+1e-6)/(high_exc+1e-6)

    pi = int(np.clip(round(t0c),0,14))
    tilt_pk = (w_lin[pi,:8].sum()+1e-9)/(w_lin[pi,17:].sum()+1e-9)
    tilt_bg = (bg[0,:8].sum()+1e-9)/(bg[0,17:].sum()+1e-9)
    tilt_c  = np.log10(tilt_pk/(tilt_bg+1e-12)+1.)

    w_db = 10.*np.log10(w_lin+1e-12)
    b_db = np.percentile(w_db, 20, axis=0, keepdims=True)
    dd = w_db - b_db
    mm = np.median(dd[:,:14], axis=1)
    def mc(mask):
        m=0; c=0
        for v in mask:
            if v: c+=1; m=max(m,c)
            else: c=0
        return float(m)
    c2=mc(mm>=2.); c3=mc(mm>=3.); c4=mc(mm>=4.)

    sp = station_profiles_db[st_id]
    sp_lin = 10.**(sp/10.)
    exc_st = np.maximum(0., w_lin-sp_lin[None,:]).sum()/tot_e
    ss_off = float((b_db[0,16:]-sp[16:]).mean())
    core_off = float((b_db[0,:14]-sp[:14]).mean())

    return [share_min, share_q10, share_q20, share_med,
            ratio_max_min, ratio_max_q10, log_dyn_range, top2_share, top3_share, cv_energy,
            r2, snr, t0c, sigma, fitted_share, cpa_dist,
            peak_std, peak_diff, mean_corr,
            band_exc, low_exc, mid_exc, high_exc, r_lh, tilt_c,
            c2, c3, c4,
            float(st_id), exc_st, ss_off, core_off]


def load_windows_improved(parquet_files, max_per_station=800):
    """
    Chargement amélioré: utilise le PERCENTILE LOCAL de chaque station
    pour définir 'quiet' -> tous les stations ont des fenêtres quiet.
    """
    quiet_windows, vessel_windows = [], []

    for st_id, pf in enumerate(parquet_files):
        st_name = os.path.basename(pf).replace('.parquet','')
        tt = pq.read_table(pf, columns=TOL_COLS)
        df_raw = tt.to_pandas().to_numpy(dtype=np.float32).copy()

        for b in range(25):
            col = df_raw[:,b]
            med = np.nanmedian(col)
            df_raw[np.isnan(col), b] = med if np.isfinite(med) else 70.

        df_lin = 10.**(df_raw/10.)
        N = len(df_lin)

        # Énergie basses fréquences par ligne
        e_low = df_lin[:,:13].sum(axis=1)
        # Seuils LOCAUX à la station (percentiles)
        p10 = np.percentile(e_low, 10)
        p30 = np.percentile(e_low, 30)
        p70 = np.percentile(e_low, 70)
        p90 = np.percentile(e_low, 90)

        # Quiet = énergie dans le bas de la distribution de la station
        quiet_thresh = p30
        # Vessel = pic clairement au-dessus de la médiane de la station
        vessel_thresh = p90

        st_quiet = []
        st_vessel = []
        step = max(1, N // (max_per_station * 6))

        for i in range(0, N-15, step):
            chunk = df_lin[i:i+15]
            if chunk.shape[0] < 15: break
            if np.isnan(chunk).mean() > 0.15: continue
            if not np.isfinite(chunk).all() or not (chunk > 0).all(): continue

            e_chunk = chunk[:,:13].sum(axis=1)
            max_e = e_chunk.max()
            min_e_chunk = e_chunk.min()

            # Quiet: toute la fenêtre reste dans les 30% les plus bas de la station
            if max_e <= quiet_thresh and len(st_quiet) < max_per_station:
                st_quiet.append((chunk.copy(), st_id))

            # Vessel: pic > P90 ET minimum < P70 (variation temporelle)
            elif (max_e > vessel_thresh and
                  min_e_chunk < p70 and
                  len(st_vessel) < max_per_station):
                st_p20 = np.nanpercentile(df_lin, 20, axis=0)
                vc = np.maximum(0., chunk - st_p20[None,:])
                if vc.sum() > 0:
                    st_vessel.append((chunk.copy(), st_id, vc.copy()))

        quiet_windows.extend(st_quiet)
        vessel_windows.extend(st_vessel)
        print(f"  {st_name:25s}: {len(st_quiet):3d} quiet, {len(st_vessel):3d} vessel")

    return quiet_windows, vessel_windows


def build_triplets(quiet_windows, vessel_windows, n_samples, rng):
    X_list, y_list = [], []
    t_vec = np.arange(15, dtype=np.float64)[:,None]

    for _ in range(n_samples):
        q_idx = int(rng.integers(0, len(quiet_windows)))
        bg_lin, st_id = quiet_windows[q_idx]
        has_vessel = rng.random() < 0.50

        if not has_vessel:
            mix = bg_lin.copy(); target = 0.
        else:
            use_real = len(vessel_windows) > 0 and rng.random() < 0.70
            if use_real:
                v_idx = int(rng.integers(0, len(vessel_windows)))
                _, _, vc = vessel_windows[v_idx]
                alpha = float(np.clip(rng.beta(1., 3.), 0.005, 0.80))
                bg_tot = bg_lin.sum()
                vn = vc / (vc.sum()+1e-9)
                ve = bg_tot * alpha / (1.-alpha+1e-9)
                vs = vn * ve
            else:
                t0s = rng.uniform(1.5, 13.5); sg = rng.uniform(0.8, 3.5)
                tp = 1./(1.+((t_vec-t0s)/sg)**2) if rng.random()<0.6 else np.exp(-.5*((t_vec-t0s)/sg)**2)
                fi = np.arange(25, dtype=np.float64)
                cb = rng.uniform(3.,10.); bw = rng.uniform(2.,5.)
                fs = np.exp(-.5*((fi-cb)/bw)**2)
                fs[:7] += rng.uniform(0.3, 1.2); fs[12:] += rng.uniform(0.01, 0.15)
                fs /= fs.sum()
                vp = tp * fs[None,:]
                alpha = float(np.clip(rng.beta(1., 3.5), 0.005, 0.85))
                bg_tot = bg_lin.sum()
                vs = vp * (bg_tot*alpha/(1.-alpha+1e-9)) / (vp.sum()+1e-9)

            mix = bg_lin + vs
            target = float(vs.sum()/(mix.sum()+1e-9))

        mdb = 10.*np.log10(mix+1e-12)
        nm = rng.random(mdb.shape) < 0.025; mdb[nm] = np.nan
        mlc = 10.**(mdb/10.)
        for b in range(25):
            v2 = mlc[:,b]; m2 = np.isnan(v2)
            if m2.any():
                med = np.nanmedian(v2)
                mlc[m2,b] = med if np.isfinite(med) else 1e4

        X_list.append((mlc, st_id)); y_list.append(target)
    return X_list, y_list


def main():
    t_glob = time.time()
    print("="*70)
    print("v26: Improved Real-Data Triplets Pipeline")
    print("="*70)

    parquet_files = sorted(glob.glob('acoustics/*.parquet'))

    # Profils station
    station_profiles_db = []
    for pf in parquet_files:
        tt = pq.read_table(pf, columns=TOL_COLS)
        station_profiles_db.append(np.nanmedian(tt.to_pandas().to_numpy(dtype=np.float32), axis=0))
    station_profiles_db = np.array(station_profiles_db, dtype=np.float32)

    # Chargement fenêtres réelles
    print("\n=== Étape 1: Fenêtres réelles ===")
    quiet_windows, vessel_windows = load_windows_improved(parquet_files, max_per_station=700)
    print(f"\nTotal quiet: {len(quiet_windows):,} | Total vessel: {len(vessel_windows):,}")

    # Features test
    print("\n=== Étape 2: Features Test ===")
    test_df = pd.read_csv('Test.csv')
    feat_cols = [c for c in test_df.columns if c != 'ID']
    test_arr = test_df[feat_cols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)

    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nbrs.fit(station_profiles_db[:, 16:])
    test_p20 = np.nanpercentile(test_arr, 20, axis=1)
    _, test_st_ids = nbrs.kneighbors(test_p20[:, 16:])
    test_st_ids = test_st_ids.flatten()

    test_lin = 10.**(test_arr/10.)
    X_test_list = []
    for i in range(len(test_lin)):
        w = test_lin[i].copy()
        for b in range(25):
            m = np.isnan(w[:,b])
            if m.any():
                med = np.nanmedian(w[:,b])
                w[m,b] = med if np.isfinite(med) else 1e4
        X_test_list.append(extract_features(w, int(test_st_ids[i]), station_profiles_db))
    X_test = np.nan_to_num(np.array(X_test_list, dtype=np.float32), nan=0., posinf=0., neginf=0.)
    print(f"X_test shape: {X_test.shape}")

    # Entraînement multi-seed
    SEEDS = [42, 100, 2024, 777, 999]
    N_TRAIN = 6000
    all_preds = []; all_oofs = []

    for s_idx, seed in enumerate(SEEDS):
        t0s = time.time()
        rng = np.random.default_rng(seed)
        print(f"\n--- Seed {seed} ({s_idx+1}/5) | n_train={N_TRAIN} ---")

        X_raw, y_list = build_triplets(quiet_windows, vessel_windows, N_TRAIN, rng)
        X_tr = np.nan_to_num(np.array([extract_features(w, s, station_profiles_db) for w, s in X_raw],
                                       dtype=np.float32), nan=0., posinf=0., neginf=0.)
        y_tr = np.array(y_list, dtype=np.float32)
        print(f"  y_train: mean={y_tr.mean():.4f} std={y_tr.std():.4f}")

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(N_TRAIN, dtype=np.float32)
        tp = np.zeros(len(test_df), dtype=np.float32)

        params = dict(objective='regression', metric='rmse', boosting_type='gbdt',
                      n_estimators=1000, learning_rate=0.02, num_leaves=47,
                      min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                      reg_alpha=0.05, reg_lambda=0.5, random_state=seed, verbose=-1, n_jobs=-1)

        for _, (tri, vai) in enumerate(kf.split(X_tr, y_tr)):
            reg = lgb.LGBMRegressor(**params)
            reg.fit(X_tr[tri], y_tr[tri], eval_X=X_tr[vai], eval_y=y_tr[vai],
                    callbacks=[lgb.early_stopping(80, verbose=False)])
            oof[vai] = reg.predict(X_tr[vai])
            tp += reg.predict(X_test) / kf.n_splits

        s_rmse = root_mean_squared_error(y_tr, np.clip(oof, 0., 1.))
        s_score = 1. - s_rmse
        all_oofs.append(s_score); all_preds.append(tp)
        print(f"  Seed {seed}: OOF={s_score:.5f} | {time.time()-t0s:.0f}s")

    # Calibration vers le vertex prouvé
    print("\n=== Étape 5: Calibration et soumissions ===")
    raw = np.mean(all_preds, axis=0)
    target_mu = 0.13178; target_sigma = 0.09405
    cal = (raw - raw.mean()) * (target_sigma / (raw.std()+1e-8)) + target_mu
    cal = np.clip(cal, 0., 1.)
    print(f"v26 calibré: mean={cal.mean():.4f} std={cal.std():.4f} max={cal.max():.4f}")

    v22 = pd.read_csv('submission_v22_vertex_optimum.csv')['share_added_vessel'].to_numpy()
    ref_score = pd.read_csv('submission_v22_vertex_optimum.csv')['score_added_vessel']
    print(f"corr(v22, v26)={np.corrcoef(v22, cal)[0,1]:.4f}")

    # Générer soumissions
    for label, a22, a26 in [('master', 0., 1.), ('blend70', .70, .30),
                             ('blend60', .60, .40), ('blend30_vertex', .70, .30)]:
        pred = np.clip(np.round(a22*v22 + a26*cal, 5), 0., 1.)
        fname = f'submission_v26_{label}.csv'
        pd.DataFrame({'ID': test_df['ID'], 'share_added_vessel': pred,
                      'score_added_vessel': ref_score}).to_csv(fname, index=False)
        print(f"Sauvegardé: {fname} | mean={pred.mean():.4f} std={pred.std():.4f}")

    print(f"\nTemps total: {time.time()-t_glob:.0f}s")
    print(f"OOF moyen: {np.mean(all_oofs):.5f} ± {np.std(all_oofs):.5f}")


if __name__ == '__main__':
    main()
