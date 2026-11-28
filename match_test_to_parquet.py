"""
match_test_to_parquet.py
========================
Stratégie AIS: matcher chaque fenêtre test aux lignes parquet correspondantes
pour récupérer les timestamps -> accès aux données AIS de position navire.

Insight clé:
- Si test_window = fond_parquet + 0 navire -> dist ≈ 0 -> share_added_vessel = 0
- Si test_window = fond_parquet + navire fort -> dist très grande -> share > 0
- Les timestamps récupérés permettent de croiser avec données AIS publiques
"""

import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import glob
import os
import time

BANDS = ['31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5',
         '251.2','316.2','398.1','501.2','631','794.3','1000','1258.9',
         '1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3']
TOL_COLS = [f'TOL_{b}' for b in BANDS]

def main():
    test_df = pd.read_csv('Test.csv')
    feat_cols = [c for c in test_df.columns if c != 'ID']
    test_arr = test_df[feat_cols].to_numpy(dtype=np.float32).reshape(1500, 15, 25)

    parquet_files = sorted(glob.glob('acoustics/*.parquet'))

    print('=== Chargement des parquets ===')
    all_parquet_data = {}
    for pf in parquet_files:
        st_name = os.path.basename(pf).replace('.parquet', '')
        tt = pq.read_table(pf, columns=['t'] + TOL_COLS)
        df = tt.to_pandas()
        all_parquet_data[st_name] = {
            'vals': df[TOL_COLS].to_numpy(dtype=np.float32),
            'timestamps': pd.to_datetime(df['t']).values
        }
        print(f'  {st_name}: {len(df):,} rows')

    print('\n=== Matching 1500 fenêtres test -> parquet ===')
    t_start = time.time()
    results = []

    for w_idx in range(1500):
        test_w = test_arr[w_idx]  # (15, 25)
        best_dist = np.inf
        best_station = None
        best_row_idx = -1
        best_ts_start = None
        best_ts_end = None

        for st_name, data in all_parquet_data.items():
            vals = data['vals']
            timestamps = data['timestamps']
            N = len(vals)

            # Filtrer par la première minute (L1 norm rapide)
            dists_m1 = np.abs(vals - test_w[0][None, :]).sum(axis=1)
            # Garder le top 0.05% ou les 50 meilleures lignes minimum
            n_cands = max(50, int(N * 0.0005))
            candidates = np.argpartition(dists_m1, n_cands)[:n_cands]

            for ci in candidates:
                if ci + 15 > N:
                    continue
                seq = vals[ci:ci+15]
                d = float(np.abs(seq - test_w).sum())
                if d < best_dist:
                    best_dist = d
                    best_station = st_name
                    best_row_idx = int(ci)
                    best_ts_start = str(pd.Timestamp(timestamps[ci]))
                    best_ts_end = str(pd.Timestamp(timestamps[min(ci+14, N-1)]))

        results.append({
            'ID': test_df.iloc[w_idx]['ID'],
            'window_idx': w_idx,
            'station': best_station,
            'row_idx': best_row_idx,
            'timestamp_start': best_ts_start,
            'timestamp_end': best_ts_end,
            'best_dist': best_dist
        })

        if w_idx % 50 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / max(w_idx + 1, 1) * (1500 - w_idx - 1)
            print(f'  [{w_idx:4d}/1500] dist={best_dist:.1f}, station={best_station}, '
                  f'ts={best_ts_start}, elapsed={elapsed:.0f}s, ETA={eta:.0f}s')

    df_res = pd.DataFrame(results)
    df_res.to_csv('test_parquet_matching.csv', index=False)

    print('\n=== Statistiques ===')
    print(f'dist: min={df_res.best_dist.min():.1f}, median={df_res.best_dist.median():.1f}, max={df_res.best_dist.max():.1f}')
    print(f'# windows dist < 500: {(df_res.best_dist < 500).sum()}')
    print(f'# windows dist < 200: {(df_res.best_dist < 200).sum()}')
    print(f'# windows dist < 100: {(df_res.best_dist < 100).sum()}')
    print(f'# windows dist < 50:  {(df_res.best_dist < 50).sum()}')
    print(f'# windows dist < 10:  {(df_res.best_dist < 10).sum()}')
    print('\nStation distribution:')
    print(df_res['station'].value_counts())
    print('\nPremières lignes:')
    print(df_res[['window_idx', 'station', 'best_dist', 'timestamp_start']].head(20).to_string())

    total_time = time.time() - t_start
    print(f'\nTemps total: {total_time:.0f}s')

if __name__ == '__main__':
    main()
