import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib


def load_and_interpolate(path):
    df = pd.read_csv(path)
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.set_index('created_at')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    cols = ['temperature', 'humidity', 'pm25']
    if df[cols].isna().any().any():
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        df = df.reindex(full_range)
        df = df.infer_objects(copy=False)
        df = df.interpolate(method='time')
        df = df.dropna(subset=['pm25'])

    df = df.reset_index().rename(columns={'index': 'created_at'})
    return df


def add_time_features(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    timestamp_s = df['created_at'].map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    df_f = df.drop(columns=['created_at', 'id', 'year', 'month', 'sht31_temp', 'sht31_hum'], errors='ignore')
    return df_f


def normalize(df_f):
    sensor_cols = ['temperature', 'humidity', 'pm25']

    df_ready = df_f.copy()
    df_ready['pm25'] = np.log1p(np.maximum(df_ready['pm25'], 0))

    scaler = RobustScaler()
    df_ready[sensor_cols] = scaler.fit_transform(df_ready[sensor_cols])
    df_ready[sensor_cols] = np.clip(df_ready[sensor_cols], -3.0, 3.0)
    joblib.dump(scaler, 'scaler.pkl')

    print("--- VERIFIKASI ANGKA MENTAH ---")
    print(f"Max PM2.5      : {df_ready['pm25'].max():.4f}")
    print(f"Min Humidity   : {df_ready['humidity'].min():.4f}")
    print(f"Max Temperature: {df_ready['temperature'].max():.4f}")

    return df_ready, scaler


def split_data(df_ready):
    from config import FEATURE_COLS, LABEL_COLS

    n = len(df_ready)
    train_df = df_ready[0:int(n * 0.7)].copy()
    val_df   = df_ready[int(n * 0.7):int(n * 0.9)].copy()
    test_df  = df_ready[int(n * 0.9):].copy()

    # Derived features dihitung per split agar tidak cross-contaminate antar split
    for df in [train_df, val_df, test_df]:
        df['pm25_diff']         = df['pm25'].diff().bfill()
        df['pm25_rolling_mean'] = df['pm25'].rolling(6, min_periods=1).mean()
        df['pm25_rolling_std']  = df['pm25'].rolling(6, min_periods=1).std().fillna(0)

    print("\n[SPLIT STATS - PM2.5 Setelah Normalisasi]")
    print(f"{'Split':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 40)
    for name, part in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        v = part['pm25'].dropna()
        print(f"{name:<8} {v.mean():>8.4f} {v.std():>8.4f} {v.min():>8.4f} {v.max():>8.4f}")
    print("-" * 40)

    all_cols = list(dict.fromkeys(FEATURE_COLS + LABEL_COLS))
    return train_df[all_cols], val_df[all_cols], test_df[all_cols]