import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_interpolation(path):
    df = pd.read_csv(path)
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.set_index('created_at')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index
    
    cols = ['temperature', 'humidity', 'pm25']
    if df[cols].isna().any().any():
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        df = df.reindex(full_range)
        df = df.infer_objects(copy=False)
        df = df.interpolate(method='time')
        df = df.dropna(subset=['pm25'])

    df = df.reset_index().rename(columns={'index'})
    return df

def add_time_features(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    timestamp_s = df['created_at'].map(pd.Timestamp.timestamp)
    day = 24*60*60
    year = (365.2425)*day
    
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    
    df_f = df.drop(columns=['created_at', 'id', 'year', 'month','sht31_temp','sht31_hum'], errors='ignore')
    df_f['pm25_diff'] = df_f['pm25'].diff().bfill()
    return df_f

def split_and_scale(df_f, handle_outlier=True):
    n = len(df_f)
    train_df = df_f[0:int(n*0.7)].copy()
    val_df = df_f[int(n*0.7):int(n*0.9)].copy()
    test_df = df_f[int(n*0.9):].copy()

    cols = ['temperature', 'humidity', 'pm25']
    
    if handle_outlier:
        cap_params = {}
        for col in cols:
            lower = train_df[col].quantile(0.01)
            upper = train_df[col].quantile(0.99)
            cap_params[col] = (lower, upper)
            train_df[col] = train_df[col].clip(lower, upper)
            val_df[col] = val_df[col].clip(lower, upper)
            test_df[col] = test_df[col].clip(lower, upper)
        joblib.dump(cap_params, 'cap_params.pkl')

    scaler = StandardScaler()
    scaler.fit(train_df[cols])
    joblib.dump(scaler, 'scaler.pkl')

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[cols] = scaler.transform(train_df[cols])
    val_scaled[cols] = scaler.transform(val_df[cols])
    test_scaled[cols] = scaler.transform(test_df[cols])
    
    return train_scaled, val_scaled, test_scaled, scaler