import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Get absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change working directory to script directory so relative file loading works flawlessly
os.chdir(script_dir)

# Add parent directory to path so we can import data_processing
sys.path.append(script_dir)
from data_processing import load_and_interpolate, add_time_features

def calculate_ispu(x_x):
    """
    Menghitung Nilai ISPU berdasarkan Permen LHK No. 14 Tahun 2020 untuk parameter PM2.5.
    Rumus:
    I = ((I_a - I_b) / (X_a - X_b)) * (X_x - X_b) + I_b
    
    I: Nilai ISPU terhitung (yang ingin dicari)
    I_a: Nilai ISPU batas atas untuk kategori konsentrasi yang bersangkutan
    I_b: Nilai ISPU batas bawah untuk kategori konsentrasi yang bersangkutan
    X_a: Konsentrasi ambien batas atas (ug/m3) untuk kategori tersebut
    X_b: Konsentrasi ambien batas bawah (ug/m3) untuk kategori tersebut
    X_x: Konsentrasi ambien nyata PM2.5 hasil pengukuran di lapangan (ug/m3)
    """
    if x_x < 0:
        x_x = 0.0

    # Menentukan kategori konsentrasi PM2.5 berdasarkan Permen LHK 14/2020
    if x_x <= 15.5:
        Ia, Ib = 50, 0
        Xa, Xb = 15.5, 0.0
    elif x_x <= 55.4:
        Ia, Ib = 100, 51
        Xa, Xb = 55.4, 15.5
    elif x_x <= 150.4:
        Ia, Ib = 200, 101
        Xa, Xb = 150.4, 55.4
    elif x_x <= 250.4:
        Ia, Ib = 300, 201
        Xa, Xb = 250.4, 150.4
    else:
        Ia, Ib = 500, 301
        Xa, Xb = 500.0, 250.4
        # Capping PM2.5 at 500 to prevent division errors/extreme values if extrapolated
        if x_x > 500.0:
            x_x = 500.0
            
    # Perhitungan rumus interpolasi linear piecewise ISPU
    ispu = ((Ia - Ib) / (Xa - Xb)) * (x_x - Xb) + Ib
    return int(round(ispu))

def get_ews_status(ispu):
    """
    Mengambil status kategori EWS dan rekomendasi tindakan berdasarkan nilai ISPU.
    """
    if ispu <= 50:
        category = "Baik"
        action = "Aman untuk beraktivitas di luar ruangan."
    elif ispu <= 100:
        category = "Sedang"
        action = "Kualitas udara sedang. Kelompok sensitif sebaiknya mengurangi aktivitas fisik berat."
    elif ispu <= 200:
        category = "Tidak Sehat"
        action = "PENTING: Kurangi aktivitas fisik berat di luar ruangan. Gunakan masker jika harus keluar."
    elif ispu <= 300:
        category = "Sangat Tidak Sehat"
        action = "WARNING: Hindari semua aktivitas luar ruangan bagi kelompok rentan. Batasi bagi umum."
    else:
        category = "Berbahaya"
        action = "CAUTION: Tingkat berbahaya! Dilarang beraktivitas di luar ruangan. Tetap di dalam ruangan."
    return category, action

def generate_forecast():
    print("=" * 70)
    print("      GENERATING 7-DAY HOURLY FORECAST & EWS (CNN-LSTM - DELI)      ")
    print("=" * 70)
    
    # 1. Paths configuration
    model_path = 'pm25_model_1hour.keras'
    scaler_path = 'scaler.pkl'
    cap_path = 'cap_params.pkl'
    
    data_paths = [
        "D:/development/predictive_pm25/data/training/data_training_deli.csv",
        "d:/predictive_pm25/data/training/data_training_deli.csv",
        os.path.join(script_dir, "../../../../data/training/data_training_deli.csv"),
        os.path.join(script_dir, "../../../data/training/data_training_deli.csv")
    ]
    
    data_path = None
    for p in data_paths:
        if os.path.exists(p):
            data_path = p
            break
            
    if data_path is None:
        print("❌ Error: File data training data_training_deli.csv tidak ditemukan!")
        return
        
    print(f"[+] Menggunakan dataset: {data_path}")
    
    # 2. Load model, scaler, and cap_params
    if not os.path.exists(model_path):
        print(f"❌ Error: Model {model_path} tidak ditemukan!")
        return
    if not os.path.exists(scaler_path):
        print(f"❌ Error: Scaler {scaler_path} tidak ditemukan!")
        return
    if not os.path.exists(cap_path):
        print(f"❌ Error: Outlier clip parameters {cap_path} tidak ditemukan!")
        return
        
    print("[1/5] Memuat model CNN-LSTM dan scaler...")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    cap_params = joblib.load(cap_path)
    
    # 3. Load and preprocess initial data
    print("[2/5] Memproses data historis teraktual...")
    df = load_and_interpolate(data_path)
    df_f = add_time_features(df)
    
    # Ambil 24 data terakhir sebagai basis history window (input_width=24)
    last_timestamp = df['created_at'].max()
    print(f"[+] Waktu data historis terakhir: {last_timestamp}")
    
    history_unscaled = df_f.iloc[-24:].copy().reset_index(drop=True)
    history_scaled = history_unscaled.copy()
    
    # Skalakan data history awal
    cols = ['temperature', 'humidity', 'pm25']
    for col in cols:
        lower, upper = cap_params[col]
        history_scaled[col] = history_scaled[col].clip(lower, upper)
    history_scaled[cols] = scaler.transform(history_scaled[cols])
    
    # 4. Autoregressive Forecasting loop for 168 hours (24 hours * 7 days)
    print("[3/5] Melakukan peramalan autoregresif 168 jam ke depan...")
    
    FEATURE_COLS = ['pm25', 'temperature', 'humidity', 'pm25_diff', 'Day sin', 'Day cos', 'Year sin', 'Year cos']
    forecast_records = []
    
    for h in range(1, 169):
        # Ambil window terakhir untuk input ke model
        input_data = history_scaled[FEATURE_COLS].values.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0) # Shape: (1, 24, 8)
        
        # Prediksi pm25 skala z-score
        pred_scaled_pm25 = model.predict(input_data, verbose=0)[0, 0]
        
        # Inverse transform untuk mendapatkan nilai PM2.5 asli
        pred_pm25 = (pred_scaled_pm25 * scaler.scale_[2]) + scaler.mean_[2]
        if pred_pm25 < 0:
            pred_pm25 = 0.0
            
        next_timestamp = last_timestamp + pd.Timedelta(hours=h)
        
        # Dapatkan nilai temperature & humidity dari 24 jam sebelumnya (pola harian)
        temp_val = history_unscaled.loc[len(history_unscaled) - 24, 'temperature']
        hum_val = history_unscaled.loc[len(history_unscaled) - 24, 'humidity']
        
        # Hitung pm25_diff (unscaled)
        prev_pm25 = history_unscaled.loc[len(history_unscaled) - 1, 'pm25']
        pm25_diff = pred_pm25 - prev_pm25
        
        # Hitung time features untuk jam ini
        timestamp_s = next_timestamp.timestamp()
        day = 24 * 60 * 60
        year = 365.2425 * day
        day_sin = np.sin(timestamp_s * (2 * np.pi / day))
        day_cos = np.cos(timestamp_s * (2 * np.pi / day))
        year_sin = np.sin(timestamp_s * (2 * np.pi / year))
        year_cos = np.cos(timestamp_s * (2 * np.pi / year))
        
        # Clip dan standardisasikan fitur numerik baru
        temp_val_clipped = np.clip(temp_val, cap_params['temperature'][0], cap_params['temperature'][1])
        hum_val_clipped = np.clip(hum_val, cap_params['humidity'][0], cap_params['humidity'][1])
        pred_pm25_clipped = np.clip(pred_pm25, cap_params['pm25'][0], cap_params['pm25'][1])
        
        scaled_feats = scaler.transform([[temp_val_clipped, hum_val_clipped, pred_pm25_clipped]])[0]
        scaled_temp = scaled_feats[0]
        scaled_hum = scaled_feats[1]
        scaled_pm25 = scaled_feats[2]
        
        # Append ke history unscaled
        new_row_unscaled = {
            'pm25': pred_pm25,
            'temperature': temp_val,
            'humidity': hum_val,
            'pm25_diff': pm25_diff,
            'Day sin': day_sin,
            'Day cos': day_cos,
            'Year sin': year_sin,
            'Year cos': year_cos
        }
        history_unscaled = pd.concat([history_unscaled, pd.DataFrame([new_row_unscaled])], ignore_index=True)
        history_unscaled = history_unscaled.iloc[1:].reset_index(drop=True)
        
        # Append ke history scaled
        new_row_scaled = {
            'pm25': scaled_pm25,
            'temperature': scaled_temp,
            'humidity': scaled_hum,
            'pm25_diff': pm25_diff,
            'Day sin': day_sin,
            'Day cos': day_cos,
            'Year sin': year_sin,
            'Year cos': year_cos
        }
        history_scaled = pd.concat([history_scaled, pd.DataFrame([new_row_scaled])], ignore_index=True)
        history_scaled = history_scaled.iloc[1:].reset_index(drop=True)
        
        # Konversi ISPU dan hitung EWS
        ispu_val = calculate_ispu(pred_pm25)
        ews_cat, ews_act = get_ews_status(ispu_val)
        
        forecast_records.append({
            'timestamp': next_timestamp,
            'predicted_pm25': round(pred_pm25, 2),
            'ispu_value': ispu_val,
            'ews_category': ews_cat,
            'ews_action': ews_act
        })
        
    # 5. Pack into DataFrame
    forecast_df = pd.DataFrame(forecast_records)
    
    # 6. Save results to CSV files
    print("[4/5] Menyimpan hasil prediksi dan EWS ke dalam CSV...")
    csv_local = 'prediksi_dan_ews_deli.csv'
    forecast_df.to_csv(csv_local, index=False)
    print(f"[+] Berhasil disimpan secara lokal di: {os.path.abspath(csv_local)}")
    
    # Simpan juga ke direktori data jika tersedia
    dest_data_dir = os.path.join(script_dir, "../../../../data")
    if os.path.exists(dest_data_dir):
        csv_dest = os.path.join(dest_data_dir, 'prediksi_dan_ews_deli_cnn_lstm.csv')
        forecast_df.to_csv(csv_dest, index=False)
        print(f"[+] Berhasil disalin ke direktori data di: {os.path.abspath(csv_dest)}")
        
    # 7. Print Beautiful Table for next 24 hours (Hourly) and next 7 days (Daily summary)
    print("\n" + "="*80)
    print("                PREDIKSI PER JAM UNTUK 24 JAM PERTAMA (EWS)                ")
    print("="*80)
    print(f"{'Waktu':<20} | {'PM2.5 (ug/m3)':<14} | {'ISPU':<6} | {'Kategori EWS':<20}")
    print("-"*80)
    
    for i in range(24):
        row = forecast_df.iloc[i]
        time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M')
        print(f"{time_str:<20} | {row['predicted_pm25']:<14.2f} | {row['ispu_value']:<6} | {row['ews_category']:<20}")
        
    print("\n" + "="*80)
    print("                     RINGKASAN HARIAN UNTUK 7 HARI KE DEPAN                   ")
    print("="*80)
    print(f"{'Hari Ke-':<8} | {'Tanggal':<12} | {'Rata-Rata PM2.5':<16} | {'Max ISPU':<8} | {'EWS Terburuk':<20}")
    print("-"*80)
    
    forecast_df['date'] = forecast_df['timestamp'].dt.date
    daily_groups = forecast_df.groupby('date')
    
    day_idx = 1
    for date, group in daily_groups:
        avg_pm25 = group['predicted_pm25'].mean()
        max_ispu = group['ispu_value'].max()
        
        # Get worst category of the day
        worst_cat = "Baik"
        categories_order = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]
        for cat in group['ews_category'].unique():
            if categories_order.index(cat) > categories_order.index(worst_cat):
                worst_cat = cat
                
        print(f"Hari {day_idx:<3} | {date.strftime('%Y-%m-%d'):<12} | {avg_pm25:<16.2f} | {max_ispu:<8} | {worst_cat:<20}")
        day_idx += 1
        
    print("="*80)
    print("[5/5] Selesai! Seluruh data prediksi per jam telah berhasil dimuat ke dalam DataFrame.")
    print("="*80)

if __name__ == '__main__':
    generate_forecast()
