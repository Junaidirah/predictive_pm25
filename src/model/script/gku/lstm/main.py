import tensorflow as tf
import os
from data_processing import load_and_interpolate, add_time_features, normalize, split_data
from window_generator import WindowGenerator
from model import build_lstm, R2Callback
from evaluation import evaluate_detailed
from analyze_distribution import print_stats
from dashboard import plot_master_dashboard
from config import FEATURE_COLS, LABEL_COLS

def main():
    print("="*60)
    print("MENGJALANKAN PIPELINE PREDIKSI PM2.5 (PURE LSTM)")
    print("="*60)
    print("KONFIGURASI EKSPERIMEN GKU (PURE LSTM)")
    print("> Lokasi Data   : Gedung GKU")
    print("> Horizon       : 1 Jam Ke Depan")
    print("> Input Width   : 24 Jam")
    print("> Label Width   : 1 Jam")
    print("> Model         : Bi-Directional LSTM")
    print("> Max Epochs    : 150")
    print("="*60)

    path = "D:/development/predictive_pm25/data/training/data_training_GKU.csv"
    if not os.path.exists(path):
        print(f"File {path} tidak ditemukan!")
        return
        
    print("[1/5] Memuat dan memproses data...")
    df = load_and_interpolate(path)
    df_f = add_time_features(df)

    print("\n[2/5] Normalisasi seluruh dataset (RobustScaler + Clip)...")
    df_ready, scaler = normalize(df_f)

    print("\n[3/5] Split data yang sudah dinormalisasi...")
    train_scaled, val_scaled, test_scaled = split_data(df_ready)

    multi_window = WindowGenerator(
        input_width=24,
        label_width=1,
        shift=1,
        train_df=train_scaled,
        val_df=val_scaled,
        test_df=test_scaled,
        feature_columns=FEATURE_COLS,
        label_columns=LABEL_COLS
    )

    print("\n[4/5] Membangun & melatih model LSTM...")
    model = build_lstm(24, len(FEATURE_COLS))
    # Compile Model (Init LR: 1e-4, Decay ditangani oleh ReduceLROnPlateau)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
    )
    model.summary()

    history = model.fit(
        multi_window.train,
        validation_data=multi_window.val,
        epochs=100,
        callbacks=[
            R2Callback(multi_window.val),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]
    )

    print("\n" + "="*55)
    print("[STATIC EVAL] Evaluasi tanpa Dropout (bobot terkunci)")
    train_loss, train_mae = model.evaluate(multi_window.train, verbose=0)
    val_loss,   val_mae   = model.evaluate(multi_window.val,   verbose=0)
    print(f"  Train Loss : {train_loss:.4f}  |  Train MAE : {train_mae:.4f}")
    print(f"  Val Loss   : {val_loss:.4f}  |  Val MAE   : {val_mae:.4f}")
    ratio = val_loss / train_loss
    print(f"  Rasio Val/Train : {ratio:.3f}")
    if ratio < 0.90:
        print("  >> Val jauh lebih rendah: distribusi val lebih mudah dari train.")
    elif ratio > 1.25:
        print("  >> OVERFITTING: Val loss jauh lebih tinggi dari train.")
    else:
        print("  >> Sehat: perbedaan train/val dalam batas wajar.")
    print("="*55)

    print("[5/5] Evaluasi & Render Master Dashboard...")
    preds, labels, r2, metrics = evaluate_detailed(model, multi_window, scaler)

    print("\n[INFO] Statistik distribusi per split setelah normalisasi:")
    print_stats(train_scaled, val_scaled, test_scaled)

    plot_master_dashboard(
        train_df=train_scaled,
        val_df=val_scaled,
        test_df=test_scaled,
        labels=labels,
        preds=preds,
        r2=r2,
        metrics=metrics,
        history=history,
        save_path='dashboard_master_gku_lstm.png'
    )

    model.save('pm25_model_1hour.keras')
    print("Selesai! Model tersimpan di 'pm25_model_1hour.keras'")

if __name__ == '__main__':
    main()