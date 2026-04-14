import tensorflow as tf
import os
from data_processing import load_and_interpolate, add_time_features, split_and_scale
from window_generator import WindowGenerator
from model import build_lstm, R2Callback
from evaluation import evaluate_detailed, plot_results

def main():
    print("="*60)
    print("MENGJALANKAN PIPELINE PREDIKSI PM2.5 (PURE LSTM)")
    print("="*60)
    print("KONFIGURASI EKSPERIMEN TULT (PURE LSTM)")
    print("> Lokasi Data   : Gedung TULT")
    print("> Horizon       : 1 Jam Ke Depan")
    print("> Input Width   : 24 Jam")
    print("> Label Width   : 1 Jam")
    print("> Model         : Bi-Directional LSTM")
    print("> Max Epochs    : 100")
    print("="*60)

    path = "D:/development/predictive_pm25/data/training/data_training_tult.csv"
    if not os.path.exists(path):
        print(f"File {path} tidak ditemukan!")
        return
        
    print("[1/5] Memuat dan memproses data...")
    df = load_and_interpolate(path)
    df_f = add_time_features(df)
    train_scaled, val_scaled, test_scaled, scaler = split_and_scale(df_f)

    FEATURE_COLS = ['pm25', 'temperature', 'humidity', 'pm25_diff', 'Day sin', 'Day cos', 'Year sin', 'Year cos']
    LABEL_COLS = ['pm25']

    print("[2/5] Menyiapkan WindowGenerator...")
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

    # Membangun Model
    model = build_lstm(24, len(FEATURE_COLS))
    
    # Compile Model (Init LR: 1e-4, Decay ditangani oleh ReduceLROnPlateau)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
    )
    model.summary()

    print("[4/5] Melatih model...")
    history = model.fit(
        multi_window.train,
        validation_data=multi_window.val,
        epochs=100, # Dinaikkan agar model bisa belajar lebih lama
        callbacks=[
            R2Callback(multi_window.val),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]
    )

    print("[5/5] Evaluasi Model...")
    preds, labels, r2, metrics = evaluate_detailed(model, multi_window, scaler)
    plot_results(labels, preds, r2, history=history, metrics=metrics)
    
    model.save('pm25_model_pure_lstm.keras')
    print("Selesai! Model tersimpan di 'pm25_model_pure_lstm.keras'")

if __name__ == '__main__':
    main()
