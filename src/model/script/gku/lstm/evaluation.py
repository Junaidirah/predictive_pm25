import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def evaluate_detailed(model, window_gen, scaler):
    all_preds, all_labels = [], []
    for inputs, labels in window_gen.test:
        preds = model.predict(inputs, verbose=0)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()

    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels)**2))

    ss_res = np.sum((labels - preds)**2)
    ss_tot = np.sum((labels - np.mean(labels))**2)
    r2 = 1 - (ss_res / ss_tot)

    pm25_std = scaler.scale_[2]
    pm25_mean = scaler.mean_[2]

    preds_real = (preds * pm25_std) + pm25_mean
    labels_real = (labels * pm25_std) + pm25_mean

    mae_real = np.mean(np.abs(preds_real - labels_real))
    rmse_real = np.sqrt(np.mean((preds_real - labels_real)**2))
    mape_real = np.mean(np.abs((labels_real - preds_real) / (labels_real + 1e-8))) * 100

    print(f"\n[METRICS - Z-Score Scale]")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    print(f"\n[METRICS - Asli (ug/m3)]")
    print(f"MAE:  {mae_real:.2f}")
    print(f"RMSE: {rmse_real:.2f}")
    print(f"MAPE: {mape_real:.2f}%")
    
    if r2 > 0.5:
        print(f"\n>>> TARGET TERCAPAI! R2 = {r2:.4f} > 0.5")
    else:
        print(f"\n>>> Target belum tercapai. R2 = {r2:.4f}")

    metrics = {
        'MAE': mae_real,
        'RMSE': rmse_real,
        'MAPE': mape_real,
        'R2 (%)': r2 * 100
    }

    # Menyimpan metrics ke tabel CSV
    csv_path = 'metrik_performa_gku.csv'
    komplit_metrik = {
        'Model': 'GKU PURE LSTM',
        'Z_MAE': round(mae, 4),
        'Z_RMSE': round(rmse, 4),
        'Z_R2': round(r2, 4),
        'ASLI_MAE': round(mae_real, 4),
        'ASLI_RMSE': round(rmse_real, 4),
        'ASLI_MAPE(%)': round(mape_real, 4),
        'ASLI_R2(%)': round(r2 * 100, 4)
    }
    df_new = pd.DataFrame([komplit_metrik])
    
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
        
    df_new.to_csv(csv_path, index=False)
    print(f"\n[+] Metrik Performa berhasil direkap dalam wujud tabel ke '{csv_path}'")

    return preds_real, labels_real, r2, metrics

def plot_results(labels, preds, r2, history=None, metrics=None):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Dashboard Evaluasi Model GKU MURNI LSTM', fontsize=18, weight='bold')

    # 1. Kurva Training-Validation Loss
    ax1 = fig.add_subplot(2, 3, 1)
    if history is not None:
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'History tidak tersedia', ha='center', va='center')
        ax1.set_title('Training & Validation Loss')

    # 2. Grafik aktual vs prediksi (Sample Time Series)
    ax2 = fig.add_subplot(2, 3, 2)
    sample_size = min(200, len(labels))
    ax2.plot(range(sample_size), labels[:sample_size], 'b-', label='Actual', linewidth=2)
    ax2.plot(range(sample_size), preds[:sample_size], 'r--', label='Predicted', linewidth=2)
    ax2.set_xlabel('Time Step (Jam)')
    ax2.set_ylabel('PM2.5 (ug/m3)')
    ax2.set_title('Grafik Aktual vs Prediksi (200 Jam)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Scatter aktual vs prediksi
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(labels, preds, alpha=0.5, s=20, color='purple')
    min_val = min(np.min(labels), np.min(preds))
    max_val = max(np.max(labels), np.max(preds))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit', linewidth=2)
    ax3.set_xlabel('Actual PM2.5 (ug/m3)')
    ax3.set_ylabel('Predicted PM2.5 (ug/m3)')
    ax3.set_title(f'Scatter Aktual vs Prediksi (R² = {r2:.4f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Bar Chart Metrik Performa
    ax4 = fig.add_subplot(2, 3, 4)
    if metrics is not None:
        names = list(metrics.keys())
        values = list(metrics.values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax4.bar(names, values, color=colors)
        ax4.set_title('Tabel/Bar Chart Metrik Performa')
        for bar in bars:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.05), f'{yval:.2f}', ha='center', va='bottom', fontsize=11)
        ax4.set_ylim(0, max(values) * 1.2)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Analisis Residual (Scatter)
    ax5 = fig.add_subplot(2, 3, 5)
    residuals = labels - preds
    ax5.scatter(preds, residuals, alpha=0.5, color='brown', s=20)
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax5.set_title('Analisis Residual vs Prediksi')
    ax5.set_xlabel('Predicted PM2.5 (ug/m3)')
    ax5.set_ylabel('Residual (Actual - Predicted)')
    ax5.grid(True, alpha=0.3)

    # 6. Distribusi Residual
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(residuals, bins=50, color='teal', alpha=0.7, edgecolor='black')
    ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax6.set_title('Distribusi Histogram Residual (Error)')
    ax6.set_xlabel('Error PM2.5 (ug/m3)')
    ax6.set_ylabel('Frekuensi')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('dashboard_evaluasi_gku_lstm.png', dpi=300)
    print("\n[+] Dashboard Evaluasi berhasil disimpan ke 'dashboard_evaluasi_gku_lstm.png'")