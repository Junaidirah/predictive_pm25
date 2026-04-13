import numpy as np
import matplotlib.pyplot as plt

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
    mape = np.mean(np.abs((labels - preds) / (labels + 1e-8))) * 100

    ss_res = np.sum((labels - preds)**2)
    ss_tot = np.sum((labels - np.mean(labels))**2)
    r2 = 1 - (ss_res / ss_tot)

    pm25_std = scaler.scale_[2]
    pm25_mean = scaler.mean_[2]

    mae_real = mae * pm25_std
    rmse_real = rmse * pm25_std

    print(f"\n[METRICS - Z-Score Scale]")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    print(f"\n[METRICS - Asli (ug/m3)]")
    print(f"MAE:  {mae_real:.2f}")
    print(f"RMSE: {rmse_real:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    if r2 > 0.5:
        print(f"\n>>> TARGET TERCAPAI! R2 = {r2:.4f} > 0.5")
    else:
        print(f"\n>>> Target belum tercapai. R2 = {r2:.4f}")

    return preds, labels, r2, mae_real

def plot_results(labels, preds, r2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(labels, preds, alpha=0.5, s=20)
    axes[0].plot([-3, 3], [-3, 3], 'r--', label='Perfect Prediction')
    axes[0].set_xlabel('Actual PM2.5 (Z-score)')
    axes[0].set_ylabel('Predicted PM2.5 (Z-score)')
    axes[0].set_title(f'Prediksi vs Aktual (R2 = {r2:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    sample_size = 200
    axes[1].plot(range(sample_size), labels[:sample_size], 'b-', label='Actual', linewidth=2)
    axes[1].plot(range(sample_size), preds[:sample_size], 'r--', label='Predicted', linewidth=2)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('PM2.5 (Z-score)')
    axes[1].set_title('Sample Prediksi 1 Jam ke Depan')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('result_plot.png')
    print("Grafik disimpan ke 'result_plot.png'")