"""dashboard.py — Master Dashboard GKU LSTM

Mengumpulkan semua panel visualisasi dalam satu figure besar (4 baris × 3 kolom
= 12 panel) yang mencakup:

  Baris 1 — Distribusi Data (Pra-Evaluasi)
    [1] Violin plot distribusi fitur setelah normalisasi (Train set)
    [2] Boxplot perbandingan PM2.5 train/val/test
    [3] Statistik deskriptif sebagai tabel teks

  Baris 2 — Proses Training
    [4] Kurva Train & Validation Loss
    [5] Kurva Train & Validation MAE
    [6] Rasio val/train loss per epoch (deteksi overfitting)

  Baris 3 — Hasil Prediksi
    [7] Time-series: Aktual vs Prediksi (200 jam pertama)
    [8] Scatter: Aktual vs Prediksi (semua test set)
    [9] Scatter Residual vs Prediksi

  Baris 4 — Analisis Error
    [10] Histogram distribusi residual
    [11] Bar chart metrik performa (MAE, RMSE, MAPE, R²)
    [12] Running MAPE — galat kumulatif per time step
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from config import FEATURE_COLS, SCALED_COLS, CYCLICAL_COLS

# ─────────────────────────────────────────────────────────────────────────────
# Palet warna
# ─────────────────────────────────────────────────────────────────────────────
C_TRAIN  = '#2196F3'   # biru
C_VAL    = '#FF9800'   # oranye
C_TEST   = '#4CAF50'   # hijau
C_PRED   = '#E91E63'   # pink/merah
C_ACCENT = '#9C27B0'   # ungu
C_GRID   = '#E0E0E0'

_SCALED_SET = set(SCALED_COLS)


def _style_ax(ax, title, xlabel='', ylabel=''):
    """Terapkan gaya konsisten ke setiap subplot."""
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_facecolor('#FAFAFA')
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Panel individu
# ─────────────────────────────────────────────────────────────────────────────

def _panel_violin(ax, train_df):
    """[1] Violin distribusi fitur (Train set) setelah normalisasi."""
    cols = [c for c in FEATURE_COLS if c in train_df.columns]
    data = [train_df[c].dropna().values for c in cols]
    parts = ax.violinplot(data, positions=range(len(cols)),
                          showmedians=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(C_TRAIN if cols[i] in _SCALED_SET else C_VAL)
        pc.set_alpha(0.75)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha='right', fontsize=7)
    ax.axhline(0,  color='gray', ls='--', lw=0.8, alpha=0.6)
    ax.axhline(3,  color='red',  ls=':',  lw=1.0, alpha=0.7)
    ax.axhline(-3, color='red',  ls=':',  lw=1.0, alpha=0.7)
    legend_els = [Patch(facecolor=C_TRAIN, alpha=0.75, label='Scaled (sensor)'),
                  Patch(facecolor=C_VAL,   alpha=0.75, label='Cyclical (waktu)'),
                  Line2D([0],[0], color='red', ls=':', label='±3 batas')]
    ax.legend(handles=legend_els, fontsize=6.5, loc='upper right')
    _style_ax(ax, 'Violin — Distribusi Fitur Setelah Normalisasi (Train)',
              ylabel='Nilai Normalized')


def _panel_boxplot_pm25(ax, train_df, val_df, test_df):
    """[2] Boxplot PM2.5 train / val / test."""
    data   = [train_df['pm25'].dropna().values,
              val_df['pm25'].dropna().values,
              test_df['pm25'].dropna().values]
    labels = ['Train', 'Val', 'Test']
    colors = [C_TRAIN, C_VAL, C_TEST]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0,  color='gray', ls='--', lw=0.8, alpha=0.6)
    ax.axhline(3,  color='red',  ls=':',  lw=1.0, alpha=0.7)
    ax.axhline(-3, color='red',  ls=':',  lw=1.0, alpha=0.7)
    _style_ax(ax, 'Boxplot — PM2.5 per Split (Normalized)',
              ylabel='Nilai Normalized')


def _panel_stats_table(ax, train_df, val_df, test_df):
    """[3] Tabel statistik deskriptif PM2.5 (min/max/mean/std/median) per split."""
    ax.axis('off')
    splits = {'Train': train_df, 'Val': val_df, 'Test': test_df}
    col_labels = ['Split', 'Min', 'Max', 'Mean', 'Std', 'Median', 'N']
    rows = []
    for name, df in splits.items():
        v = df['pm25'].dropna()
        rows.append([name,
                     f'{v.min():.3f}', f'{v.max():.3f}',
                     f'{v.mean():.3f}', f'{v.std():.3f}',
                     f'{v.median():.3f}', str(len(v))])
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#37474F')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 1:
            cell.set_facecolor('#ECEFF1')
        cell.set_edgecolor('#CFD8DC')
    ax.set_title('Statistik Deskriptif PM2.5 (Normalized)',
                 fontsize=10, fontweight='bold', pad=8)


def _panel_loss(ax, history):
    """[4] Kurva train & val loss."""
    if history is not None:
        epochs = range(1, len(history.history['loss']) + 1)
        ax.plot(epochs, history.history['loss'],     color=C_TRAIN, lw=1.5, label='Train Loss')
        ax.plot(epochs, history.history['val_loss'], color=C_VAL,   lw=1.5, label='Val Loss', ls='--')
        best_ep = int(np.argmin(history.history['val_loss'])) + 1
        best_vl = min(history.history['val_loss'])
        ax.axvline(best_ep, color='red', ls=':', lw=1, alpha=0.7, label=f'Best ep={best_ep}')
        ax.scatter([best_ep], [best_vl], color='red', zorder=5, s=30)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'History tidak tersedia', ha='center', va='center',
                transform=ax.transAxes)
    _style_ax(ax, 'Training & Validation Loss', 'Epoch', 'Loss (Huber)')


def _panel_mae(ax, history):
    """[5] Kurva train & val MAE."""
    if history is not None:
        epochs = range(1, len(history.history['mae']) + 1)
        ax.plot(epochs, history.history['mae'],     color=C_TRAIN, lw=1.5, label='Train MAE')
        ax.plot(epochs, history.history['val_mae'], color=C_VAL,   lw=1.5, label='Val MAE', ls='--')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'History tidak tersedia', ha='center', va='center',
                transform=ax.transAxes)
    _style_ax(ax, 'Training & Validation MAE', 'Epoch', 'MAE')


def _panel_val_ratio(ax, history):
    """[6] Rasio val_loss / train_loss per epoch — deteksi overfitting."""
    if history is not None:
        train_l = np.array(history.history['loss'])
        val_l   = np.array(history.history['val_loss'])
        ratio   = val_l / (train_l + 1e-9)
        epochs  = range(1, len(ratio) + 1)
        ax.plot(epochs, ratio, color=C_ACCENT, lw=1.5, label='val/train loss')
        ax.axhline(1.0, color='green', ls='--', lw=1, alpha=0.7, label='Rasio ideal = 1.0')
        ax.axhline(1.2, color='red',   ls=':',  lw=1, alpha=0.7, label='Threshold overfit 1.2')
        ax.fill_between(epochs, 1.0, ratio,
                        where=np.array(ratio) > 1.2,
                        color='red', alpha=0.15, label='Zona overfit')
        ax.legend(fontsize=6.5)
    else:
        ax.text(0.5, 0.5, 'History tidak tersedia', ha='center', va='center',
                transform=ax.transAxes)
    _style_ax(ax, 'Rasio Val/Train Loss per Epoch (Deteksi Overfitting)',
              'Epoch', 'Rasio')


def _panel_timeseries(ax, labels, preds, n=300):
    """[7] Time-series aktual vs prediksi (n jam pertama)."""
    size = min(n, len(labels))
    t = range(size)
    ax.fill_between(t, labels[:size], preds[:size],
                    alpha=0.15, color=C_ACCENT, label='Selisih')
    ax.plot(t, labels[:size], color=C_TRAIN, lw=1.5, label='Aktual')
    ax.plot(t, preds[:size],  color=C_PRED,  lw=1.5, ls='--', label='Prediksi')
    ax.legend(fontsize=7)
    _style_ax(ax, f'Time-Series Aktual vs Prediksi ({size} jam)',
              'Time Step (Jam)', 'PM2.5 (µg/m³)')


def _panel_scatter(ax, labels, preds, r2):
    """[8] Scatter aktual vs prediksi."""
    ax.scatter(labels, preds, alpha=0.35, s=12, color=C_ACCENT, zorder=3)
    lo = min(np.min(labels), np.min(preds))
    hi = max(np.max(labels), np.max(preds))
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect Fit')
    # Garis regresi
    z = np.polyfit(labels, preds, 1)
    p = np.poly1d(z)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, p(xs), color=C_TRAIN, lw=1.2, ls='-.', label='Regresi aktual')
    ax.legend(fontsize=7)
    ax.text(0.05, 0.93, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=9, fontweight='bold', color=C_ACCENT,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    _style_ax(ax, 'Scatter — Aktual vs Prediksi',
              'Aktual PM2.5 (µg/m³)', 'Prediksi PM2.5 (µg/m³)')


def _panel_residual_scatter(ax, labels, preds):
    """[9] Scatter residual vs prediksi."""
    residuals = labels - preds
    ax.scatter(preds, residuals, alpha=0.35, s=12, color='#FF5722', zorder=3)
    ax.axhline(0, color='black', lw=1.5, ls='--')
    ax.axhline(residuals.std() * 2,  color='red', lw=0.8, ls=':', alpha=0.6)
    ax.axhline(-residuals.std() * 2, color='red', lw=0.8, ls=':', alpha=0.6,
               label='±2σ')
    ax.legend(fontsize=7)
    _style_ax(ax, 'Residual vs Prediksi',
              'Prediksi PM2.5 (µg/m³)', 'Residual (Aktual − Prediksi)')


def _panel_residual_hist(ax, labels, preds):
    """[10] Histogram distribusi residual."""
    residuals = labels - preds
    n, bins, patches = ax.hist(residuals, bins=50, color='teal',
                                alpha=0.75, edgecolor='white', linewidth=0.4)
    ax.axvline(0,                color='red',   lw=1.5, ls='--', label='Nol')
    ax.axvline(residuals.mean(), color='orange', lw=1.2, ls='-.',
               label=f'Mean={residuals.mean():.2f}')
    ax.legend(fontsize=7)
    _style_ax(ax, 'Distribusi Histogram Residual',
              'Error (µg/m³)', 'Frekuensi')


def _panel_metrics_bar(ax, metrics):
    """[11] Bar chart metrik performa."""
    if metrics is None:
        ax.axis('off')
        return
    names  = list(metrics.keys())
    values = list(metrics.values())
    palette = [C_TRAIN, C_VAL, C_TEST, C_ACCENT]
    bars = ax.bar(names, values, color=palette[:len(names)],
                  edgecolor='white', linewidth=0.8, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.03,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.25)
    _style_ax(ax, 'Metrik Performa (Skala Asli µg/m³)',
              '', 'Nilai Metrik')


def _panel_running_mape(ax, labels, preds):
    """[12] Running MAPE — galat kumulatif per time step."""
    ape = np.abs((labels - preds) / (labels + 1e-8)) * 100
    running = np.cumsum(ape) / (np.arange(len(ape)) + 1)
    ax.plot(running, color=C_TEST, lw=1.5, label='Running MAPE')
    ax.axhline(running[-1], color='red', ls='--', lw=1,
               label=f'Final MAPE = {running[-1]:.2f}%')
    ax.fill_between(range(len(running)), running, alpha=0.15, color=C_TEST)
    ax.legend(fontsize=7)
    _style_ax(ax, 'Running MAPE Kumulatif',
              'Time Step (Jam)', 'MAPE (%)')


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi utama — Master Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_master_dashboard(train_df, val_df, test_df,
                          labels, preds, r2, metrics,
                          history=None,
                          save_path='dashboard_master_gku_lstm.png'):
    """Render semua 12 panel dalam satu figure dan simpan ke PNG.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Data split setelah normalisasi (dari split_data()).
    labels : np.ndarray
        Nilai aktual PM2.5 (skala asli, µg/m³) dari test set.
    preds : np.ndarray
        Nilai prediksi PM2.5 (skala asli, µg/m³) dari test set.
    r2 : float
        Koefisien determinasi R².
    metrics : dict
        {'MAE': float, 'RMSE': float, 'MAPE': float, 'R2 (%)': float}
    history : tf.keras.callbacks.History or None
        Riwayat training dari model.fit().
    save_path : str
        Path output gambar PNG.
    """
    fig = plt.figure(figsize=(24, 28))
    fig.patch.set_facecolor('#F5F5F5')

    fig.suptitle(
        'Master Dashboard — GKU Pure LSTM | PM2.5 Forecasting (1 Jam ke Depan)',
        fontsize=16, fontweight='bold', y=0.995,
        color='#212121'
    )

    # Layout: 4 baris × 3 kolom dengan sedikit extra space
    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.48,
        wspace=0.35,
        top=0.97, bottom=0.04,
        left=0.06, right=0.97
    )

    # ── Baris 1: Distribusi data ──────────────────────────────────────────── #
    ax1  = fig.add_subplot(gs[0, 0:2])   # violin lebih lebar
    ax2  = fig.add_subplot(gs[0, 2])     # boxplot PM2.5
    # tidak ada ax3 di grid—ganti dengan tabel menggunakan subplot terpisah
    ax3  = fig.add_subplot(gs[1, 0])

    # ── Baris 2: Training ─────────────────────────────────────────────────── #
    ax4  = fig.add_subplot(gs[1, 1])
    ax5  = fig.add_subplot(gs[1, 2])

    # ── Baris 3: Prediksi ─────────────────────────────────────────────────── #
    ax6  = fig.add_subplot(gs[2, 0:2])   # time-series lebih lebar
    ax7  = fig.add_subplot(gs[2, 2])     # scatter

    # ── Baris 4: Error ───────────────────────────────────────────────────── #
    ax8  = fig.add_subplot(gs[3, 0])
    ax9  = fig.add_subplot(gs[3, 1])
    ax10 = fig.add_subplot(gs[3, 2])

    # Tambahkan panel yang belum tercakup (rasio & residual scatter)
    # Susun ulang agar 12 panel masuk: perluas ke 5 baris
    plt.close(fig)

    # ── Layout final: 5 baris × 3 kolom = 15 slot (3 terakhir dikosongkan) ── #
    fig = plt.figure(figsize=(24, 32))
    fig.patch.set_facecolor('#F5F5F5')
    fig.suptitle(
        'Master Dashboard — GKU Pure LSTM  |  PM2.5 Forecasting  |  Horizon: 1 Jam',
        fontsize=15, fontweight='bold', y=0.998, color='#212121'
    )

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.52,
        wspace=0.38,
        top=0.97, bottom=0.04,
        left=0.06, right=0.97
    )

    # Baris 0: Distribusi
    ax_violin   = fig.add_subplot(gs[0, 0:2])   # [1] lebar 2 kolom
    ax_box      = fig.add_subplot(gs[0, 2])      # [2]

    # Baris 1: Training + tabel
    ax_tbl      = fig.add_subplot(gs[1, 0])      # [3] tabel statistik
    ax_loss     = fig.add_subplot(gs[1, 1])      # [4] loss
    ax_mae_p    = fig.add_subplot(gs[1, 2])      # [5] mae

    # Baris 2: Prediksi utama
    ax_ts       = fig.add_subplot(gs[2, 0:2])    # [6] time-series lebar
    ax_scatter  = fig.add_subplot(gs[2, 2])      # [7] scatter

    # Baris 3: Error analysis
    ax_res_sc   = fig.add_subplot(gs[3, 0])      # [8] residual scatter
    ax_res_hist = fig.add_subplot(gs[3, 1])      # [9] histogram
    ax_metrics  = fig.add_subplot(gs[3, 2])      # [10] bar metrik

    # Render setiap panel
    _panel_violin          (ax_violin,   train_df)
    _panel_boxplot_pm25    (ax_box,      train_df, val_df, test_df)
    _panel_stats_table     (ax_tbl,      train_df, val_df, test_df)
    _panel_loss            (ax_loss,     history)
    _panel_mae             (ax_mae_p,    history)
    _panel_timeseries      (ax_ts,       labels, preds)
    _panel_scatter         (ax_scatter,  labels, preds, r2)
    _panel_residual_scatter(ax_res_sc,   labels, preds)
    _panel_residual_hist   (ax_res_hist, labels, preds)
    _panel_metrics_bar     (ax_metrics,  metrics)

    # Label seksi besar
    _add_section_label(fig, gs, 0, 'SEKSI 1 — Analisis Distribusi Data (Pasca Normalisasi)')
    _add_section_label(fig, gs, 1, 'SEKSI 2 — Proses Training Model')
    _add_section_label(fig, gs, 2, 'SEKSI 3 — Hasil Prediksi Test Set')
    _add_section_label(fig, gs, 3, 'SEKSI 4 — Analisis Error & Metrik Performa')

    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[+] Master Dashboard tersimpan → '{save_path}'")


def _add_section_label(fig, gs, row, text):
    """Tambahkan label seksi di atas setiap baris subplot."""
    # Ambil posisi baris dari GridSpec
    ss = gs[row, 0].get_position(fig)
    se = gs[row, 2].get_position(fig)
    x  = (ss.x0 + se.x1) / 2
    y  = ss.y1 + 0.008
    fig.text(x, y, text, ha='center', va='bottom',
             fontsize=9.5, fontweight='bold', color='#455A64',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#ECEFF1',
                       edgecolor='#B0BEC5', linewidth=0.8))
