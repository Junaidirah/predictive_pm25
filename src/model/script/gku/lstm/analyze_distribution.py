import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from data_processing import load_and_interpolate, add_time_features, normalize, split_data
from config import SCALED_COLS, CYCLICAL_COLS, FEATURE_COLS

_SCALED_SET = set(SCALED_COLS)

def plot_violin_distribution(train_scaled, val_scaled, test_scaled):
    splits = {'Train': train_scaled, 'Val': val_scaled, 'Test': test_scaled}

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('Analisis Distribusi Z-Score Setelah Normalisasi', fontsize=16, weight='bold')
    gs = gridspec.GridSpec(2, 1, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    data_to_plot = [train_scaled[col].dropna().values for col in FEATURE_COLS]
    parts = ax1.violinplot(data_to_plot, positions=range(len(FEATURE_COLS)), showmedians=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('#4C72B0' if FEATURE_COLS[i] in _SCALED_SET else '#DD8452')
        pc.set_alpha(0.7)
    ax1.set_xticks(range(len(FEATURE_COLS)))
    ax1.set_xticklabels(FEATURE_COLS, rotation=25, ha='right', fontsize=9)
    ax1.axhline(y=0,  color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.axhline(y=3,  color='red',  linestyle=':',  linewidth=1,   alpha=0.7, label='±3σ batas')
    ax1.axhline(y=-3, color='red',  linestyle=':',  linewidth=1,   alpha=0.7)
    ax1.set_ylabel('Nilai Normalized')
    ax1.set_title('Distribusi Z-Score (Train Set) — Biru: Scaled | Oranye: Cyclical')
    ax1.legend(fontsize=9)
    ax1.grid(True, axis='y', alpha=0.3)

    for i, col in enumerate(FEATURE_COLS):
        v = train_scaled[col].dropna()
        ax1.text(i, ax1.get_ylim()[1] * 0.93,
                 f'med={np.median(v):.2f}\nstd={np.std(v):.2f}',
                 ha='center', va='top', fontsize=7, color='navy')

    ax2 = fig.add_subplot(gs[1])
    # Bandingkan distribusi PM2.5 per split (train/val/test)
    focus_cols = ['pm25']
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    positions, labels_x, box_data = [], [], []
    pos = 1
    for col in focus_cols:
        for (split_name, df), color in zip(splits.items(), colors):
            box_data.append(df[col].dropna().values)
            positions.append(pos)
            labels_x.append(f'{col}\n({split_name})')
            pos += 1
        pos += 1

    bp = ax2.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6,
                     medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors * len(focus_cols)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels_x, fontsize=7.5)
    ax2.axhline(y=0,  color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.axhline(y=3,  color='red',  linestyle=':',  linewidth=1,   alpha=0.7, label='±3σ batas')
    ax2.axhline(y=-3, color='red',  linestyle=':',  linewidth=1,   alpha=0.7)
    ax2.set_ylabel('Nilai Normalized')
    ax2.set_title('Perbandingan Distribusi Fitur PM2.5 per Split (Train/Val/Test)')
    ax2.legend(handles=[Patch(facecolor=c, alpha=0.7, label=s) for c, s in zip(colors, splits)], fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.savefig('distribusi_zscore_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[+] Violin plot disimpan ke 'distribusi_zscore_analysis.png'")

def print_stats(train_scaled, val_scaled, test_scaled):
    print("\n" + "="*65)
    print(f"{'Fitur':<20} {'Split':<8} {'Min':>7} {'Max':>7} {'Mean':>7} {'Std':>7} {'Outlier>3σ':>10}")
    print("="*65)
    for col in SCALED_COLS:
        for split_name, df in [('Train', train_scaled), ('Val', val_scaled), ('Test', test_scaled)]:
            v = df[col].dropna()
            outliers = ((v > 3) | (v < -3)).sum()
            print(f"{col:<20} {split_name:<8} {v.min():>7.2f} {v.max():>7.2f} {v.mean():>7.2f} {v.std():>7.2f} {outliers:>5} ({outliers/len(v)*100:.1f}%)")
    print("="*65)

if __name__ == '__main__':
    path = "D:/development/predictive_pm25/data/training/data_training_GKU.csv"
    df            = load_and_interpolate(path)
    df_f          = add_time_features(df)
    df_ready, _   = normalize(df_f)
    train_scaled, val_scaled, test_scaled = split_data(df_ready)

    print_stats(train_scaled, val_scaled, test_scaled)
    plot_violin_distribution(train_scaled, val_scaled, test_scaled)
