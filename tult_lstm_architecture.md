# Arsitektur Pipeline Predictive PM2.5 (TULT - Pure LSTM)

Dokumen ini berisi pemodelan visual untuk alur kerja (pipeline) penuh dari awal persiapan data hingga akhir evaluasi pada algoritma **Murni LSTM** yang dialokasikan di Gedung TULT.

> [!NOTE]
> Diagram ini menggunakan ekstensi sintaks **Mermaid**.

```mermaid
flowchart TD
    %% Styling
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b;
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c;
    classDef window fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100;
    classDef model fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#1b5e20;
    classDef train fill:#ffebee,stroke:#b71c1c,stroke-width:2px,color:#b71c1c;
    classDef eval fill:#fffde7,stroke:#f57f17,stroke-width:2px,color:#f57f17;
    classDef output fill:#eceff1,stroke:#263238,stroke-width:2px,color:#263238;

    %% Data Processing Phase
    subgraph DataProcessing ["1. Data Processing (data_processing.py)"]
        D1[("Raw Data\n(data_training_tult.csv)")]:::data --> P1["load_and_interpolate()\n- Parsing Waktu\n- Interpolasi Linier"]:::process
        P1 --> P2["add_time_features()\n- cyclical sine/cosine\n(Day/Year)"]:::process
        P2 --> P3["split_and_scale()\n- Split: 70% Train, 20% Val, 10% Test\n- StandardScaler (Normalisasi)"]:::process
    end

    %% Windowing Phase
    subgraph Windowing ["2. Time Series Windowing (window_generator.py)"]
        P3 --> W1["Window Generator\n- Input: 24 Jam\n- Label: 1 Jam ke depan\n- Shift: 1"]:::window
    end

    %% Model Construction Phase
    subgraph Architecture ["3. Model Architecture (model.py)"]
        W1 --> M1["Input Layer\nShape: (24, 8 Fitur)"]:::model
        M1 --> M2["Bidirectional LSTM\n- Units: 32\n- L2 Regularization (0.002)"]:::model
        M2 --> M3["Dropout Layer (50%)"]:::model
        M3 --> M4["Dense Layer\n- Units: 16, Activation: ReLU\n- L2 Regularization (0.002)"]:::model
        M4 --> M5["Dropout Layer (30%)"]:::model
        M5 --> M6["Output Layer\n- Dense(1)\nPrediksi (t+1)"]:::model
    end

    %% Training Pipeline Phase
    subgraph Training ["4. Model Training (main.py)"]
        M6 --> T1["Compile\n- Optimizer: Adam (lr=1e-4)\n- Loss: Huber\n- Metrics: MAE"]:::train
        T1 --> T2["Fit Model\n- Epochs: 100\n- Batch Size: default (32)"]:::train
        T2 --> T3["Callbacks\n- EarlyStopping (Patience 15)\n- ReduceLROnPlateau\n- R2Callback"]:::train
    end

    %% Evaluation & Export Phase
    subgraph Evaluation ["5. Evaluation & Export (evaluation.py)"]
        T3 --> E1["Predict on Test Set"]:::eval
        E1 --> E2["Inverse Transform Scaler\n(Z-Score -> uf/m3 Asli)"]:::eval
        E2 --> E3["Kalkulasi Metrik Asli\n(MAE, RMSE, MAPE, R2)"]:::eval
        
        E3 --> O1[("metrik_performa_tult.csv\n(Direkap Otomatis)")]:::output
        E3 --> O2[("dashboard_evaluasi_tult_lstm.png\n(Visualisasi 6 Panel Matplotlib)")]:::output
        E3 --> O3[("pm25_model_1hour.keras\n(Saved Model Weights)")]:::output
    end
```

### Keterangan Lapisan (Layers)
- **Input Features (8 Variabel):** `pm25`, `temperature`, `humidity`, `pm25_diff`, `Day sin`, `Day cos`, `Year sin`, `Year cos`.
- **Regulasi Anti-Overfitting:** Dibangun langsung terintegrasi menggunakan L2 constraint pada bobot koneksi `Bidirectional LSTM` (32 unit) dan `Dense` (16 unit) yang diselingi rasio pemutusan sel neuron *Dropout* sebesar 50% dan 30%.
- **Mekanisme Otomatis:** Sistem sudah meredam *Learning Rate* perlahan secara instan jika pergerakan validasi stagnan (*plateau*) di titik peluruhan ke-5 komputasi, dan memberhentikan komputasi (*early stopping*) sepenuhnya jika tak ada pembaruan dalam 15 rentetan putaran (epoch).
- **Evaluasi Hilir:** Hasil prediksi jaringan mesin (format normalisasi Z-Score) diterjemahkan ulang ke konversi format `ug/m3` menggunakan koefisien rata-rata `StandardScaler` sehingga perhitungannya langsung valid di kehidupan nyata!
