# Predictive PM2.5 - Deep Learning Forecast

Proyek ini adalah sistem prediksi kadar PM2.5 menggunakan arsitektur *Deep Learning* seperti **LSTM (Long Short-Term Memory)** dan hibrida **CNN-LSTM**. Sistem ini dilatih menggunakan histori data deret waktu (*time-series*) untuk meramalkan tingkat polusi/kualitas udara 1 jam ke depan di beberapa lokasi observasi, yaitu **Deli**, **GKU**, dan **TULT**.

---

## 📁 Struktur Proyek

Secara garis besar, struktur repositori berfokus pada folder *script* model yang dikelompokkan berdasarkan nama observasi dan jenis arsitektur:

```text
src/model/script/
├── deli/
│   ├── cnn-lstm/       # Pipeline CNN-LSTM untuk data Deli
│   └── lstm/           # Pipeline Murni LSTM untuk data Deli
├── gku/
│   ├── cnn-lstm/       # Pipeline CNN-LSTM untuk data GKU
│   └── lstm/           # Pipeline Murni LSTM untuk data GKU
└── tult/
    ├── cnn-lstm/       # Pipeline CNN-LSTM untuk data TULT
    └── lstm/           # Pipeline Murni LSTM untuk data TULT
```

Setiap *pipeline* terdiri dari berbagai modul utama:
- `main.py`: Entri utama untuk menjalankan seluruh proses dari persiapan data, perancangan model, hingga evaluasi akhir.
- `data_processing.py`: Melakukan tahap prapemrosesan (*handling missing values* dengan interpolasi *time-series*, ekstraksi fitur periodik sin kosinus, split & scale data).
- `window_generator.py`: Implementasi utilitas generator dataset berbasis *sliding window* yang digunakan bersama *TensorFlow Dataset* guna melatih LSTM.
- `model.py`: Berisi definisi model struktur saraf tiruan (Layer konfigurasi `Bi-Directional LSTM`, `Conv1D`, `Dense`) menggunakan TensorFlow/Keras.
- `evaluation.py`: Algoritma perhitungan metrik performa (*R²*, *MAE*, *RMSE*, *MAPE*) dan ekspor visualisasi grafik antara nilai prediksi & nilai aktual.

---

## 🔧 Kebutuhan Sistem & Instalasi

Proyek ini memerlukan **Python 3.10** ke atas dan berjalan efisien dalam lingkungan isolasi virtual. Daftar seluruh ketergantungan dapat dilihat dalam berkas `requirement.txt`.

### 1. Pembuatan Lingkungan Conda (Disarankan)
Sebaiknya gunakan Conda atau virtual environment sejenis untuk kelancaran komputasi model AI:

```bash
# Pembuatan environment dengan nama pemanggil pm25research
conda create -n pm25research python=3.10

# Melakukan aktivasi environment yang baru terbuat
conda activate pm25research
```

### 2. Instalasi Ketergantungan Ekstensi
Komponen pustaka utama yang wajib diinstall:
- **Data Engineer**: `pandas`, `numpy`, `scikit-learn`, `joblib`
- **Algoritma AI / DL**: `tensorflow`
- **Visualisasi**: `matplotlib`, `seaborn`

```bash
pip install -r requirement.txt
```

---

## 🚀 Cara Penggunaan

Gunakan terminal *command-line* untuk menjalankan langkah *pipeline* prediksi PM2.5:

1. **Aktivasi Environment**: Pastikan *virtual env* (misal: `pm25research`) berhasil diaktivasi.
2. Tempatkan atau posisikan dataset latih (`.csv`) yang benar ke dalam *folder* terpusat misalnya `data/` tergantung jalur yang ada dalam rujukan *path* di Python.
3. Masuk ke direktori model / lokasi sensor yang Anda kehendaki menggunakan command `cd`. *Sebagai contoh*:
   ```bash
   # Masuk melatih arsitektur Murni LSTM untuk gedung TULT
   cd src/model/script/tult/lstm
   ```
4. Eksekusi Script Pipeline Utama:
   ```bash
   python main.py
   ```

### Tahapan Sistem Berjalan
Ketika `main.py` dipanggil, log proses akan berjalan sesuai tahapan berikut:
- **`[1/5]`** Memuat dataset dan injeksi fitur waktu otomatis.
- **`[2/5]`** Pengorganisasian tensor generator `WindowGenerator`.
- **`[3/5]`** Membangun dan merakit desain topologi matriks neuron TensorFlow.
- **`[4/5]`** Melatih skema iterasi data secara progresif (terlengkapi *Callbcaks: Early Stopping, ReduceLROnPlateau*).
- **`[5/5]`** Kalkulasi Skor R², *Error rate*, cetak plot visual (`.png`), dan pengekstrasian model (`.keras`).
