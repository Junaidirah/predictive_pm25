from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import sys

# --- 1. SETUP LIFESPAN (Load Model Sekali Saja) ---
# Variable global untuk menyimpan model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model saat server nyala
    print("🔄 Loading Model... Jangan request dulu.")
    try:
        # CONTOH: Ganti baris ini dengan loading model aslimu
        # model = torch.load("model_terbaik_kamu.pth")
        # ml_models["model"] = model.eval().cuda() # Pindahkan ke GPU jika ada
        
        # Simulasi dummy model biar script ini bisa kamu tes langsung
        ml_models["model"] = "DUMMY_MODEL_LOADED" 
        print("✅ Model berhasil di-load ke Memory/GPU!")
    except Exception as e:
        print(f"❌ Gagal load model: {e}")
        sys.exit(1)
        
    yield # Server berjalan di sini
    
    # Clean up saat server mati (opsional)
    ml_models.clear()
    print("🛑 Cleaning up resources...")

# Inisialisasi App
app = FastAPI(lifespan=lifespan)

# --- 2. DEFINISI SKEMA DATA (Pydantic) ---
# Ini mencegah data sampah masuk ke modelmu
class InputData(BaseModel):
    feature_1: float
    feature_2: float
    text_input: str | None = None

class PredictionResult(BaseModel):
    prediction: str
    confidence: float

# --- 3. ENDPOINT PREDIKSI ---
@app.post("/predict", response_model=PredictionResult)
async def predict(data: InputData):
    if not ml_models["model"]:
        raise HTTPException(status_code=500, detail="Model belum siap.")

    # --- LOGIKA INFERENCE DI SINI ---
    try:
        # Contoh akses data: data.feature_1, data.text_input
        # result = ml_models["model"](input_tensor)
        
        # Simulasi hasil prediksi
        print(f"Menerima data: {data}")
        fake_prediction = "Kelas_A"
        fake_confidence = 0.98
        
        return {"prediction": fake_prediction, "confidence": fake_confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat prediksi: {str(e)}")

# --- 4. HEALTH CHECK ---
@app.get("/")
def read_root():
    return {"status": "online", "backend": "FastAPI Deep Learning"}