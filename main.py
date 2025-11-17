from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os
import gc
import tensorflow as tf 

# --------- 설정 ---------
MODEL_KEYS = ["hangul", "digit"]

app = FastAPI()

# --------- CORS 설정 ---------
origins = [
    "http://localhost:5173",
]
origin_regex = r"https://.*\.vercel\.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ [추가됨] 루트 경로 접속 시 404 방지용 (Render 헬스 체크 통과용)
@app.get("/")
def home():
    return {"message": "Smart Sign Language Server (TFLite) is Running!"}

@app.head("/")
def keep_alive():
    return {"message": "I am alive"}

# --------- 모델 로드 (TFLite 버전) ---------
interpreters = {}
input_details = {}
output_details = {}
scalers = {}
encoders = {}

@app.on_event("startup")
def load_models():
    print("🚀 [STARTUP] TFLite 모델 로딩 시작 (초경량 모드)...")
    
    for key in MODEL_KEYS:
        try:
            path = f"model_{key}.tflite"
            
            if os.path.exists(path):
                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                
                interpreters[key] = interpreter
                input_details[key] = interpreter.get_input_details()
                output_details[key] = interpreter.get_output_details()
                
                scalers[key] = joblib.load(f"scaler_{key}.pkl")
                encoders[key] = joblib.load(f"label_encoder_{key}.pkl")
                
                print(f"   ✅ Loaded {key} (TFLite) successfully.")
                gc.collect()
            else:
                print(f"   ⚠️ TFLite file not found: {path}")
        except Exception as e:
            print(f"   ❌ Failed to load {key}: {e}")

class PredictIn(BaseModel):
    model_key: str
    features: list[float]

@app.post("/predict")
def predict(inp: PredictIn):
    try:
        if inp.model_key not in interpreters:
            return {"label": "준비중", "confidence": 0.0}

        interpreter = interpreters[inp.model_key]
        scaler = scalers[inp.model_key]
        encoder = encoders[inp.model_key]
        in_det = input_details[inp.model_key]
        out_det = output_details[inp.model_key]

        features_arr = np.array(inp.features, dtype=np.float32).reshape(1, -1)
        x = scaler.transform(features_arr)
        x = x.astype(np.float32)

        interpreter.set_tensor(in_det[0]['index'], x)
        interpreter.invoke()
        
        y = interpreter.get_tensor(out_det[0]['index'])[0]
        
        idx = int(np.argmax(y))
        label = encoder.inverse_transform([idx])[0]
        confidence = float(y[idx])

        return {"label": label, "confidence": confidence}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"label": "Error", "confidence": 0.0, "detail": str(e)}