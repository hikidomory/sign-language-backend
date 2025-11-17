# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os
import gc
import tensorflow as tf # TFLite Interpreter 사용

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

# --------- 모델 로드 (TFLite 버전) ---------
interpreters = {} # 모델 대신 인터프리터 저장
input_details = {}
output_details = {}
scalers = {}
encoders = {}

@app.on_event("startup")
def load_models():
    print("🚀 [STARTUP] TFLite 모델 로딩 시작 (초경량 모드)...")
    
    for key in MODEL_KEYS:
        try:
            # .tflite 파일 경로
            path = f"model_{key}.tflite"
            
            if os.path.exists(path):
                # 1. 인터프리터 로드 (Keras model.load_model보다 훨씬 가벼움)
                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                
                interpreters[key] = interpreter
                
                # 입출력 정보 저장 (나중에 predict할 때 필요)
                input_details[key] = interpreter.get_input_details()
                output_details[key] = interpreter.get_output_details()
                
                # 스케일러/인코더 로드
                scalers[key] = joblib.load(f"scaler_{key}.pkl")
                encoders[key] = joblib.load(f"label_encoder_{key}.pkl")
                
                print(f"   ✅ Loaded {key} (TFLite) successfully.")
                gc.collect()
            else:
                print(f"   ⚠️ TFLite file not found: {path} (변환했나요?)")
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

        # 1. 데이터 전처리
        features_arr = np.array(inp.features, dtype=np.float32).reshape(1, -1)
        x = scaler.transform(features_arr)
        x = x.astype(np.float32) # TFLite는 타입에 민감함

        # 2. 추론 실행 (Invoke)
        interpreter.set_tensor(in_det[0]['index'], x)
        interpreter.invoke() # 실행!
        
        # 3. 결과 가져오기
        y = interpreter.get_tensor(out_det[0]['index'])[0]
        
        idx = int(np.argmax(y))
        label = encoder.inverse_transform([idx])[0]
        confidence = float(y[idx])

        return {"label": label, "confidence": confidence}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"label": "Error", "confidence": 0.0, "detail": str(e)}