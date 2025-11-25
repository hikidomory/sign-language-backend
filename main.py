from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os
import tensorflow as tf
from typing import List, Union

# --------- 설정 ---------
MODEL_KEYS = ["hangul", "digit", "word"] # 'word' 포함

app = FastAPI()

# --------- CORS 설정 ---------
origins = ["*"] # 개발 편의를 위해 전체 허용 (배포 시 수정 권장)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Server is Running!"}

# --------- 모델 로드 ---------
interpreters = {}
input_details = {}
output_details = {}
scalers = {}
encoders = {}

@app.on_event("startup")
def load_models():
    print("🚀 모델 로딩 시작...")
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
                print(f"   ✅ {key} 모델 로드 성공")
            else:
                print(f"   ⚠️ 파일 없음: {path}")
        except Exception as e:
            print(f"   ❌ {key} 모델 로드 실패: {e}")

# 요청 데이터 형식 정의
# features는 1차원 리스트(기존)일 수도 있고, 2차원 리스트(새 모델, 90x258)일 수도 있음
class PredictIn(BaseModel):
    model_key: str
    features: Union[List[float], List[List[float]]] 

@app.post("/predict")
def predict(inp: PredictIn):
    try:
        if inp.model_key not in interpreters:
            return {"label": "Error", "detail": "모델 없음"}

        interpreter = interpreters[inp.model_key]
        # scaler = scalers[inp.model_key] # 새 모델은 스케일러 패스 (필요시 활성화)
        encoder = encoders[inp.model_key]
        in_det = input_details[inp.model_key]
        out_det = output_details[inp.model_key]

        # 데이터 변환 로직
        if inp.model_key == "word":
            # (90, 258) -> (1, 90, 258) 형태로 변환
            data = np.array(inp.features, dtype=np.float32)
            data = np.expand_dims(data, axis=0) 
        else:
            # 기존 모델: (22,) -> (1, 22)
            scaler = scalers[inp.model_key]
            data = np.array(inp.features, dtype=np.float32).reshape(1, -1)
            data = scaler.transform(data).astype(np.float32)

        # 추론 실행
        interpreter.set_tensor(in_det[0]['index'], data)
        interpreter.invoke()
        
        # 결과 해석
        y = interpreter.get_tensor(out_det[0]['index'])[0]
        idx = int(np.argmax(y))
        label = encoder.inverse_transform([idx])[0]
        confidence = float(y[idx])

        return {"label": label, "confidence": confidence}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"label": "Error", "detail": str(e)}