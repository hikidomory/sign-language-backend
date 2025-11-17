from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import gc

# --------- 설정 ---------
MODEL_KEYS = ["hangul", "digit"]

app = FastAPI()

# --------- CORS 설정 (필살기 버전) ---------
# '*' 대신에 실제 주소를 명시하는 것이 가장 안전합니다.
origins = [
    "http://localhost:5173",
    "https://sign-language-project.vercel.app",      # 기본 주소
    "https://sign-language-project-teal.vercel.app", # teal 버전 주소
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    # '*' 대신 구체적인 주소 리스트 사용
    allow_credentials=True,   # 이제 True여도 안전함 (주소를 명시했으니까!)
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- 모델 로드 (메모리 절약 모드) ---------
models = {}
scalers = {}
encoders = {}

@app.on_event("startup")
def load_models():
    print("지금은 메모리 테스트 중이라 모델을 안 불러옵니다!")
    # 모델 로딩 실패해도 서버가 꺼지지 않게 방어
    for key in MODEL_KEYS:
        try:
            # h5 파일 경로 확인
            path = f"model_{key}.h5"
            if os.path.exists(path):
                models[key] = load_model(path)
                scalers[key] = joblib.load(f"scaler_{key}.pkl")
                encoders[key] = joblib.load(f"label_encoder_{key}.pkl")
                print(f"[INFO] Loaded {key} successfully.")
            else:
                print(f"[WARNING] Model file not found: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to load {key}: {e}")


class PredictIn(BaseModel):
    model_key: str
    features: list[float]

@app.get("/")
def home():
    return {"message": "Smart Sign Language Server is Running!"}

@app.post("/predict")
def predict(inp: PredictIn):
    try:
        # 1. 메모리 청소부터 하고 시작
        gc.collect()
        
        if inp.model_key not in models:
            return {"label": "준비중", "confidence": 0.0}

        model = models[inp.model_key]
        scaler = scalers[inp.model_key]
        encoder = encoders[inp.model_key]

        # 2. 데이터 변환 (최대한 가볍게)
        # float32로 변환하여 메모리 절약
        features_arr = np.asarray(inp.features, dtype=np.float32)
        
        # 2차원 배열 변환 (1, N)
        x = features_arr.reshape(1, -1) 
        
        x = scaler.transform(x)

        # 3. 예측 실행 (배치 사이즈 1로 제한)
        # verbose=0으로 설정하여 로그 출력 메모리 아낌
        y = model.predict(x, batch_size=1, verbose=0)[0]
        
        idx = int(np.argmax(y))
        label = encoder.inverse_transform([idx])[0]
        confidence = float(y[idx])

        # 4. 사용한 메모리 즉시 반납
        del x
        del features_arr
        gc.collect()

        return {"label": label, "confidence": confidence}

    except Exception as e:
        print(f"❌ 예측 중 에러 발생: {e}")
        return {"label": "Error", "confidence": 0.0, "detail": str(e)}