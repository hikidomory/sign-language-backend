from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

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

# @app.on_event("startup")
# def load_models():
#     print("지금은 메모리 테스트 중이라 모델을 안 불러옵니다!")
#     # 모델 로딩 실패해도 서버가 꺼지지 않게 방어
#     for key in MODEL_KEYS:
#         try:
#             # h5 파일 경로 확인
#             path = f"model_{key}.h5"
#             if os.path.exists(path):
#                 models[key] = load_model(path)
#                 scalers[key] = joblib.load(f"scaler_{key}.pkl")
#                 encoders[key] = joblib.load(f"label_encoder_{key}.pkl")
#                 print(f"[INFO] Loaded {key} successfully.")
#             else:
#                 print(f"[WARNING] Model file not found: {path}")
#         except Exception as e:
#             print(f"[ERROR] Failed to load {key}: {e}")
@app.on_event("startup")
def load_models():
    print("🚧 [TEST MODE] 지금은 메모리 테스트 중이라 모델을 불러오지 않습니다! 🚧")
    
    # 아래 코드들을 따옴표 3개(""")로 감싸서 실행 안 되게 만듭니다.
    """
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
    """
    pass # 아무것도 안 하고 넘어가라는 뜻

class PredictIn(BaseModel):
    model_key: str
    features: list[float]

@app.get("/")
def home():
    return {"message": "Smart Sign Language Server is Running!"}

@app.post("/predict")
def predict(inp: PredictIn):
    try:
        if inp.model_key not in models:
            # 모델이 없으면 에러 대신 가짜 응답을 줘서 연결 테스트 (디버깅용)
            return {"label": "준비중", "confidence": 0.0}

        model = models[inp.model_key]
        scaler = scalers[inp.model_key]
        encoder = encoders[inp.model_key]

        x = np.asarray(inp.features, dtype=np.float32)[None, :]
        x = scaler.transform(x)
        y = model.predict(x, verbose=0)[0]
        
        idx = int(np.argmax(y))
        label = encoder.inverse_transform([idx])[0]
        confidence = float(y[idx])

        return {"label": label, "confidence": confidence}

    except Exception as e:
        # 에러가 나도 500 대신 200 OK로 에러 메시지를 보냄 (CORS 회피)
        return {"label": "Error", "confidence": 0.0, "detail": str(e)}