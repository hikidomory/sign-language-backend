from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --------- 설정 ---------
MODEL_KEYS = ["hangul", "digit"]

app = FastAPI()

# CORS 허용 (localhost 테스트용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 ["http://localhost:5500", "http://127.0.0.1:5500"] 로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- 모델/스케일러/인코더 로드 ---------
models = {}
scalers = {}
encoders = {}

for key in MODEL_KEYS:
    try:
        models[key] = load_model(f"model_{key}.h5")
        scalers[key] = joblib.load(f"scaler_{key}.pkl")
        encoders[key] = joblib.load(f"label_encoder_{key}.pkl")
        print(f"[INFO] Loaded model set: {key}")
    except Exception as e:
        print(f"[ERROR] Failed to load {key}: {e}")

# --------- 요청 형식 ---------
class PredictIn(BaseModel):
    model_key: str
    features: list[float]


# --------- 예측 엔드포인트 ---------
@app.post("/predict")
def predict(inp: PredictIn):
    try:
        if inp.model_key not in models:
            raise HTTPException(status_code=400, detail=f"Invalid model key: {inp.model_key}")

        model = models[inp.model_key]
        scaler = scalers[inp.model_key]
        encoder = encoders[inp.model_key]

        # 입력 벡터 확인
        x = np.asarray(inp.features, dtype=np.float32)[None, :]
        print(f"[DEBUG] Received {len(inp.features)} features for {inp.model_key}", flush=True)

        # 스케일러 적용
        try:
            x = scaler.transform(x)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scaler error: {e}")

        # 예측 수행
        try:
            y = model.predict(x, verbose=0)[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

        # 결과 후처리
        if y.ndim > 1:
            y = y.ravel()

        idx = int(np.argmax(y))
        label = encoder.inverse_transform([idx])[0]
        confidence = float(y[idx])

        print(f"[INFO] Predict → {inp.model_key}: {label} ({confidence:.3f})", flush=True)
        return {"label": label, "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- 실행 ---------
# ⚠️ 아래 부분은 절대 다시 uvicorn.run()을 쓰지 않습니다!
# uvicorn은 외부에서 실행합니다:
#   uvicorn main:app --host 127.0.0.1 --port 8000
