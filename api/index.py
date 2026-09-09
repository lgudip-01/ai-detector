from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()


class TextPayload(BaseModel):
    text: str


# Model paths relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model_pac.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

# Safe loading with fallback checks
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    model = None
    vectorizer = None
    load_error = str(e)

# Handle both /api and /api/index to prevent 404s


@app.post("/api")
@app.post("/api/index")
def analyze(payload: TextPayload):
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=500, detail=f"Model failed to load: {load_error}")

    cleaned = payload.text.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        vec = vectorizer.transform([cleaned])
        pred = int(model.predict(vec)[0])
        return {
            "prediction": pred,
            "label": "AI-Generated" if pred == 1 else "Human-Written"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")
