import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Define the request payload structure


class TextPayload(BaseModel):
    text: str


# Load models using safe relative paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model_pac.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


@app.post("/api/analyze")
def analyze_text(payload: TextPayload):
    user_text = payload.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        vectorized_text = vectorizer.transform([user_text])
        prediction = int(model.predict(vectorized_text)[0])

        label = "AI-Generated" if prediction == 1 else "Genuine / Human"

        return {
            "label": label,
            "prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
