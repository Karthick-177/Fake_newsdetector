from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# ── Load Model & Vectorizer ───────────────────────────────────
model      = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Schema ────────────────────────────────────────────
class NewsInput(BaseModel):
    text: str

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Fake News Detector API is running!"}

@app.post("/predict")
def predict(news: NewsInput):
    vec = vectorizer.transform([news.text])
    proba = model.predict_proba(vec)[0]
    fake_score = proba[1]
    
    # Only call Fake if confidence > 60%
    if fake_score > 0.60:
        label = "Fake"
        confidence = fake_score
    else:
        label = "Real"
        confidence = proba[0]

    return {
        "prediction": label,
        "confidence": f"{confidence * 100:.2f}%"
    }
