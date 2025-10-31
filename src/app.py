from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
from typing import Dict
from src.utils import preprocess_text


class TextIn(BaseModel):
    text: str


app = FastAPI(title='Emotion Detection API')

MODEL_PATH = os.path.join('models', 'sk_model.joblib')
VECT_PATH = os.path.join('models', 'tfidf_vectorizer.joblib')

model = None
vectorizer = None
labels = None


@app.on_event('startup')
def load_model():
    global model, vectorizer, labels
    if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        vectorizer = joblib.load(VECT_PATH)
        model = joblib.load(MODEL_PATH)
        # try to get labels from model.classes_
        try:
            labels = list(model.classes_)
        except Exception:
            labels = None
        print('Model and vectorizer loaded.')
    else:
        print('Model files not found. Run `python -m src.train train` to create them.')


@app.get('/')
def root():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.post('/predict')
def predict(payload: TextIn) -> Dict:
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail='Model not available. Run the training script first.')
    text = payload.text or ''
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    proba = model.predict_proba(vec)[0]
    pred_idx = proba.argmax()
    pred_label = model.classes_[pred_idx] if hasattr(model, 'classes_') else str(pred_idx)
    scores = {str(l): float(p) for l, p in zip(model.classes_, proba)}
    return {'label': pred_label, 'scores': scores, 'input': text}
