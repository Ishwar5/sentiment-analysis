# Emotion Detection from Text

This repository implements a simple emotion detection API for text inputs. It includes:

- A small demo dataset (for quick experiments)
- A training script using scikit-learn (TF-IDF + LogisticRegression)
- A FastAPI server to serve predictions
- Utilities using NLTK for text preprocessing

This project is intended as a starter/demo. It also includes notes for using Hugging Face transformer models as an optional improvement.

## Contents

- `src/train.py` — training script that produces serialized model and vectorizer in `models/`
- `src/app.py` — FastAPI application exposing `/predict` endpoint
- `src/utils.py` — preprocessing helpers
- `data/emotion_small.csv` — small demo dataset
- `requirements.txt` — Python dependencies

## Quick setup (Windows PowerShell)

1. Create a virtual environment and activate it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Train the demo model (saves to `models/`)

```powershell
python -m src.train train
```

4. Run the API

```powershell
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

5. Try a request (PowerShell example using curl)

```powershell
curl -Method POST -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{"text":"I am so happy and excited today!"}'
```

## API

POST /predict

Request JSON: { "text": "your text here" }

Response JSON example:

{
  "label": "happy",
  "scores": {"happy": 0.84, "sad": 0.03, ...},
  "input": "I am so happy and excited today!"
}

## Notes and next steps

- The included dataset is intentionally small to keep the demo lightweight. For production use, replace it with a larger labeled dataset (for example the `emotion` dataset on Hugging Face or a domain-specific dataset).
- Optional: add a Hugging Face transformer-based model for stronger performance. The `transformers` and `torch` packages are listed in `requirements.txt` as optional dependencies.

## License

MIT
