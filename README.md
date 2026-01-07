# 03_document_intelligence
## Transformer-based Document Intelligence System

This project demonstrates an end-to-end NLP system combining text classification and named entity recognition (NER) using fine-tuned transformer models, exposed via FastAPI and consumed through a Streamlit UI.

## Features

- Text Classification
  - Model: DistilBERT
  - Dataset: AG News
  - Labels: World, Sports, Business, Sci/Tech

- Named Entity Recognition (NER)
  - Model: DistilBERT (token classification)
  - Dataset: CoNLL-2003
  - Entities: PER, ORG, LOC, MISC

- API
  - FastAPI backend
  - Endpoints:
    - GET /health
    - POST /classify
    - POST /ner
- UI
  - Streamlit frontend
  - Single text input → classification + extracted entities

## Project Structure
```
03_document_intelligence/
│
├── notebooks/               # Training notebooks (Colab)
│   ├── 01_text_classification_training.ipynb
│   └── 02_ner_training.ipynb
│
├── models_artifacts/         # Local model checkpoints (gitignored)
│   ├── agnews_distilbert/best/
│   └── conll2003_distilbert_ner/best/
│
├── api/
│   └── main.py               # FastAPI app
│
├── app/
│   └── streamlit_app.py      # Streamlit UI
│
├── requirements.txt
└── README.md
```

## How to Run Locally
1. Setup environment
   
```
python -m venv .venv
# activate venv
pip install -r requirements.txt
```

2. Start API
```
uvicorn api.main:app --reload
```

API will be available at:

http://127.0.0.1:8000
Swagger UI: http://127.0.0.1:8000/docs

3. Start Streamlit UI
```
python -m streamlit run app/streamlit_app.py
```

UI will open at:

http://localhost:8501

## Notes

 - Models are trained in Google Colab and loaded locally for inference.
 - Inference runs on CPU (GPU optional).
 - Model artifacts are intentionally excluded from Git history.
