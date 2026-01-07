from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
)

# ----- Paths -----
REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = REPO_ROOT / "models_artifacts" / "agnews_distilbert" / "best"
NER_DIR = REPO_ROOT / "models_artifacts" / "conll2003_distilbert_ner" / "best"

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
NER_LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
ID2LABEL = {i: lab for i, lab in enumerate(NER_LABELS)}
LABEL2ID = {lab: i for i, lab in enumerate(NER_LABELS)}


# ----- Device (classifier uses torch directly) -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load classifier once -----
tokenizer_cls = AutoTokenizer.from_pretrained(str(CLASSIFIER_DIR))
model_cls = AutoModelForSequenceClassification.from_pretrained(str(CLASSIFIER_DIR))
model_cls.eval()
model_cls.to(device)

# ----- Load NER once (pipeline) -----
tokenizer_ner = AutoTokenizer.from_pretrained(str(NER_DIR))
model_ner = AutoModelForTokenClassification.from_pretrained(
    str(NER_DIR),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
model_ner.eval()


ner_pipe = pipeline(
    "token-classification",
    model=model_ner,
    tokenizer=tokenizer_ner,
    aggregation_strategy="simple",
    device=-1,
)


# ----- API -----
app = FastAPI(title="Document Intelligence API", version="0.1.0")


class ClassifyRequest(BaseModel):
    text: str


class NerRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/classify")
def classify(req: ClassifyRequest):
    inputs = tokenizer_cls(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        logits = model_cls(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        conf = float(probs[pred_id].item())

    return {"label": LABELS[pred_id], "confidence": conf}


@app.post("/ner")
def ner(req: NerRequest):
    entities = ner_pipe(req.text)

    cleaned = []
    for e in entities:
        # entity_group will become something like "B-ORG" or "I-ORG" (or "O" depending on version)
        raw = e.get("entity_group") or e.get("entity")

        if raw == "O":
            continue

        # normalize: B-ORG -> ORG
        label = raw.split("-")[-1]

        cleaned.append(
            {
                "text": e.get("word"),
                "label": label,
                "score": float(e.get("score")),
                "start": int(e.get("start")),
                "end": int(e.get("end")),
            }
        )

    return {"entities": cleaned}



