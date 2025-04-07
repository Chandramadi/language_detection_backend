from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

# ─── Configuration ────────────────────────────────────────────────────────────

# Labels in the exact order you trained your adapter
LABELS = ["bn", "gu", "hi", "english"]

# FastAPI app instance
app = FastAPI(title="Indian Language Detection API")

# Pydantic models for request bodies
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

# Load tokenizer (from local folder)
tokenizer = AutoTokenizer.from_pretrained("model/tokenizer")

# Load base model and wrap with PEFT adapter
base_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(LABELS)
)
model = PeftModel.from_pretrained(base_model, "model/adapter_model")
model.eval()

# ─── Single‑sentence prediction ───────────────────────────────────────────────
@app.post("/predict", summary="Predict language of a single sentence")
async def predict(input: TextInput):
    try:
        # Tokenize
        inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
        # Inference
        with torch.no_grad():
            logits = model(**inputs).logits
        # Decode
        pred_id = int(torch.argmax(logits, dim=-1))
        confidence = float(torch.softmax(logits, dim=-1)[0, pred_id])
        return {
            "text": input.text,
            "language": LABELS[pred_id],
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Batch‑sentence prediction ────────────────────────────────────────────────
@app.post("/predict_batch", summary="Predict languages for a list of sentences")
async def predict_batch(batch: BatchInput):
    try:
        # Tokenize batch
        inputs = tokenizer(batch.texts, return_tensors="pt", truncation=True, padding=True)
        # Inference
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).tolist()
        confidences = probs.max(dim=-1).values.tolist()

        # Build response
        results = []
        for text, p, c in zip(batch.texts, preds, confidences):
            results.append({
                "text": text,
                "language": LABELS[p],
                "confidence": round(c, 4)
            })
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def read_root():
    return {"status": "OK", "message": "Language Detection API is running"}
