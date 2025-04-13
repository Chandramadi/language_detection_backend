
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

LABELS = ["bn", "gu", "hi", "english"]

tokenizer = AutoTokenizer.from_pretrained("model/tokenizer")
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(LABELS))
model = PeftModel.from_pretrained(base_model, "model/adapter_model")
model.eval()

def predict_single(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(torch.argmax(logits, dim=-1))
    confidence = float(torch.softmax(logits, dim=-1)[0, pred_id])
    return {"text": text, "language": LABELS[pred_id], "confidence": round(confidence, 4)}

def predict_batch(texts: list):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1).tolist()
    confidences = probs.max(dim=-1).values.tolist()

    results = []
    for text, p, c in zip(texts, preds, confidences):
        results.append({
            "text": text,
            "language": LABELS[p],
            "confidence": round(c, 4)
        })
    return results
