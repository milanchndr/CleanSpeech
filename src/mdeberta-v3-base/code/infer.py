# src/mdeberta-v3-base/code/infer.py
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import LABELS, default_cfg

# --------------------------------------------------
# Load model/tokenizer
# --------------------------------------------------
def load_pipeline(model_dir: str | None = None):
    cfg = default_cfg()
    model_dir = model_dir or cfg.paths.model_dir
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tok, device

# --------------------------------------------------
# Predict probabilities
# --------------------------------------------------
def predict_proba(texts, model=None, tokenizer=None, model_dir: str | None = None,
                  max_len: int | None = None, device: str | None = None):
    if isinstance(texts, str):
        texts = [texts]
    if model is None or tokenizer is None:
        model, tokenizer, device = load_pipeline(model_dir)
    if max_len is None:
        max_len = default_cfg().train.max_len
    if device is None:
        device = next(model.parameters()).device

    probs_list = []
    for txt in texts:
        enc = tokenizer(txt, truncation=True, padding="max_length",
                        max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(**enc).logits).cpu().numpy()[0]
        probs_list.append(p)
    return np.vstack(probs_list)

# --------------------------------------------------
# Predict discrete labels (with optional thresholds)
# --------------------------------------------------
def _load_thresholds(path: str | None):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {lab: 0.5 for lab in LABELS}

def predict(texts, model=None, tokenizer=None, model_dir: str | None = None,
            thresholds_path: str | None = None, max_len: int | None = None):
    if isinstance(texts, str):
        texts = [texts]
    if model is None or tokenizer is None:
        model, tokenizer, device = load_pipeline(model_dir)
    else:
        device = next(model.parameters()).device

    probs = predict_proba(texts, model, tokenizer, model_dir=model_dir, max_len=max_len, device=device)
    thresholds = _load_thresholds(thresholds_path)
    thr_vec = np.array([thresholds[l] for l in LABELS], dtype=np.float32)
    labels = (probs >= thr_vec).astype(int)
    results = [dict(zip(LABELS, row.tolist())) for row in labels]
    return results, probs
