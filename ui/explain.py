# ui/explain.py
"""
Explainability view for CleanSpeech.
We will add features incrementally:
1) Load baseline pipeline + labels
2) Compute per-label probabilities
3) Explain one label via linear token contributions (w × x)
4) What-if: remove/replace a token
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
import joblib
import json
import streamlit as st
import re

# --- Step 2: load artifacts (pipeline + meta) ---
@st.cache_resource(show_spinner=False)
def load_artifacts():
    # Reuse your existing path helpers so Explain uses the same folder
    try:
        from .paths import get_root, get_models_dir
    except ImportError:
        from paths import get_root, get_models_dir

    root: Path = get_root(__file__)
    models_dir: Path = get_models_dir(root)

    pipe_path = models_dir / "baseline_pipeline.joblib"
    meta_path = models_dir / "baseline_meta.json"
    if not pipe_path.exists():
        raise FileNotFoundError(f"Missing model: {pipe_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    pipe = joblib.load(pipe_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Extract vectorizer + per-label estimators
    assert "tfidf" in pipe.named_steps, "Pipeline missing 'tfidf' step."
    assert "clf"   in pipe.named_steps, "Pipeline missing 'clf' (OVR) step."
    tfidf = pipe.named_steps["tfidf"]
    ovr   = pipe.named_steps["clf"]
    assert hasattr(ovr, "estimators_"), "Classifier not fitted (no estimators_)."

    labels = meta.get("label_cols")
    if not isinstance(labels, list) or len(labels) != len(ovr.estimators_):
        labels = labels or [f"label_{i}" for i in range(len(ovr.estimators_))]

    models = {lbl: est for lbl, est in zip(labels, ovr.estimators_)}
    threshold = float(meta.get("threshold", 0.5))

    return {"vectorizer": tfidf, "models": models, "labels": labels, "threshold": threshold}


# --- Step 4: explainer utilities (linear w × x on TF-IDF) ---
def _logit_from_estimator(X, est) -> float:
    if hasattr(est, "decision_function"):
        z = est.decision_function(X)
        return float(np.asarray(z).ravel()[0])
    if hasattr(est, "predict_proba"):
        p = float(est.predict_proba(X)[:, 1][0])
        p = float(np.clip(p, 1e-9, 1 - 1e-9))
        return float(np.log(p / (1 - p)))
    raise ValueError("Estimator lacks decision_function/predict_proba")

def _weights_intercept(est):
    if hasattr(est, "coef_"):
        w = est.coef_.ravel()
        b = float(getattr(est, "intercept_", np.array([0.0]))[0])
        return w, b
    for attr in ("base_estimator", "estimator", "classifier"):
        inner = getattr(est, attr, None)
        if inner is not None and hasattr(inner, "coef_"):
            w = inner.coef_.ravel()
            b = float(getattr(inner, "intercept_", np.array([0.0]))[0])
            return w, b
    raise ValueError("Cannot access linear weights (coef_). Is the base model linear?")

# --- Step 4: explainer utilities (linear w × x on TF-IDF) ---
def linear_token_contribs(text: str, label: str, *, vectorizer, models):
    X = vectorizer.transform([text])
    est = models[label]
    z = _logit_from_estimator(X, est)
    p = 1 / (1 + np.exp(-z))
    w, b = _weights_intercept(est)

    Xc = X.tocsr()
    idxs, vals = Xc.indices, Xc.data
    contribs = w[idxs] * vals

    inv_vocab = {j: t for t, j in vectorizer.vocabulary_.items()}
    tokens = [inv_vocab[j] for j in idxs]

    agg = {}
    for t, c in zip(tokens, contribs):
        agg[t] = agg.get(t, 0.0) + float(c)

    return {"prob": float(p), "logit": float(z), "bias": float(b), "token_contributions": agg}

def highlight_text(raw_text: str, token_scores: Dict[str, float], top_k: int = 12) -> str:
    if not token_scores:
        return raw_text

    # pick top tokens by |contribution|
    top = sorted(token_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]

    # sort longer phrases first to avoid partial overlaps
    top = sorted(top, key=lambda kv: len(kv[0]), reverse=True)

    html = raw_text
    for tok, score in top:
        if not tok.strip():
            continue
        alpha = min(0.15 + 0.85 * (abs(score) / (max(abs(v) for v in token_scores.values()) + 1e-9)), 1.0)
        color = "rgba(220, 38, 38, {:.3f})".format(alpha) if score >= 0 else "rgba(37, 99, 235, {:.3f})".format(alpha)
        style = f"background-color:{color}; padding:1px 3px; border-radius:4px;"

        # \b...\b so we match whole tokens/phrases; escape special chars
        pattern = r"\b" + re.escape(tok) + r"\b"
        repl = lambda m: f"<span style=\"{style}\">{m.group(0)}</span>"
        html = re.sub(pattern, repl, html, flags=re.IGNORECASE)

    return html




def render():
    st.title("Explain")
    st.caption("Enter text → we tag the most likely label and show token contributions for that label.")

    arts = load_artifacts()
    vectorizer = arts["vectorizer"]
    models = arts["models"]
    labels = arts["labels"]
    default_threshold = float(arts.get("threshold", 0.5))

    text = st.text_area("Text", height=160, placeholder="Type or paste a comment…")


    if not st.button("Explain"):
        return

    if not text.strip():
        st.warning("Please enter some text first.")
        return


    # 1) Compute per-label probabilities
    X = vectorizer.transform([text])
    probs = {}
    for lbl in labels:
        est = models[lbl]
        if hasattr(est, "predict_proba"):
            p = float(est.predict_proba(X)[:, 1][0])
        elif hasattr(est, "decision_function"):
            z = float(est.decision_function(X)[0])
            p = 1 / (1 + np.exp(-z))
        else:
            p = float(est.predict(X)[0])
        probs[lbl] = p

    # 2) Pick the top label
    top_label = max(probs, key=probs.get)
    top_prob = probs[top_label]

    # --- Result message (green if clean, red if flagged) ---
    if top_prob >= default_threshold:
        st.markdown(
            f"<div style='background-color:rgb(145 44 44);padding:8px;border-radius:6px;'>"
            f"<b>Flagged as {top_label}</b> (p = {top_prob:.3f})"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background-color:rgb(37 113 63);padding:8px;border-radius:6px;'>"
            f"<b>No toxic content detected</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # --- Explainability: linear token contributions (w × x) ---
    st.subheader("Token contributions")


    # 3) Explain that single label
    exp = linear_token_contribs(text, top_label, vectorizer=vectorizer, models=models)
    token_scores = exp["token_contributions"]

    # Handle case: no active features found (short or unseen text)
    if not token_scores:
        st.info("No influential tokens found — text too short or contains words unseen in training.")
        return
    
    # 4) Highlighted text
    st.markdown(highlight_text(text, token_scores, top_k=12), unsafe_allow_html=True)

    # 5) Small table of the most influential tokens
    try:
        import pandas as pd
        df = pd.DataFrame(
            [(t, v, "raises" if v >= 0 else "lowers") for t, v in token_scores.items()],
            columns=["token", "contribution", "direction"],
        )
        df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
        st.dataframe(df.head(20), use_container_width=True)
    except Exception:
        st.write(dict(sorted(token_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]))


