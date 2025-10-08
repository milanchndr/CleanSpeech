# ui/components.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st


def render_header(title: str, caption: str) -> None:
    """Top-of-page title + caption (no page_config here; keep that in app.py)."""
    st.title(title)
    if caption:
        st.caption(caption)


def render_model_picker(model_paths: List[Path]) -> Path:
    """Select a model file from a list of Paths; returns the chosen Path."""
    if not model_paths:
        st.error("No models found.")
        st.stop()

    options = [p.name for p in model_paths]
    choice = st.selectbox("Select model (.joblib)", options=options, index=0)
    return next(p for p in model_paths if p.name == choice)


def render_text_input(
    label: str = "Enter your text:",
    placeholder: str = "Type a comment here...",
    height: int = 150,
) -> Tuple[str, bool]:
    """
    Draw a text area and a 'Predict' button.
    Returns (text, predict_clicked).
    """
    text = st.text_area(label, height=height, placeholder=placeholder)
    clicked = st.button("Predict")
    return (text or "").strip(), clicked


def flash_top_prediction(
    probs: np.ndarray, labels: List[str], threshold: float
) -> Tuple[int, np.ndarray]:
    """
    Show a single prominent flash:
      - st.error('Flagged as <label> (p=...)') if top prob >= threshold
      - st.success('No toxicity detected (p=...)') otherwise
    Returns (top_idx, order) where order are indices sorted by descending prob.
    """
    # Ensure 1D array
    probs = np.asarray(probs).reshape(-1)
    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    top_label = labels[top_idx]
    top_prob = float(probs[top_idx])

    if top_prob >= threshold:
        st.error(f"Flagged as {top_label} ({top_prob:.3f})")
    else:
        st.success(f"No toxicity detected ({top_prob:.3f})")

    return top_idx, order


def render_footer(model_name: str, meta_name: str, threshold: float) -> None:
    """Compact footer with model/meta names and threshold."""
    meta_display = meta_name if meta_name else "NOT FOUND"
    st.caption(f"Model: {model_name} • Meta: {meta_display} • Threshold={threshold}")
