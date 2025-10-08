# ui/inference.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import json
import numpy as np
import joblib
import streamlit as st

try:
    # package mode
    from .config import DEFAULT_THRESHOLD, META_FALLBACK_FILENAME
except ImportError:
    # script mode
    from config import DEFAULT_THRESHOLD, META_FALLBACK_FILENAME


def discover_models(models_dir: Path) -> List[Path]:
    """
    Return a sorted list of available model files (*.joblib) under models_dir.
    """
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.joblib"))


def resolve_meta(model_path: Path, models_dir: Path) -> Tuple[Path, List[Path]]:
    """
    Determine the metadata file for a given model.

    Priority:
      1) <model_stem>_meta.json              (next to the model)
      2) models_dir / META_FALLBACK_FILENAME (project-wide fallback)

    Returns:
      (chosen_meta_path, tried_paths_in_order)

    Raises:
      FileNotFoundError if none exist.
    """
    stem_meta = model_path.with_name(model_path.stem + "_meta.json")
    fallback_meta = models_dir / META_FALLBACK_FILENAME
    tried = [stem_meta, fallback_meta]

    for p in tried:
        if p.exists():
            return p, tried

    raise FileNotFoundError(
        "Metadata JSON not found. Looked for:\n" + "\n".join(f"- {p}" for p in tried)
    )


@st.cache_resource(show_spinner=False)
def load_model(path: Path) -> Any:
    """
    Load a joblib-serialized sklearn Pipeline (or estimator).
    """
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}") from e


@st.cache_resource(show_spinner=False)
def load_meta(path: Path) -> Dict[str, Any]:
    """
    Load and minimally validate metadata JSON.

    Ensures:
      - 'label_cols' exists and is a non-empty list
      - 'threshold' exists (defaults to DEFAULT_THRESHOLD if missing)
    """
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta: Dict[str, Any] = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read metadata from {path}: {e}") from e

    labels = meta.get("label_cols")
    if not isinstance(labels, list) or len(labels) == 0:
        raise KeyError("'label_cols' missing or empty in metadata")

    if "threshold" not in meta:
        meta["threshold"] = DEFAULT_THRESHOLD

    # Normalize types
    meta["label_cols"] = [str(x) for x in labels]
    meta["threshold"] = float(meta["threshold"])

    return meta


def predict_proba(model: Any, text_or_texts: str | Sequence[str]) -> np.ndarray:
    """
    Run model.predict_proba on a single string or a list of strings.

    Returns:
      - For a single string: 1D array of shape (n_labels,)
      - For a list of strings: 2D array of shape (n_samples, n_labels)
    """
    if isinstance(text_or_texts, str):
        X = [text_or_texts]
        probs = model.predict_proba(X)
        # Some sklearn setups return a list of arrays; enforce ndarray
        probs = np.asarray(probs)
        # Shape normalize to 1D for single sample
        if probs.ndim == 2 and probs.shape[0] == 1:
            return probs[0]
        return probs
    else:
        X = list(text_or_texts)
        probs = model.predict_proba(X)
        return np.asarray(probs)
