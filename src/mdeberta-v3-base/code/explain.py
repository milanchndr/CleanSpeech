# src/mdeberta-v3-base/code/explain.py
"""
Minimal SHAP wrapper for the fine-tuned HF classifier.

Usage (after training):
    from .infer import load_pipeline
    from .explain import explain_text, save_shap_text_plot

    model, tok, device = load_pipeline()
    info = explain_text("You are an absolute idiot!", model, tok)   # dict with tokens + contribs
    out = save_shap_text_plot("You are an absolute idiot!", "src/mdeberta-v3-base/reports/figs/shap_example.png",
                              model, tok)
"""
import os
import shap
import torch
from transformers import TextClassificationPipeline
from .config import default_cfg

def _pipeline(model, tokenizer, device):
    # device index: 0 for CUDA, -1 for CPU
    dev_idx = 0 if (isinstance(device, str) and device == "cuda") or (
        hasattr(device, "type") and device.type == "cuda"
    ) else -1
    return TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        function_to_apply="sigmoid",
        device=dev_idx
    )

def explain_text(text: str, model=None, tokenizer=None, model_dir: str | None = None):
    """Return token-level attributions (mean over labels) and per-label values."""
    if model is None or tokenizer is None:
        from .infer import load_pipeline
        model, tokenizer, device = load_pipeline(model_dir)
    else:
        device = next(model.parameters()).device

    pipe = _pipeline(model, tokenizer, device)
    explainer = shap.Explainer(pipe)  # partition explainer path for transformers
    sv = explainer([text])  # single text
    tokens = sv.data[0]
    per_label_vals = sv.values[0]          # shape: (tokens, labels)
    values_mean = per_label_vals.mean(axis=1).tolist()
    labels_out = [d["label"] for d in pipe(text)[0]]
    return {"tokens": tokens, "values_mean": values_mean, "per_label_values": per_label_vals.tolist(), "labels": labels_out}

def save_shap_text_plot(text: str, out_png: str, model=None, tokenizer=None, model_dir: str | None = None) -> str:
    """Save a SHAP text plot as PNG and return the path."""
    if model is None or tokenizer is None:
        from .infer import load_pipeline
        model, tokenizer, device = load_pipeline(model_dir)
    else:
        device = next(model.parameters()).device

    pipe = _pipeline(model, tokenizer, device)
    explainer = shap.Explainer(pipe)
    sv = explainer([text])

    # Render and save
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.plots.text(sv[0], display=False, show=False)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    return out_png
