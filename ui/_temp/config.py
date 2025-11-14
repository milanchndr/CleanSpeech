PAGE_CONFIG = {
    "page_title": "CleanSpeech — Toxicity Classifier", 
    "page_icon": "https://fonts.gstatic.com/s/i/materialiconsoutlined/model_training/v6/24px.svg",                     
    "initial_sidebar_state": "auto", 
    "layout" : "wide"
}

MODEL_MAPPER = {
    # "base_line_model": "model_baseline.joblib",
    "advanced_model": "best_model",
}

THRESHOLD = 0.3
GEMINI_MODEL = "gemini-2.5-flash"
F1_THRESHOLD = 0.8