# ui/config.py

APP_TITLE = "CleanSpeech — Toxicity Classifier"
APP_CAPTION = "Pick a model • Paste text • Predict"

# Default decision threshold if metadata omits it
DEFAULT_THRESHOLD = 0.5

# Metadata lookup priority:
# 1) <model_stem>_meta.json (next to model)
# 2) baseline_meta.json (project-wide fallback in src/models)
META_FALLBACK_FILENAME = "baseline_meta.json"

# Chart defaults
CHART_HEIGHT = 260

# Feature flags (for later extensions)
ENABLE_EXPLAIN = False  # keep False for now; we’ll add later when needed
