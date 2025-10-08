# ui/paths.py
from __future__ import annotations
from pathlib import Path

def get_root(current_file: str | None = None) -> Path:
    """
    Return the CleanSpeech project root.
    Assumes this file lives in CleanSpeech/ui/ and project root is one level up.
    """
    if current_file is None:
        # Fallback: resolve relative to this module file
        here = Path(__file__).resolve()
    else:
        here = Path(current_file).resolve()
    return here.parent.parent  # .../CleanSpeech

def get_models_dir(root: Path) -> Path:
    """
    Return the models directory under src/models, ensuring it exists.
    """
    models_dir = root / "src" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
