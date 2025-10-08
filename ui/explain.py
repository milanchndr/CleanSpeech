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

def render():
    st.title("Explain")
    st.caption("This page will show per-label probabilities and token-level contributions. (WIP)")
