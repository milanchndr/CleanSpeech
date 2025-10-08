# ui/charts.py
from __future__ import annotations

import numpy as np
import pandas as pd
import altair as alt


def probability_bar_chart(labels: list[str], probs: np.ndarray, height: int = 260) -> alt.Chart:
    """
    Build a horizontal bar chart of label probabilities (descending).
    Returns an Altair Chart; caller renders with st.altair_chart(chart, use_container_width=True).
    """
    probs = np.asarray(probs).reshape(-1)
    df = (
        pd.DataFrame({"label": labels, "probability": probs})
        .sort_values("probability", ascending=False, ignore_index=True)
    )
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("label:N", sort="-x", title="Label"),
            x=alt.X("probability:Q", title="Probability"),
            tooltip=["label", alt.Tooltip("probability:Q", format=".3f")],
        )
        .properties(height=height)
    )
    return chart
