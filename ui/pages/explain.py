import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.api_base import get_response

def score_to_color(score, min_score, max_score): 
    norm = (score - min_score) / (max_score - min_score)
    return norm


st.title("CleanSpeech — Model Explanations")
st.write("Understand how our hate speech detection model makes predictions")

def explain_prediction(user_text: str = ""): 
    with st.spinner("Calling backend API..."):
        try:
            # response = requests.post(API_URL, json={"text": user_input})
            # data = response.json()
            if not st.session_state.api_response:
                data = get_response(user_text)
            else : 
                data = st.session_state.api_response

            st.success("✅ Analysis complete!")
        except Exception as e:
            st.error(f"❌ Error calling API: {e}")
            st.stop()

    # ----------------------------------------------------
    # 1️⃣ INPUT TEXT
    # ----------------------------------------------------
    st.subheader("Original Input")
    st.write(data["input_text"])

    # ----------------------------------------------------
    # 2️⃣ TOXICITY PROBABILITIES
    # ----------------------------------------------------
    st.subheader("Toxicity Probabilities")
    probs = pd.DataFrame(list(data["probabilities"].items()), columns=["Label", "Probability"])
    st.plotly_chart(px.bar(probs, x="Label", y="Probability", title="Toxicity Probabilities"), use_container_width=True)

    # Compute final labels dynamically
    final_labels = {k: int(v >= st.session_state.toxicity_threshold[k]) for k, v in data["probabilities"].items()}

    # ----------------------------------------------------
    # 3️⃣ FINAL LABELS
    # ----------------------------------------------------
    st.subheader("Final Classification Labels (Based on Threshold)")
    cols = st.columns(len(final_labels))
    for i, (label, val) in enumerate(final_labels.items()):
        color = "#ef4444" if val == 1 else "#22c55e"
        cols[i].markdown(
            f"<div style='background:{color};color:white;padding:6px;border-radius:6px;text-align:center'>{label}: {'Toxic' if val==1 else 'Safe'}</div>",
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------
    # 4️⃣ WORD IMPORTANCE VISUALIZATION
    # ----------------------------------------------------
    st.subheader("Word Importance (Explainability)")
    imp = data["word_importance"]
    tokens = imp["tokens"]
    scores = imp["importance_scores"]

    # Colored tokens (HTML)
    st.markdown("**Colored Tokens:** (Red ↑ increases toxicity, Blue ↓ decreases toxicity)")
    colored_text = ""
    for tok, score in zip(tokens, scores):
        red = int(255 * max(score, 0))
        blue = int(255 * abs(min(score, 0)))
        colored_text += f"<span style='background-color:rgba({red},0,{blue},0.4);padding:4px;margin:2px;border-radius:4px'>{tok}</span> "
    st.markdown(colored_text, unsafe_allow_html=True)

    # --------- Plotly Bar Chart (Word Importance) ---------
    colors = ["#ef4444" if s > 0 else "#3b82f6" for s in scores]
    fig_importance = go.Figure(data=go.Bar(x=tokens, y=scores, marker_color=colors))
    fig_importance.update_layout(
        title="Word Importance Scores",
        xaxis_title="Tokens",
        yaxis_title="Importance Score",
        template="plotly_white"
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    # ----------------------------------------------------
    # 5️⃣ CUMULATIVE IMPACT CURVE
    # ----------------------------------------------------
    st.subheader("Cumulative Impact on Toxicity Prediction")
    cumulative = np.cumsum(scores) + imp["base_value"]

    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(
        x=list(range(len(tokens))),
        y=cumulative,
        mode="lines+markers",
        text=tokens
    ))
    fig_cumulative.update_layout(
        title="Cumulative Contribution to Toxicity",
        xaxis=dict(tickvals=list(range(len(tokens))), ticktext=tokens),
        yaxis_title="Model Output Value",
        template="plotly_white"
    )
    st.plotly_chart(fig_cumulative, use_container_width=True)

    # ----------------------------------------------------
    # 6️⃣ ATTENTION HEATMAP (Gray Scale, Plotly)
    # ----------------------------------------------------
    if "attention" in data:
        st.subheader("Attention Heatmap")

        att = np.array(data["attention"]["matrix"])
        tok = data["attention"]["tokens"]
        text_values = np.round(att, 2).astype(str)
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=att,
                x=tok,
                y=tok,
                colorscale="gray",   
                reversescale=True,
                zmin=0,
                zmax=att.max(), 
                text=text_values,                  # <-- text for each cell
                texttemplate="%{text}",            # <-- display text
                textfont={"size": 10, "color": "black"}  # <-- cell text style
            )
        )
        fig_heatmap.update_layout(
            title="Attention Matrix (Gray Scale)",
            xaxis_title="Tokens",
            yaxis_title="Tokens",
            template="plotly_white"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ----------------------------------------------------
    # 7️⃣ SUMMARY TABLE
    # ----------------------------------------------------
    st.subheader("Summary Table")
    summary_df = pd.DataFrame({
        "Label": list(data["probabilities"].keys()),
        "Probability": list(data["probabilities"].values()),
        "Final Label": ["Toxic" if v == 1 else "Safe" for v in final_labels.values()]
    })
    st.dataframe(summary_df, use_container_width=True)


if st.session_state.current_message:
    st.markdown("### Current Message")
    explain_prediction(st.session_state.current_message)

    clear_chat = st.button("Clear Current Message")
    if clear_chat: 
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.current_message = None
        st.session_state.api_response = None
        st.rerun()
else:
    input_text = st.text_area(
        "Enter text to explain model prediction:", key="explain_input",
        placeholder="Type your text here...",
    )

    predict_btn = st.button("Explain Prediction")
    if predict_btn: 
        explain_prediction(input_text) 