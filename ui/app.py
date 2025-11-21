import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.api_base import get_response
from utils.rewrite_with_ai import rewrite_with_gemini
from _temp.config import PAGE_CONFIG, MODEL_MAPPER, F1_THRESHOLD
from bert_score import score as bert_score
import os
from utils.style import page_style

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for "Classy" & Minimalist look
st.markdown(page_style, unsafe_allow_html=True)

# Initialize Session State
if "model" not in st.session_state:
    st.session_state.model = "advanced_model"

# --- FIX: Initialize toxicity_threshold to prevent AttributeError ---
if "toxicity_threshold" not in st.session_state:
    st.session_state.toxicity_threshold = {
        "toxic": 0.8, "severe_toxic": 0.5, "obscene": 0.7,
        "threat": 0.1, "insult": 0.5, "identity_hate": 0.4
    }

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙ Settings")
    
    st.markdown("*Move the slider to set the sensitivity of the detection*")
    thresholds = {}
    # Now this loop will work because toxicity_threshold is initialized
    for key, default_val in st.session_state.toxicity_threshold.items():
        thresholds[key] = st.slider(f"{key.replace('', ' ').title()}", 0.0, 1.0, float(default_val), 0.05, key=f"slider{key}")
    st.session_state.toxicity_threshold = thresholds
    
    if st.button("Reset Defaults"):
        st.session_state.toxicity_threshold = {
            "toxic": 0.8, "severe_toxic": 0.5, "obscene": 0.7,
            "threat": 0.1, "insult": 0.5, "identity_hate": 0.4
        }
        st.rerun()

# -----------------------------------------------------------------------------
# 3. VISUALIZATION FUNCTIONS
# -----------------------------------------------------------------------------

def process_toxicity_classes(probs):
    """
    Helper to process toxicity probabilities.
    Separates the main 'toxic' score from the specific subtypes.
    """
    # Get overall toxicity (default to 0 if missing)
    overall_score = probs.get("toxic", 0.0)
    
    # Separate subtypes (exclude 'toxic')
    subtypes = {k: v for k, v in probs.items() if k != "toxic"}
    
    # Determine dominant subtype
    if subtypes:
        dom_sub = max(subtypes, key=subtypes.get)
    else:
        dom_sub = "None"
        
    return overall_score, dom_sub, subtypes

def render_colored_tokens(data):
    """
    Renders the sentence with tokens colored by their importance score.
    """
    tokens = data['word_importance']['tokens']
    scores = data['word_importance']['importance_scores']
    
    # Safety: match lengths
    min_len = min(len(tokens), len(scores))
    tokens = tokens[:min_len]
    scores = scores[:min_len]
    
    html_content = """
    <div style="font-family: sans-serif; background-color: white; padding: 20px; border-radius: 10px; color: white;">
    <h3 style="margin-top:0; color: black;">Word Contribution to Toxicity</h3>
    <p style="font-size: 14px; color: #888; margin-bottom: 15px;">
        <span style="color:#ff4b4b;">Red ↑ increases toxicity</span> | 
        <span style="color:#4b9eff;">Blue ↓ decreases/neutralizes</span>
    </p>
    <div style="display: flex; flex-wrap: wrap; gap: 8px; line-height: 1.5;">
    """
    
    max_score = max(max(abs(s) for s in scores), 0.001) # Avoid div by zero
    word_score = []
    for word, score in zip(tokens, scores):
        # Calculate opacity based on score magnitude
        opacity = (abs(score) / max_score)
        
        # Determine Color
        if score > 0: # Toxic (Red)
            bg_color = f"rgba(180, 20, 20, {0.2 + opacity * 0.8})"
        else: # Safe/Neutral (Blue)
            bg_color = f"rgba(20, 100, 220, {0.2 + opacity * 0.8})"
            
        # If score is very close to 0, make it barely visible dark grey
        if abs(score) < 0.01:
             bg_color = "rgba(255, 255, 255, 0.05)"

        word_score.append(f"""
        <div style="
            background-color: {bg_color};
            padding: 5px 10px;
            border-radius: 6px;
            font-weight: 500;
            border: 1px solid rgba(255,255,255,0.1);
            display: inline-block;
        ">{word}</div>
        """)
        
    html_content += "</div></div>"
    return html_content, word_score

def plot_cumulative_toxicity(data):
    tokens = data['word_importance']['tokens']
    scores = data['word_importance']['importance_scores']
    
    # Safety: match lengths
    min_len = min(len(tokens), len(scores))
    tokens = tokens[:min_len]
    scores = scores[:min_len]
    
    # Calculate Cumulative Sum
    cumulative_scores = np.cumsum(scores)
    
    # Create indexed positions (0, 1, 2, ...) instead of using token names
    positions = list(range(len(tokens)))
    
    fig = go.Figure()
    
    # 1. The Line with positions as x-axis
    fig.add_trace(go.Scatter(
        x=positions, 
        y=cumulative_scores,
        mode='lines+markers',
        line=dict(color='#FF4136', width=3, shape='spline'), 
        marker=dict(size=8, color='white', line=dict(width=2, color='#FF4136')),
        name='Cumulative Toxicity',
        customdata=tokens,  # Store token names for hover
        hovertemplate="<b>Position:</b> %{x}<br><b>Word:</b> %{customdata}<br><b>Total Toxicity:</b> %{y:.2f}<extra></extra>"
    ))
    
    # 2. The "Neutral Line" (Zero baseline)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral Baseline")
    
    fig.update_layout(
        title="<b>Cumulative Toxicity Flow</b> (Where did the sentence turn toxic?)",
        yaxis_title="Cumulative Toxicity Score",
        xaxis_title="Sequence of Words (Position)",
        template="plotly_white",
        height=450,
        hovermode="x unified"
    )
    return fig

def plot_toxicity_dashboard(data):
    if not data:
        return None

    # --- PREPARE DATA ---
    probs = data['probabilities']
    # FIX: Use the new logic to separate overall from subtypes
    overall_score, dom_sub, subtypes = process_toxicity_classes(probs)
    
    # --- 1. SORTING LOGIC ---
    sorted_subtypes = sorted(subtypes.items(), key=lambda x: x[1])
    
    # Unpack into lists for Plotly
    subtype_names = [x[0] for x in sorted_subtypes]
    subtype_vals = [x[1] for x in sorted_subtypes]

    # --- FIGURE SETUP ---
    fig_dashboard = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'xy'}]],
        column_widths=[0.35, 0.65], 
        subplot_titles=("Overall Toxicity Probability", "Specific Category Breakdown")
    )

    # --- A. GAUGE CHART (Main Label) ---
    fig_dashboard.add_trace(go.Indicator(
        mode = "gauge+number",
        value = overall_score * 100,
        number = {'suffix': "%"},
        title = {'text': "IS IT TOXIC?", 'font': {'size': 16}},
        domain = {'x': [0, 0.5], 'y': [0.2, 1]},  # Add top margin with y=[0.2, 1]
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if overall_score > 0.5 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "#f0f2f6"},
                {'range': [50, 100], 'color': "#ffe5e5"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ), row=1, col=1)

    # --- B. BAR CHART (Sorted & Locked) ---
    
    colors = ['#FF4136' if name == dom_sub and val > 0.01 else '#cccccc' for name, val in zip(subtype_names, subtype_vals)]
    
    fig_dashboard.add_trace(go.Bar(
        x=subtype_vals,
        y=subtype_names,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1%}" for v in subtype_vals],
        textposition='auto', 
        name="Subtypes"
    ), row=1, col=2)

    # --- LAYOUT UPDATES ---
    fig_dashboard.update_layout(
        template="plotly_white",
        height=400,
        title_text=f"<b>Toxicity Analysis</b> | Dominant Category: <span style='color:red'>{dom_sub.upper()}</span>",
        showlegend=False
    )
    
    # --- 2. LOCK AXIS LOGIC ---
    fig_dashboard.update_xaxes(range=[0, 1.05], row=1, col=2, showgrid=True) 

    return fig_dashboard

def plot_word_importance(data):
    tokens_raw = data['word_importance']['tokens']
    importance_raw = data['word_importance']['importance_scores']
    
    min_len = min(len(tokens_raw), len(importance_raw))
    
    combined = sorted(zip(tokens_raw[:min_len], importance_raw[:min_len]), key=lambda x: x[1])
    tokens_sorted, importance_sorted = zip(*combined)
    
    lolli_colors = ['#FF4136' if x > 0 else '#1f77b4' for x in importance_sorted]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance_sorted, y=tokens_sorted, orientation='h',
        marker=dict(color=lolli_colors), width=0.05, hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=importance_sorted, y=tokens_sorted, mode='markers',
        marker=dict(color=lolli_colors, size=12, line=dict(color='white', width=1)),
        hovertemplate="<b>Word:</b> %{y}<br><b>Impact:</b> %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title="<b>Word Importance Ranking</b>", xaxis_title="Impact Score",
        template="plotly_white", height=400, showlegend=False
    )
    return fig

def plot_attention_arcs(data):
    tokens = data['word_importance']['tokens']
    importance = np.array(data['word_importance']['importance_scores'])
    min_len = min(len(tokens), len(importance))
    tokens = tokens[:min_len]
    importance = importance[:min_len]
    n_tokens = len(tokens)

    fig = go.Figure()
    
    node_colors = ["rgba(255,0,0,0.8)" if i > 0.05 else "rgba(100,100,100,0.3)" for i in importance]
    fig.add_trace(go.Scatter(
        x=list(range(n_tokens)), y=[0]*n_tokens, mode='text+markers',
        text=tokens, textposition="bottom center", textfont=dict(size=14),
        marker=dict(size=10, color=node_colors, symbol="square"),
        hoverinfo='text', hovertext=[f"{t}: {s:.4f}" for t, s in zip(tokens, importance)]
    ))

    max_imp_idx = np.argmax(np.abs(importance)) if len(importance) > 0 else 0
    shapes = []
    for i in range(n_tokens):
        if i == max_imp_idx or abs(importance[i]) < 0.001: continue
        x0, x1 = max_imp_idx, i
        cx = (x0 + x1) / 2
        cy = 0.5 + (abs(x1 - x0) * 0.1)
        color = "rgba(255, 50, 50, 0.6)" if (importance[i] > 0 and importance[max_imp_idx] > 0) else "rgba(150, 150, 150, 0.3)"
        shapes.append(dict(type="path", path=f"M {x0} 0 Q {cx} {cy} {x1} 0", line=dict(color=color, width=2)))

    fig.update_layout(
        title="<b>Attention Arcs</b>", shapes=shapes, showlegend=False,
        template="plotly_white", height=300, 
        xaxis=dict(visible=False, range=[-1, n_tokens]), yaxis=dict(visible=False, range=[-0.5, 2])
    )
    return fig

def plot_attention_heatmap(data):
    if "attention" in data:
        att = np.array(data["attention"]["matrix"])
        tok = data["attention"]["tokens"]
        text_values = np.round(att, 2).astype(str)
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=att, x=tok, y=tok, colorscale="gray", reversescale=True,
                zmin=0, zmax=att.max(), text=text_values, texttemplate="%{text}",
                textfont={"size": 10, "color": "black"}
            )
        )
        fig_heatmap.update_layout(
            title="Attention Matrix (Gray Scale)", xaxis_title="Tokens", yaxis_title="Tokens",
            template="plotly_white"
        )
        return fig_heatmap
    return None

# -----------------------------------------------------------------------------
# 4. MAIN UI LAYOUT
# -----------------------------------------------------------------------------

# Banner
if os.path.exists("assets/banner.png"):
    st.image("assets/banner.png", width=400)
else:
    st.title("CleanSpeech")

st.write("Type a message below to analyze toxicity, visualize model attention, and generate constructive alternatives.")
# INPUT
with st.container():
    user_input = st.text_area("", placeholder="Enter text to analyze e.g. Would you both shut up, you don't run Wikipedia, especially a stupid kid", height=100, label_visibility="collapsed", key="analyze_input")
    analyze_btn = st.button("Analyze", type="primary")


# EXECUTION
if analyze_btn and user_input:
    with st.spinner("Processing text and generating explanations..."):
        try:
            api_data = get_response(user_input)
            st.session_state.api_response = api_data 
            # Note: We pass just the probabilities dict to rewrite_with_gemini
            rewritten_text = rewrite_with_gemini(user_input, api_data["probabilities"])
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
            st.stop()

    # 1. REWRITE (Primary Output)
    # Check if toxic using the main label
    is_toxic = api_data["final_label"] == "toxic"
    
    if is_toxic:
        st.markdown("### 🛡 Constructive Rewrite")
        st.info(f"*Rewrite:* {rewritten_text}")
    else:
        st.success("✅ Your message is clean and does not need rewriting.")
    
    st.markdown("---")

    # 2. EXPLAINABILITY (Secondary Output)
    with st.expander("View Analysis Details", expanded=True):
        
        # A. TOXICITY DASHBOARD (Gauge + Bars)
        fig_dashboard = plot_toxicity_dashboard(api_data)
        if fig_dashboard:
            st.plotly_chart(fig_dashboard, use_container_width=True)
        
        st.markdown("---")

        # B. TOKEN ANALYSIS (Dark Mode Style)
        # st.subheader("Token Contribution Analysis")
        html_tokens, word_score = render_colored_tokens(api_data)
        st.markdown(html_tokens, unsafe_allow_html=True)
        # Combine all word score divs into a single line
        combined_tokens = "".join(word_score)
        st.markdown(combined_tokens, unsafe_allow_html=True)
        
        st.write("")
        
        # C. DETAILED TABS
        subtab1, subtab2, subtab3 = st.tabs(["Cumulative Flow", "Word Importance", "Attention Arcs"])

        with subtab1:
            st.markdown("*Cumulative Toxicity Flow*")
            fig_cum = plot_cumulative_toxicity(api_data)
            st.plotly_chart(fig_cum, use_container_width=True)

        with subtab2:
            st.markdown("*Word Importance Score for toxicity detection*")
            fig_lolli = plot_word_importance(api_data)
            st.plotly_chart(fig_lolli, use_container_width=True)

        with subtab3:
            st.markdown("*the relationship of the most toxic words to other words(red shows strong relationship and grey shows weak relationship)*")
            fig_arcs = plot_attention_arcs(api_data)
            st.plotly_chart(fig_arcs, use_container_width=True)

        # with subtab4:
        #     st.markdown("*Attention Heatmap*")
        #     fig_heatmap = plot_attention_heatmap(api_data)
        #     if fig_heatmap:
        #         st.plotly_chart(fig_heatmap, use_container_width=True)
        #     else:
        #         st.info("Attention matrix not available for this model.")
        
    st.markdown("---")
    with st.expander("Raw Data JSON"):
        st.json(api_data, expanded=False)

elif not user_input:
    st.info("👋 Welcome to CleanSpeech. Enter text above to begin analysis.")