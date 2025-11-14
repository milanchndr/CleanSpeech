import streamlit as st 
from _temp.config import PAGE_CONFIG, MODEL_MAPPER
st.set_page_config(**PAGE_CONFIG)  

available_models = list(MODEL_MAPPER.keys())
if "model" not in st.session_state:
    st.session_state.model = available_models[0]

if "current_message" not in st.session_state:
    st.session_state.current_message = None
    
if "api_response" not in st.session_state:
    st.session_state.api_response = None

if "toxicity_threshold" not in st.session_state:
    st.session_state.toxicity_threshold = {
        "toxic" :0.8,
        "severe_toxic":0.5,
        "obscene":0.7,
        "threat":0.1,
        "insult":0.5,
        "identity_hate":0.4
    }



explain_page = st.Page("pages/explain.py", title = "Explain", icon = ":material/text_fields_alt:")
chat_page = st.Page("pages/chat.py", title = "Chat", icon = ":material/robot_2:")

pg = st.navigation([
    chat_page, explain_page
])

st.sidebar.markdown("### Select Model")  
selected_model = st.sidebar.selectbox('Model Selection', options=available_models) 
submit_btn = st.sidebar.button('Submit')

toxic_threshold = st.sidebar.slider(
    f"Select threshold for toxic (default = {st.session_state.toxicity_threshold["toxic"]})", 0.0, 1.0, 0.8, 0.05
)
severe_toxic_threshold = st.sidebar.slider(
    f"Select threshold for severe toxic (default = {st.session_state.toxicity_threshold["severe_toxic"]})", 0.0, 1.0, 0.5, 0.05
)
obscene_threshold = st.sidebar.slider(
    f"Select threshold for obscene (default = {st.session_state.toxicity_threshold["obscene"]})", 0.0, 1.0, 0.7, 0.05
)
threat_threshold = st.sidebar.slider(
    f"Select threshold for threat (default = {st.session_state.toxicity_threshold["threat"]})", 0.0, 1.0, 0.1, 0.05
)
insult_threshold = st.sidebar.slider(
    f"Select threshold for insult (default = {st.session_state.toxicity_threshold["insult"]})", 0.0, 1.0, 0.5, 0.05
)
identity_hate_threshold = st.sidebar.slider(
    f"Select threshold for identity_hate (default = {st.session_state.toxicity_threshold["identity_hate"]})", 0.0, 1.0, 0.4, 0.05
)
st.session_state.toxicity_threshold = {
    "toxic" : toxic_threshold,
    "severe_toxic": severe_toxic_threshold,
    "obscene": obscene_threshold,
    "threat": threat_threshold,
    "insult": insult_threshold,
    "identity_hate": identity_hate_threshold
}

if submit_btn:
    st.session_state.model = selected_model

if __name__ == "__main__" : 
    pg.run()