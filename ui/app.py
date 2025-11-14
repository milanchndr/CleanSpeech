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
    st.session_state.toxicity_threshold = 0.5



explain_page = st.Page("pages/explain.py", title = "Explain", icon = ":material/text_fields_alt:")
chat_page = st.Page("pages/chat.py", title = "Chat", icon = ":material/robot_2:")

pg = st.navigation([
    chat_page, explain_page
])

st.sidebar.markdown("### Select Model")  
selected_model = st.sidebar.selectbox('Model Selection', options=available_models) 
toxicity_threshold = st.sidebar.slider("Select threshold for toxicity (default = 0.5)", 0.0, 1.0, 0.5, 0.05)

submit_btn = st.sidebar.button('Submit')

if submit_btn:
    st.session_state.model = selected_model
    st.session_state.toxicity_threshold = toxicity_threshold

if __name__ == "__main__" : 
    pg.run()