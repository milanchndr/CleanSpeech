import streamlit as st

st.set_page_config(page_title="My Project", page_icon="🚀", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "About"])

if page == "Home":
    st.title("🏠 Home")
    st.write("Welcome to the collaborative project!")
    st.info("This is a placeholder for our main app interface.")

elif page == "About":
    st.title("ℹ️ About")
    st.write("This project is a collaboration from group-10-DS-LAB.")
    st.success("UI built with Streamlit, backend to be added soon.")
st.write("Stay tuned for more updates!")