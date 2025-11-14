import streamlit as st
from utils.api_base import get_response
from utils.rewrite_with_ai import rewrite_with_gemini
from bert_score import score as bert_score
from _temp.config import F1_THRESHOLD

st.title("CleanSpeech — Toxicity Classifier")
st.write("A simple chat interface to detect and remove toxicity.")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Pick a model • Paste text • Predict toxicity"}
    ]


def get_bot_reply(user_text: str) -> str:
    """Placeholder bot reply. Replace with model inference integration.

    Args:
        user_text: the user's message

    Returns:
        A string reply from the bot.
    """
    api_response = get_response(user_text)
    st.session_state.api_response = api_response

    response = rewrite_with_gemini(user_text, api_response["probabilities"])
    P, R, F1 = bert_score([response], [user_text], lang="en", verbose=False)
    return response, F1 


col1, col2 = st.columns([8, 2])
with col1:
    st.caption(f"• selected model {st.session_state.get('model', 'Model A')}")
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.current_message = None
        st.session_state.api_response = None
        st.rerun()


def render_messages():
    use_chat = hasattr(st, "chat_message")
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "system":
            # system messages are hidden or lightly shown
            st.info(content)
            continue

        if use_chat:
            try:
                with st.chat_message(role):
                    st.write(content)
            except Exception:
                # Fallback if chat_message signature differs
                st.write(f"**{role.capitalize()}:** {content}")
        else:
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Bot:** {content}")


render_messages()


with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Your message", height=120, key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input and user_input.strip():
    # append user message
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    st.session_state.current_message = user_input.strip()

    # get bot reply (replace this with real inference call)
    F1, counter = -1, 1
    while F1 < F1_THRESHOLD : 
        reply, F1 = get_bot_reply(user_input.strip()) 
        print(f"**BERTScore F1:** {F1.mean().item():.4f} try {counter}")
        counter += 1
    
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # rerun so the new messages are shown (keeps form cleared because of clear_on_submit)
    st.rerun()
