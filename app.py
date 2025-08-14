# app.py
import streamlit as st
import os
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Hugging Face Chatbot", page_icon="🤖")
st.title("Hugging Face Chatbot")

# --- Get Hugging Face API key ---
hf_token = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY"))
if not hf_token:
    st.error(
        "Hugging Face API key not found!\n"
        "Set it in `.streamlit/secrets.toml` or as environment variable `HUGGINGFACE_API_KEY`."
    )
    st.stop()

# Initialize Hugging Face client
client = InferenceClient(hf_token)

# --- Initialize chat session ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- User input ---
user_input = st.text_input("You: ", "")

if st.button("Send") and user_input:
    st.session_state.messages.append(f"You: {user_input}")
    with st.spinner("Generating response..."):
        try:
            # Use invoke() instead of text_generation() for latest huggingface_hub
            response = client.invoke(
                "tiiuae/falcon-7b-instruct",  # HF hosted model
                inputs=user_input,
                parameters={"max_new_tokens": 200}
            )
            bot_reply = response.generated_text
        except Exception as e:
            bot_reply = f"Error generating response: {e}"

        st.session_state.messages.append(f"Bot: {bot_reply}")

# --- Display chat history ---
for msg in st.session_state.messages:
    st.write(msg)
