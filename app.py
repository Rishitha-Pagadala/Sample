import streamlit as st
import os
import requests

st.title("HF Chatbot")

api_key = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY"))
if not api_key:
    st.error("Hugging Face API key missing!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You: ")

if st.button("Send") and user_input:
    st.session_state.messages.append(f"You: {user_input}")
    with st.spinner("Generating response..."):
        url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"inputs": user_input, "parameters": {"max_new_tokens": 200}}
        try:
            response = requests.post(url, headers=headers, json=payload).json()
            bot_reply = response[0]["generated_text"]
        except Exception as e:
            bot_reply = f"Error generating response: {e}"

        st.session_state.messages.append(f"Bot: {bot_reply}")

# Display chat
for msg in st.session_state.messages:
    st.write(msg)
