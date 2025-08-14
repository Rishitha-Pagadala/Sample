# app.py
import streamlit as st
from huggingface_hub import InferenceClient
import os

st.title("Hugging Face Chatbot")

# Get token from environment variable
hf_token = st.secrets["HUGGINGFACE_API_KEY"]
client = InferenceClient(hf_token)

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You: ")

if st.button("Send") and user_input:
    st.session_state.messages.append(f"You: {user_input}")
    with st.spinner("Generating response..."):
        # Replace 'tiiuae/falcon-7b-instruct' with any HF hosted model
        response = client.text_generation(
            model="tiiuae/falcon-7b-instruct",
            inputs=user_input,
            parameters={"max_new_tokens": 200}
        )
    bot_reply = response[0]["generated_text"]
    st.session_state.messages.append(f"Bot: {bot_reply}")

# Display chat
for msg in st.session_state.messages:
    st.write(msg)
