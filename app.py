import streamlit as st
import os
import requests
import time

from huggingface_hub import login

token = st.secrets["HUGGINGFACE_TOKEN"]  
login(token)


st.title("Chatbot")

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
        
        bot_reply = ""
        try:
            response = requests.post(url, headers=headers, json=payload)
            # Check if response is JSON
            if response.headers.get("content-type") == "application/json":
                result = response.json()
                if isinstance(result, dict) and "error" in result:
                    bot_reply = f"Error from API: {result['error']}"
                elif isinstance(result, list) and "generated_text" in result[0]:
                    bot_reply = result[0]["generated_text"]
                else:
                    bot_reply = f"Unexpected API response: {result}"
            else:
                bot_reply = f"Non-JSON response: {response.text}"
        except Exception as e:
            bot_reply = f"Exception: {e}"

        st.session_state.messages.append(f"Bot: {bot_reply}")

# Display chat
for msg in st.session_state.messages:
    st.write(msg)
