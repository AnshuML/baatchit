from utils.retriever import hybrid_search
from utils.retriever import hybrid_search
from utils.llm_handler import generate_answer
import streamlit as st
import requests
from markdown import markdown



# Add this function
def render_markdown(text):
    html = markdown(text)
    st.markdown(html, unsafe_allow_html=True)

# Set page config
st.set_page_config(page_title="HR Chatbot", layout="centered")

# Chatbot UI
st.title("💼 HR Policy Chatbot")
st.markdown("Ask anything about your company's HR policies!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get context from hybrid search
    contexts = hybrid_search(prompt)
    context_str = "\n\n".join(contexts)

    # Generate answer
    answer = generate_answer(context_str, prompt)

    # Show response
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": answer})

#st.sidebar.header("🛠️ Admin Panel")
#st.sidebar.markdown("[Open Admin Panel](http://localhost:5001)")