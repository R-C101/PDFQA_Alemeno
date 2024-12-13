import streamlit as st
import os
import time
from helper import (
    get_answer,
    load_local_llm,
    load_local_embeddings
)

# Streamlit UI
st.title("Intelligent PDF Q&A Application")
st.sidebar.header("Model Configuration")
embedding_model = st.sidebar.selectbox(
    "Select Embedding Model", 
    [
        'sentence-transformers/all-MiniLM-L6-v2', 
        'sentence-transformers/all-mpnet-base-v2'
    ]
)
llm_model = st.sidebar.selectbox(
    "Select LLM Model", 
    [
        'meta-llama/Llama-2-7b-chat-hf', 
        'facebook/opt-350m',
        'google/flan-t5-base'
    ]
)

# Load local LLM
try:
    llm = load_local_llm(llm_model)
except Exception as e:
    st.error(f"Failed to load LLM: {e}")
    st.error("Ensure you have the necessary model access and dependencies")
    llm = None

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Response generator function
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Chat input
if question := st.chat_input("Ask a question about your documents"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    if question and llm:
        try:
            # Get answer using RAG chain
            response, docs, category = get_answer(question, './chroma_dbs', llm, embedding_model)

            # Display assistant response
            with st.chat_message("assistant"):
                full_response = st.write_stream(response_generator(response))

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Show additional information
            st.info(f"Documents retrieved from {category.capitalize()} category")

            # Optional: Show retrieved documents
            with st.expander("Retrieved Document Details"):
                st.write(docs)

        except Exception as e:
            st.error(f"Error generating response: {e}")

# Sidebar information
st.sidebar.info("""
### Instructions
 Ask a question, and the app will automatically:
   - Detect the most relevant document category
   - Index and search through the documents
   - Provide an answer
""")